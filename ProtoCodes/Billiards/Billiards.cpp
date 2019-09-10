#include<iostream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/features2d.hpp>

using namespace std;
using namespace cv;

const double X=472.5;//作为球路计算算法测试的仿真程序，本程序没有引入白球探测程序，我们把测试图片中的白球坐标作为常量定义于此
const double Y=391.5;

double min(double x,double y){//辅助函数，找到较小的值
    return x>y?y:x;
}

double max(double x,double y){//辅助函数，找到较大的值
    return x>y?x:y;
}

void minmax(double& x,double& y){//将两个数按从小到大的方式重新排序
    if(x>y){
        double tmp;
        tmp=x;
        x=y;
        y=tmp;
    }
}

bool backgroundColorExtraction(const Mat& im,int& r,int& g,int& b,bool centerEmphasized=true){//背景色提取
//将提取得到的背景色信息保存到变量r,g,b中，为了减少台球桌外侧的边缘对于背景色统计的干扰，可以选择“强调中心区域”
    if(im.channels()!=3)return false;//只处理彩色图片
    int rows=im.rows;
    int cols=im.cols;
    int RGBHistogram[16][16][16];//将RGB色彩空间划分为16*16*16个区间
    memset(RGBHistogram,0,sizeof(int)*16*16*16);
    r=0;g=0;b=0;
    int cnt=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){//遍历图像
            RGBHistogram[int(im.at<Vec3b>(i, j)[0]) / 16][int(im.at<Vec3b>(i, j)[1]) / 16][int(im.at<Vec3b>(i, j)[2]) / 16]++;//计算每个像素的RGB值，将其归入相应的直方图区间
            if(centerEmphasized && i>rows/4 && i<rows/4*3 && j>cols/4 && j<cols/4*3){
                RGBHistogram[int(im.at<Vec3b>(i, j)[0]) / 16][int(im.at<Vec3b>(i, j)[1]) / 16][int(im.at<Vec3b>(i, j)[2]) / 16]++;//如果选择“强调中心区域”，则中部区域的色彩统计按照一个像素两票计算
            }
        }
    }
    for(int i=0;i<16;i++){//在直方图中找到RGB最大值
        for(int j=0;j<16;j++){
            for(int k=0;k<16;k++){
                if (RGBHistogram[i][j][k] > cnt) {
					cnt = RGBHistogram[i][j][k];
					r = i;
					g = j;
					b = k;
				}
            }
        }
    }
    r=r*16+8;//通过数组标号反过来估计最大RGB值，即背景色
    g=g*16+8;
    b=b*16+8;
    return true;
}

Mat binarization(const Mat& src,const int r,const int g,const int b,const int threshold,bool OR=true){//图像二值化，按照背景色r,g,b,以threshold为阈值，将图像src中与(r,g,b)差异小于阈值的像素点绘制为黑色，否则绘制为白色
    Mat dst(Size(src.cols, src.rows), CV_8UC3);
    if(OR){//或逻辑：r,g,b之一的差异超过阈值就绘制白色
        for(int i=0;i<src.rows;i++){
            for(int j=0;j<src.cols;j++){
                if( abs(src.at<Vec3b>(i,j)[0]-r)>threshold || abs(src.at<Vec3b>(i,j)[1]-g)>threshold || abs(src.at<Vec3b>(i,j)[2]-b)>threshold){
                    dst.at<Vec3b>(i, j)[0] = 255;
                    dst.at<Vec3b>(i, j)[1] = 255;
                    dst.at<Vec3b>(i, j)[2] = 255;
                }
                else{
                    dst.at<Vec3b>(i, j)[0] = 0;
                    dst.at<Vec3b>(i, j)[1] = 0;
                    dst.at<Vec3b>(i, j)[2] = 0;
                }
            }
        }
    }
    else{//与逻辑：r,g,b差异全部大于阈值才绘制为白色
        for(int i=0;i<src.rows;i++){
            for(int j=0;j<src.cols;j++){
                if( abs(src.at<Vec3b>(i,j)[0]-r)>threshold && abs(src.at<Vec3b>(i,j)[1]-g)>threshold && abs(src.at<Vec3b>(i,j)[2]-b)>threshold){
                    dst.at<Vec3b>(i, j)[0] = 255;
                    dst.at<Vec3b>(i, j)[1] = 255;
                    dst.at<Vec3b>(i, j)[2] = 255;
                }
                else{
                    dst.at<Vec3b>(i, j)[0] = 0;
                    dst.at<Vec3b>(i, j)[1] = 0;
                    dst.at<Vec3b>(i, j)[2] = 0;
                }
            }
        }
    }
    imwrite("tmp.jpg",dst);
    return dst;
}

Mat edgeDetection(const Mat& src,unsigned int Modifier=1){//辅助函数，生成Sobel边缘探测图像
    Mat gray,grad_x,grad_y,grad;
    cvtColor(src,gray,COLOR_RGB2GRAY);
    Sobel(gray,grad_x,CV_16S,1,0,3,1,0,BORDER_DEFAULT);//由x方向上的灰度梯度得到边缘图
    Sobel(gray,grad_y,CV_16S,0,1,3,1,0,BORDER_DEFAULT);//由y方向上的灰度梯度得到边缘图
    convertScaleAbs(grad_x,grad_x);
    convertScaleAbs(grad_y,grad_y);
    addWeighted(grad_x,0.5*Modifier,grad_y,0.5*Modifier,0,grad);//将两个边缘图加权相加，用参数Modifier控制图像中边缘轮廓的强度
    return grad;
}

Mat findTableEdge(const Mat& src,int& x1,int& x2,int& y1,int& y2,int marginFilter=0,int lowerBound=20,int upperBound=40,int division=40){
    //探测桌面边缘的算法
    //将计算得到的桌面边缘探测结果保存到变量x1,x2,y1,y2中
    //如果开启marginFilter=n，则将图像边缘的1/n*图像长、1/n*图像宽的区域涂黑，如果marginFilter=0，则保留原图像，不做涂黑处理
    //桌面边缘处的颜色为绿色，用Sobel算子只能提取出很弱的纹理
    //本函数的目的就是在二值化图像的黑色区域搜寻Sobel算子提取的弱纹理
    //将球-桌面间的强纹理信息弱化，将桌面本身的弱纹理信息强化
    //然后运用此信息进行桌面边缘的位置估计
    //用lowerBound和upperBound确定提取的弱纹理的强度阈值
    //将图像长、宽划分为division部分，统计落在其中的纹理像素，用于估计桌面边缘的大致位置
    int r,g,b;
    Mat dst(Size(src.cols, src.rows), CV_8UC3);
    backgroundColorExtraction(src,r,g,b);//提取背景色
    Mat im1=binarization(src,r,g,b,10);//以与背景色差异阈值为10的标准进行图像二值化
    //对于一般目的而言，该阈值在经验上可以设置为50，对于桌面边缘提取而言，把阈值减小可以缩小黑色区域，减少探测到球-桌面强纹理信息的可能性，过滤掉一部分强纹理
    Mat im2=edgeDetection(src);
    if(marginFilter==0){//不对边缘进行处理
        for(int i=0;i<src.rows;i++){
            for(int j=0;j<src.cols;j++){
                if(im1.at<Vec3b>(i,j)[0]==0 && im1.at<Vec3b>(i,j)[1]==0 && im1.at<Vec3b>(i,j)[2]==0 && im2.at<uchar>(i,j)>lowerBound && im2.at<uchar>(i,j)<upperBound){//提取绿色桌面上的弱纹理，将其绘制为白色
                    dst.at<Vec3b>(i, j)[0] = 255;
                    dst.at<Vec3b>(i, j)[1] = 255;
                    dst.at<Vec3b>(i, j)[2] = 255;
                }
                else{
                    dst.at<Vec3b>(i, j)[0] = 0;
                    dst.at<Vec3b>(i, j)[1] = 0;
                    dst.at<Vec3b>(i, j)[2] = 0;
                }
            }
        }
    }
    else{//如果marginFilter不等于0，则先进行边缘处理
        for(int i=0;i<int(src.rows/marginFilter);i++){
            for(int j=0;j<src.cols;j++){
                dst.at<Vec3b>(i, j)[0] = 0;
                dst.at<Vec3b>(i, j)[1] = 0;
                dst.at<Vec3b>(i, j)[2] = 0;
            }
        }
        for(int i=int(src.rows-src.rows/marginFilter);i<src.rows;i++){
            for(int j=0;j<src.cols;j++){
                dst.at<Vec3b>(i, j)[0] = 0;
                dst.at<Vec3b>(i, j)[1] = 0;
                dst.at<Vec3b>(i, j)[2] = 0;
            }
        }
        for(int i=int(src.rows/marginFilter);i<int(src.rows-src.rows/marginFilter);i++){
            for(int j=0;j<int(src.cols/marginFilter);j++){
                dst.at<Vec3b>(i, j)[0] = 0;
                dst.at<Vec3b>(i, j)[1] = 0;
                dst.at<Vec3b>(i, j)[2] = 0;
            }
        }
        for(int i=int(src.rows/marginFilter);i<int(src.rows-src.rows/marginFilter);i++){
            for(int j=int(src.cols-src.cols/marginFilter);j<src.cols;j++){
                dst.at<Vec3b>(i, j)[0] = 0;
                dst.at<Vec3b>(i, j)[1] = 0;
                dst.at<Vec3b>(i, j)[2] = 0;
            }
        }
        for(int i=int(src.rows/marginFilter);i<int(src.rows-src.rows/marginFilter);i++){//然后再提取弱纹理信息
            for(int j=int(src.cols/marginFilter);j<int(src.cols-src.cols/marginFilter);j++){
                if(im1.at<Vec3b>(i,j)[0]==0 && im1.at<Vec3b>(i,j)[1]==0 && im1.at<Vec3b>(i,j)[2]==0 && im2.at<uchar>(i,j)>lowerBound && im2.at<uchar>(i,j)<upperBound){
                    dst.at<Vec3b>(i, j)[0] = 255;
                    dst.at<Vec3b>(i, j)[1] = 255;
                    dst.at<Vec3b>(i, j)[2] = 255;
                }
                else{
                    dst.at<Vec3b>(i, j)[0] = 0;
                    dst.at<Vec3b>(i, j)[1] = 0;
                    dst.at<Vec3b>(i, j)[2] = 0;
                }
            }
        }
    }
    int row[dst.rows/division];//在每个长宽区间中统计纹理像素点的数目
    int col[dst.cols/division];
    memset(row,0,sizeof(int)*dst.rows/division);
    memset(col,0,sizeof(int)*dst.cols/division);
    for(int i=0;i<dst.rows;i++){
        for(int j=0;j<dst.cols;j++){
            if(dst.at<Vec3b>(i,j)[0]==255){
                row[i/division]++;
                col[j/division]++;
            }
        }
    }
    int maxI1=0,maxI2=0,maxJ1=0,maxJ2=0;//寻找出现纹理像素点最多的位置
    int i1=-1,i2=-1,j1=-1,j2=-1;
    for(int i=0;i<dst.rows/4/division;i++){
        if(row[i]>maxI1){
            maxI1=row[i];
            i1=i;
        }
    }
    for(int i=dst.rows*3/4/division;i<dst.rows/division;i++){
        if(row[i]>maxI2){
            maxI2=row[i];
            i2=i;
        }
    }
    for(int j=0;j<dst.cols/4/division;j++){
        if(col[j]>maxJ1){
            maxJ1=col[j];
            j1=j;
        }
    }
    for(int j=dst.cols*3/4/division;j<dst.cols/division;j++){
        if(col[j]>maxJ2){
            maxJ2=col[j];
            j2=j;
        }
    }
    x1=j1*division+division/2;//得到对于桌面边缘位置的估计
    x2=j2*division+division/2;
    y1=i1*division+division/2;
    y2=i2*division+division/2;
    return dst;
}

int WValue(Mat src,int y,int x,bool horizontal,int length){//计算水平或竖直连续length个像素的“白色值”——whiteValue
    int r=0;
    if(horizontal){
        for(int i=0;i<length;i++){
            r+=src.at<Vec3b>(y,x+i)[0];
            r+=src.at<Vec3b>(y,x+i)[1];
            r+=src.at<Vec3b>(y,x+i)[2];
        }
        r/=length;
        r/=3;
    }
    else{
        for(int i=0;i<length;i++){
            r+=src.at<Vec3b>(y+i,x)[0];
            r+=src.at<Vec3b>(y+i,x)[1];
            r+=src.at<Vec3b>(y+i,x)[2];
        }
        r/=length;
        r/=3;
    }
    return r;
}

void drawReflections(double startX,double startY,double endX,double endY,double x1,double x2,double y1,double y2,double& startX1,double& startY1,double& endX1,double& endY1){
    //x1 x2 y1 y2表示桌面边缘参数，startX startY endX endY表示初始线段的位置与定向，startX1 startY1 endX1 endY1表示用于下一步迭代的线段位置与定向
    double dx=endX-startX;
    double dy=endY-startY;
    if(dx<0 && dy<0){
        double soly=(x1-endX)*(startY-endY)/(startX-endX)+endY;
        double solx=(y1-endY)*(startX-endX)/(startY-endY)+endX;
        if(soly<y1){
            endX1=solx;
            endY1=y1;
            startX1=startX;
            startY1=2*y1-startY;
            return;
        }
        else{
            endX1=x1;
            endY1=soly;
            startX1=2*x1-startX;
            startY1=startY;
            return;
        }
    }
    else if(dx<0 && dy>0){
        double soly=(x1-endX)*(startY-endY)/(startX-endX)+endY;
        double solx=(y2-endY)*(startX-endX)/(startY-endY)+endX;
        if(soly>y2){
            endX1=solx;
            endY1=y2;
            startX1=startX;
            startY1=2*y2-startY;
            return;
        }
        else{
            endX1=x1;
            endY1=soly;
            startX1=2*x1-startX;
            startY1=startY;
            return;
        }
    }
    else if(dx>0 && dy<0){
        double soly=(x2-endX)*(startY-endY)/(startX-endX)+endY;
        double solx=(y1-endY)*(startX-endX)/(startY-endY)+endX;
        if(soly<y1){
            endX1=solx;
            endY1=y1;
            startX1=startX;
            startY1=2*y1-startY;
            return;
        }
        else{
            endX1=x2;
            endY1=soly;
            startX1=2*x2-startX;
            startY1=startY;
            return;
        }
    }
    else if(dx>0 && dy>0){
        double soly=(x2-endX)*(startY-endY)/(startX-endX)+endY;
        double solx=(y2-endY)*(startX-endX)/(startY-endY)+endX;
        if(soly>y2){
            endX1=solx;
            endY1=y2;
            startX1=startX;
            startY1=2*y2-startY;
            return;
        }
        else{
            endX1=x2;
            endY1=soly;
            startX1=2*x2-startX;
            startY1=startY;
            return;
        }
    }
}

double findDist(double ballX,double ballY,double startX,double startY,double endX,double endY,double& footPointX,double& footPointY) {
    //计算点线距离，将垂足坐标保存在footPointX footPointY中
	double a = startY - endY;
	double b = endX - startX;
	double c = endY*startX - startY*endX;
	double d = sqrt(a*a + b * b );
	footPointX = (b*b*ballX - a * b*ballY - a * c) / (a*a+b*b);
	footPointY = (a*a*ballY - a * b*ballX - b * c) / (a*a + b * b);
	return (abs(a*ballX+b*ballY+c)) / d;
}

void iterativeDraw(Mat src,double lx1,double ly1,double lx2,double ly2,double x1,double x2,double y1,double y2,vector<Vec3f> balls,double& lx3,double& ly3,double& lx4,double& ly4){
    //辅助函数，用于实现有台球遮挡的反射球路计算的迭代步骤
    double xx1,xx2,yy1,yy2;
    double fX,fY;
    double fXX,fYY;
    double d;
    drawReflections(lx1,ly1,lx2,ly2,x1,x2,y1,y2,xx1,yy1,xx2,yy2);//先计算到达桌面边缘的球路
    int hit=-1;
    double dis=src.cols+src.rows;
    for(int i=0;i<balls.size();i++){
        d=findDist(balls[i][0],balls[i][1],lx2,ly2,xx2,yy2,fX,fY);
        if(d<28 && (fX-xx2)*(fX-lx2)<0 && abs(fX-lx2)<dis){//找到离线段起点最近的遮挡球
            hit=i;
            dis=d;
            fXX=fX;
            fYY=fY;
        }
    }
    if(hit==-1){//确定没有遮挡，则绘制直到桌面边缘的球路
        line(src,Point(lx2,ly2),Point(xx2,yy2),Scalar(0,255,0),3);
        lx3=xx1;
        lx4=xx2;
        ly3=yy1;
        ly4=yy2;
    }
    else{//若存在遮挡
        double dx=fXX-lx2;
        double dy=fYY-ly2;
        double dr=sqrt(dx*dx+dy*dy);
        dx=dx/dr;
        dy=dy/dr;
        double u=sqrt(56*56-dis*dis);
        double px=fXX-u*dx;//计算前一个球的碰撞时末位置
        double py=fYY-u*dy;
        line(src,Point(lx2,ly2),Point(px,py),Scalar(0,255,0),3);//绘制前一个球的球路
        lx3=px;//迭代更新线段的位置与定向
        ly3=py;
        lx4=balls[hit][0];
        ly4=balls[hit][1];
    }
}

void drawPath(Mat src,double stX1,double stY1,double stX2,double stY2,double wbX,double wbY,double x1,double x2,double y1,double y2,vector<Vec3f> balls){
    //绘制有球遮挡条件下的反射球路的总体函数
    double fX,fY;
    double fXX,fYY;
    double xx1,xx2,yy1,yy2;
    double xx3,xx4,yy3,yy4;
    double d=findDist(wbX,wbY,stX1,stY1,stX2,stY2,fX,fY);
    if(d>20){//只有杆接近于指向球时，才进行绘制
        return;
    }
    else{
        line(src,Point(stX2,stY2),Point(wbX,wbY),Scalar(0,255,0),3);
        drawReflections(stX2,stY2,X,Y,x1,x2,y1,y2,xx1,yy1,xx2,yy2);
        int hit=-1;
        double dis=src.cols+src.rows;
        for(int i=0;i<balls.size();i++){
            if(abs(balls[i][0]-X)<10.0 && abs(balls[i][1]-Y)<10.0)continue;//对于第一步绘制，须排除白球的遮挡计算
            d=findDist(balls[i][0],balls[i][1],X,Y,xx2,yy2,fX,fY);
            if(d<28 && (fX-xx2)*(fX-X)<0 && abs(fX-X)<dis){
                hit=i;
                dis=d;
                fXX=fX;
                fYY=fY;
            }
        }
        if(hit==-1){
            line(src,Point(wbX,wbY),Point(xx2,yy2),Scalar(0,255,0),3);
            for(int i=0;i<5;i++){
                iterativeDraw(src,xx1,yy1,xx2,yy2,x1,x2,y1,y2,balls,xx3,yy3,xx4,yy4);//完成第一步绘制后，以后每一次都迭代地调用辅助函数
                xx1=xx3;
                xx2=xx4;
                yy1=yy3;
                yy2=yy4;
            }
        }
        else{
            double dx=fXX-X;
            double dy=fYY-Y;
            double dr=sqrt(dx*dx+dy*dy);
            dx=dx/dr;
            dy=dy/dr;
            double u=sqrt(56*56-dis*dis);
            double px=fXX-u*dx;
            double py=fYY-u*dy;
            line(src,Point(wbX,wbY),Point(px,py),Scalar(0,255,0),3);
            xx1=px;
            yy1=py;
            xx2=balls[hit][0];
            yy2=balls[hit][1];
            for(int i=0;i<2;i++){
                iterativeDraw(src,xx1,yy1,xx2,yy2,x1,x2,y1,y2,balls,xx3,yy3,xx4,yy4);
                xx1=xx3;
                yy1=yy3;
                xx2=xx4;
                yy2=yy4;
            }
        }
    }
}

void findStick(Mat src,double ballCoef=1.2,double edgeCoef=20){//球杆探测程序
    int r,g,b;
    int x1,x2,y1,y2;
    int ox1,ox2,oy1,oy2;
    Point st,ed;
    backgroundColorExtraction(src,r,g,b);
    Mat bin=binarization(src, r, g, b,50);//二值化
    Mat gray;
    findTableEdge(src,x1,x2,y1,y2);//桌面边缘探测
    if(src.channels()==3){
        cvtColor(src,gray,COLOR_RGB2GRAY);
    }
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, src.rows / 40, 20, 30, 13, 30);  //寻找球形区域
    for (size_t i = 0; i < circles.size(); i++){//将球形区域涂黑，为保证球形区域被黑色覆盖，将ballCoef取值略大于1
		for(int j=round(circles[i][0]-ballCoef*circles[i][2]);j<=round(circles[i][0]+ballCoef*circles[i][2]);j++){
            for(int k=round(circles[i][1]-ballCoef*circles[i][2]);k<=round(circles[i][1]+ballCoef*circles[i][2]);k++){
                bin.at<Vec3b>(k,j)[0]=0;
                bin.at<Vec3b>(k,j)[1]=0;
                bin.at<Vec3b>(k,j)[2]=0;
            }
        }
	}
    minmax(x1,x2);//将桌面边缘位置参数按从小到大排序
    minmax(y1,y2);
    ox1=x1;ox2=x2;oy1=y1;oy2=y2;//保存原有桌面边缘信息
    //rectangle(bin,Point(ox1,oy1),Point(ox2,oy2),Scalar(255,0,0),3);
    x1+=src.cols/edgeCoef;//从桌面边缘向内求取一个较少受到噪声影响的矩形区域
    x2-=src.cols/edgeCoef;
    y1+=src.rows/edgeCoef;
    y2-=src.rows/edgeCoef;
    //rectangle(bin,Point(x1,y1),Point(x2,y2),Scalar(0,255,0),3);
    bool cont=true;
    for(int y=y1;y<=y2;y++){//顺着矩形区域寻找比较连续的白色区域，用辅助函数WValue进行连续像素的白色判定
        if(WValue(bin,y,x1,false,10)<200){
            continue;
        }
        else{
            cont=false;
            st.x=x1;
            st.y=y;
            break;
        }
    }
    if(cont){
        for(int y=y1;y<=y2;y++){
            if(WValue(bin,y,x2,false,10)<200){
                continue;
            }
            else{
                cont=false;
                st.x=x2;
                st.y=y;
                break;
            }
        }
    }
    if(cont){
        for(int x=x1;x<=x2;x++){
            if(WValue(bin,y1,x,true,10)<200){
                continue;
            }
            else{
                cont=false;
                st.x=x;
                st.y=y1;
                break;
            }
        }
    }
    if(cont){
        for(int x=x1;x<=x2;x++){
            if(WValue(bin,y2,x,false,10)<200){
                continue;
            }
            else{
                st.x=x;
                st.y=y2;
                break;
            }
        }
    }
    //circle(bin,st,3,Scalar(0,0,255),3);
    //得到杆上位点
    vector<Point> v1;
    vector<Point> v2;
    vector<Point> result;
    vector<Point> key;
    v1.clear();
    v2.clear();
    result.clear();
    key.clear();
    for(double angle=0.0;angle<=6.28;angle+=0.01){//以杆上位点为圆心，逆时针旋转搜索
        if(bin.at<Vec3b>(st.y+50*sin(angle),st.x+50*cos(angle))[0]==255 && bin.at<Vec3b>(st.y+50*sin(angle+0.01),st.x+50*cos(angle+0.01))[0]==0){
            Point e1;
            e1.x=st.x+50*cos(angle);
            e1.y=st.y+50*sin(angle);
            v1.push_back(e1);//从二值化图像的白色区域进入二值化图像的黑色区域的位点
        }
        else if(bin.at<Vec3b>(st.y+50*sin(angle),st.x+50*cos(angle))[0]==0 && bin.at<Vec3b>(st.y+50*sin(angle+0.01),st.x+50*cos(angle+0.01))[0]==255){
            Point e2;
            e2.x=st.x+50*cos(angle);
            e2.y=st.y+50*sin(angle);
            v2.push_back(e2);//从二值化图像的黑色区域进入二值化图像的白色区域的位点
        }
    }
    for(int i=0;i<2;i++){
        //circle(bin,v1[i],3,Scalar(0,255,0),3);
        for(int j=0;j<2;j++){
            //circle(bin,v2[j],3,Scalar(255,0,0),3);
            if(abs(v1[i].x-v2[j].x)<50 && abs(v1[i].y-v2[j].y)<50){//求取上述两类位点的位置平均
                Point e;
                e.x=(v1[i].x+v2[j].x)/2;
                e.y=(v1[i].y+v2[j].y)/2;
                result.push_back(e);
            }
        }
    }
    
    int stick_x1,stick_y1,stick_x2,stick_y2;
    double dx,dy;
    if(result.size()>1){
        stick_x1=result[0].x;
        stick_y1=result[0].y;
        stick_x2=result[1].x;
        stick_y2=result[1].y;
        dx=stick_x1-stick_x2;
        dy=stick_y1-stick_y2;
    }
    bool outOfRange1=false;
    bool outOfRange2=false;
    int px1,px2,px3,px4,py1,py2,py3,py4;
    for(int i=1;i<bin.cols;i++){
        px1=int(result[0].x+i);
        py1=int(result[0].y+i*dy/dx);
        px2=int(result[0].x+i+1);
        py2=int(result[0].y+(i+1)*dy/dx);
        px3=int(result[0].x-i);
        py3=int(result[0].y-i*dy/dx);
        px4=int(result[0].x-i-1);
        py4=int(result[0].y-(i+1)*dy/dx);
        if((px2>ox2)||(px2<ox1)||(py2>oy2)||(py2<oy1)){
            outOfRange1=true;
        }
        if((px4>ox2)||(px4<ox1)||(py4>oy2)||(py4<oy1)){
            outOfRange2=true;
        }
        if(outOfRange1 && outOfRange2)break;
        if(!outOfRange1){//要么搜索到桌面边缘，要么搜索到杆的尖端
            if(int(bin.at<Vec3b>(py1,px1)[0])==255 && int(bin.at<Vec3b>(py2,px2)[0])==0){
                Point r;
                r.x=px1;
                r.y=py1;
                key.push_back(r);
                break;
            }
        }
        if(!outOfRange2){
            if(bin.at<Vec3b>(py3,px3)[0]==255 && bin.at<Vec3b>(py4,px4)[0]==0){
                Point r;
                r.x=px3;
                r.y=py3;
                key.push_back(r);
                break;
            }
        }
    }
    if(key.size()>0){
        circle(bin,key[0],3,Scalar(0,0,255),3);
        circle(bin,result[0],3,Scalar(0,0,255),3);
        circle(bin,result[1],3,Scalar(0,0,255),3);//绘制球杆上的关键点
        double xx1,xx2,yy1,yy2,xx3,yy3,xx4,yy4;//以下绘制无遮挡的反射路径
        drawReflections(result[0].x,result[0].y,key[0].x,key[0].y,ox1,ox2,oy1,oy2,xx1,yy1,xx2,yy2);
        line(src,Point(key[0].x,key[0].y),Point(xx2,yy2),Scalar(0,0,255),3);
        for(int i=0;i<5;i++){
            drawReflections(xx1,yy1,xx2,yy2,ox1,ox2,oy1,oy2,xx3,yy3,xx4,yy4);
            line(src,Point(xx2,yy2),Point(xx4,yy4),Scalar(0,0,255),3);
            xx1=xx3;
            xx2=xx4;
            yy1=yy3;
            yy2=yy4;
        }
    }
    if(key.size()>0){//绘制有遮挡的反射路径
        drawPath(src,result[0].x,result[0].y,key[0].x,key[0].y,X,Y,ox1,ox2,oy1,oy2,circles);
    }
    imshow("begin",src);//结果显示
	imshow("Circle", bin);
    imwrite("r.jpg",src);
    imwrite("8.jpg",bin);
	waitKey(0);
}


void testFunction1(Mat src){//测试Hough线检测的效果
    Mat gray;
    cvtColor(src,gray,COLOR_RGB2GRAY);
    vector<Vec4i> lines;
    HoughLinesP(gray, lines,1, CV_PI/180, 1000, 0, 0 );
    for( size_t i = 0; i < lines.size(); i++ ){
        Vec4i l = lines[i];
        line( src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1);
    }
    imshow("l",src);
    imwrite("a.jpg",src);
    waitKey(0);
}

void testFunction2(const Mat& src,int binBound=50,const int radius=20){//测试特征点提取算法是否能够运用于球杆的尖端检测
    int r,g,b;
    backgroundColorExtraction(src,r,g,b);
    Mat im1=binarization(src,r,g,b,binBound);
    Mat im2=(Mat_<float>(im1.rows,im1.cols));
    for(int i=0;i<im1.rows;i++){
        for(int j=0;j<im1.cols;j++){
            if(im1.at<Vec3b>(i,j)[0]==255){
                im2.at<float>(i,j)=1.0/(3*radius*radius);
            }
            else{
                im2.at<float>(i,j)=0;
            }
        }
    }
    Mat dst=(Mat_<float>(src.rows,src.cols));
    Point anchor;
    anchor=Point(-1,-1);
    double delta=0,ddepth=-1;
    Mat kernel=(Mat_<float>(2*radius+1,2*radius+1));
    for(int i=0;i<2*radius+1;i++){
        for(int j=0;j<2*radius+1;j++){
            if((i-radius)*(i-radius)+(j-radius)*(j-radius)<radius*radius){
                kernel.at<float>(i,j)=1;
            }
            else{
                kernel.at<float>(i,j)=0;
            }
        }
    }
    filter2D(im2,dst,ddepth,kernel,anchor,delta,BORDER_DEFAULT);
    cvtColor(dst,dst,COLOR_GRAY2RGB);
    vector<KeyPoint> keypoints;
    Ptr<FeatureDetector> fast=FastFeatureDetector::create(40);
    fast->detect(im1, keypoints);
    drawKeypoints(im1,keypoints,im1,Scalar(255,0,0),DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow("",im1);
    waitKey(0);
}



int main(int argc,char** argv){
    Mat im=imread(argv[1]);
    findStick(im,1.2,10);
    //Mat im2=imread(argv[1]);
    //testFunction2(im2);
    //int x1,x2,y1,y2;
    //imshow("det",findTableEdge(im,x1,x2,y1,y2,0,20,40));
    //rectangle(im,Point(x1,y1),Point(x2,y2),Scalar(255,0,0),3);
    //imshow("orig",im);
    //imwrite(argv[2],im);
    //int r,g,b;
    //imshow("ed",edgeDetection(im));
    //backgroundColorExtraction(im,r,g,b);
    //cout<<r<<" "<<g<<" "<<b<<endl;
    //imshow("b",binarization(im,r,g,b,50));
    //waitKey(0);
    //Mat im2=imread(argv[1]);
    //testFunction1(im2);
    return 0;
}