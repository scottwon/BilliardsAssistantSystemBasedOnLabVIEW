#include<iostream>
#include<cmath>
#include<vector>
#include<GL/glut.h>
#include<time.h>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/features2d.hpp>
using namespace std;
using namespace cv;
const double r=0.05;
double min(double a, double b) {
	return a > b ? b : a;
}
double max(double a, double b) {
	return a > b ? a : b;
}
void MinMax(double a,double b,double& min,double& max){
    if(a>b){
        min=b;
        max=a;
    }
    else{
        min=a;
        max=b;
    }
}
struct Point2D{
    double x,y;
};
struct color{
    double r,g,b;
};
vector<Point2D> billiards;
vector<color> colorTable;//C版本的随机数发生器在循环语句中有时工作不正常，此变量用于事先生成一个随机颜色表
Point2D whtbl;
vector<Point2D> goals;
struct Line2D{
    Point2D startPoint,endPoint;
};
double findAngle(const Line2D& path1, const Line2D& path2) {//辅助函数，计算入射球路与出射球路的向量夹角的余弦值
	double len1 = sqrt((path1.endPoint.y-path1.startPoint.y)*(path1.endPoint.y - path1.startPoint.y)+(path1.endPoint.x-path1.startPoint.x)*(path1.endPoint.x - path1.startPoint.x));
	double len2 = sqrt((path2.endPoint.y - path2.startPoint.y)*(path2.endPoint.y - path2.startPoint.y) + (path2.endPoint.x - path2.startPoint.x)*(path2.endPoint.x - path2.startPoint.x));
	double r= (path1.endPoint.y - path1.startPoint.y)*(path2.endPoint.y - path2.startPoint.y) + (path1.endPoint.x - path1.startPoint.x)*(path2.endPoint.x - path2.startPoint.x);
	r /= len1;
	r /= len2;
	return r;
}
double length(const Line2D& path){//辅助函数，计算球路路径的长度
    return sqrt((path.endPoint.y-path.startPoint.y)*(path.endPoint.y-path.startPoint.y)+(path.endPoint.x-path.startPoint.x)*(path.endPoint.x-path.startPoint.x));
}

bool findPath(Point2D whiteBall,Point2D target,Point2D goal,double radius,Line2D& path,Line2D& path2,double angleLimit=0) {
    //根据一个白球，一个目标球target，一个球洞goal的位置，找到候选球路保存在path,path2中，
    //若该球路不满足物理约束和几何约束条件，则返回值为false
	double startX, startY, endX, endY;
	double dy = goal.y - target.y;
	double dx = goal.x - target.x;
	double signY = (dy >= 0 ? 1 : -1);
	double signX = (dx >= 0 ? 1 : -1);
	if (dx != 0) {
		double angle = atan(abs(dy / dx));
		endY = target.y -signY*2* radius * sin(angle);//由dx dy的符号确定反向延长线的方向
		endX = target.x -signX*2* radius * cos(angle);
	}
	else {
		endY = target.y - signY*2*radius;
		endX = target.x;
	}
	startX = whiteBall.x;
	startY = whiteBall.y;
	path.startPoint.x = startX;//先保存计算结果
	path.startPoint.y = startY;
	path.endPoint.x = endX;
	path.endPoint.y = endY;
	path2.endPoint.y = goal.y;
	path2.endPoint.x = goal.x;
	path2.startPoint.y = target.y;
	path2.startPoint.x = target.x;
	if (findAngle(path, path2) <= angleLimit) return false;//验证入射与出射路径的向量夹角小于角度阈值
    //angleLimit为夹角的余弦值，余弦值小于阈值说明夹角大于阈值，因此路径被验证无效，返回false
	return true;
}

double findDist(Point2D ball,Line2D path,Point2D& footPoint) {//计算点线距离，把垂足位置保存在footPoint中
	double a = path.startPoint.y - path.endPoint.y;
	double b = path.endPoint.x - path.startPoint.x;
	double c = path.endPoint.y*path.startPoint.x - path.startPoint.y*path.endPoint.x;
	double d = sqrt(a*a + b * b );
	footPoint.x = (b*b*ball.x - a * b*ball.y - a * c) / (a*a+b*b);
	footPoint.y = (a*a*ball.y - a * b*ball.x - b * c) / (a*a + b * b);
	return (abs(a*ball.x+b*ball.y+c)) / d;
}

bool Validate(Point2D ball,Line2D path,double radius=r){//验证其他球对于特定球路是否产生遮挡关系
    Point2D tmp;
    double d=findDist(ball,path,tmp);
    if(d>2*r)return true;//球心与球路的点线距离大于二倍球半径，则无遮挡，球路通过验证
    double dx=path.endPoint.x-path.startPoint.x;
    double min,max;
    if(dx==0){
        MinMax(path.endPoint.y,path.startPoint.y,min,max);
        if(tmp.y>min && tmp.y<max)return false;//点线距离小于二倍球半径，且垂足在球路线段上，则存在遮挡，球路被否决
    }
    else{
        MinMax(path.endPoint.x,path.startPoint.x,min,max);
        if(tmp.x>min && tmp.x<max)return false;
    }
    return true;//点线距离小于二倍球半径，但垂足不在球路线段上，而是在球路延长线上，则实际不存在遮挡，球路通过验证
}

void findPath(Point2D whiteBall,vector<Point2D> targetBalls, vector<Point2D> goals, double radius, vector<Line2D>& solutions,vector<Line2D>& path,double angleLimit,double MaxLen) {
    //将对于多个目标球、多个球洞进行上述计算、判定、验证的过程写入一个遍历循环
	solutions.clear();
    path.clear();
	for (int i = 0;i < targetBalls.size();i++) {
		for (int j = 0;j < goals.size();j++) {
			Line2D tmp;
            Line2D tmp2;
			if (findPath(whiteBall, targetBalls[i], goals[j], radius, tmp,tmp2, angleLimit)) {
				Line2D path2;
				path2.endPoint.y = goals[j].y;
				path2.endPoint.x = goals[j].x;
				path2.startPoint.y = targetBalls[i].y;
				path2.startPoint.x = targetBalls[i].x;
				Point2D foot;
				bool isValid = true;
				for (int k = 0;k < targetBalls.size();k++) {
					if (k == i)continue;
					double d=findDist(targetBalls[k], path2, foot);
					if (d<2 * radius && foot.x>min(path2.startPoint.x, path2.endPoint.x) && foot.x < max(path2.startPoint.x, path2.endPoint.x)) {
						isValid = false;
						break;
					}
					d = findDist(targetBalls[k], tmp, foot);
					if (d<2 * radius && foot.x>min(tmp.startPoint.x, tmp.endPoint.x) && foot.x < max(tmp.startPoint.x, tmp.endPoint.x)) {
						isValid = false;
						break;
					}
				}
                if(length(tmp)+length(tmp2)>MaxLen)isValid=false;
				if(isValid){
                    solutions.push_back(tmp);
                    path.push_back(tmp2);
                }
			}
		}
	}
}

bool activePathFinding(const Line2D& stick,const Point2D& whiteBall,const vector<Point2D>& targetBalls,vector<Line2D>& path1,vector<Line2D>& path2,double boundLeft=-2,double boundRight=2,double boundUp=2,double boundDown=-2){
    //尝试写反射球路计算，但此处为未完成版本，完成版在另一个文件夹的Billiards.cpp文件中
    path1.clear();
    path2.clear();
    double x1=whiteBall.x-stick.endPoint.x;
    double y1=whiteBall.y-stick.endPoint.y;
    double x2=stick.endPoint.x-stick.startPoint.x;
    double y2=stick.endPoint.y-stick.startPoint.y;
    if(x1*y2!=x2*y1){
        return false;
    }
    else{
        Line2D tmp;
        tmp.startPoint.x=stick.endPoint.x;
        tmp.startPoint.y=stick.endPoint.y;
        tmp.endPoint.x=whiteBall.x;
        tmp.endPoint.y=whiteBall.y;
        bool isValid=true;
        for(int i=0;i<targetBalls.size();i++){
            if(!Validate(targetBalls[i],tmp)){
                isValid=false;
                break;
            }
        }
        if(isValid)path1.push_back(tmp);
    }
}

#define pi 3.1415926
int n=100;
GLfloat rtri = 0;
void init(void)//以下是将球、球洞、球路用OpenGL绘制出来的代码
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH,GL_NICEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
}
void display(void){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glRotatef(rtri,0.0f,1.0f,0.0f);
    srand((int)time(0));
    for(int i=0;i<billiards.size();i++){
        glBegin(GL_POLYGON);
        glColor3f(colorTable[i].r, colorTable[i].g, colorTable[i].b);
        for (int j = 0; j < 100; j++){
            glVertex2f(billiards[i].x+r*cos(2*pi/n*j), billiards[i].y+r*sin(2*pi/n*j));	
        }
        glEnd();
        glFlush();
    }
    glBegin(GL_POLYGON);
    glColor3f(1.0f, 1.0f, 1.0f);
    for (int j = 0; j < 100; j++){
        glVertex2f(whtbl.x+r*cos(2*pi/n*j), whtbl.y+r*sin(2*pi/n*j));	
    }
    glEnd();
    glFlush();
    Point2D g1,g2,g3,g4,g5,g6;//规定球洞位置坐标
    g1.x=-2;g1.y=-2;
    g2.x=-2;g2.y=0;
    g3.x=-2;g3.y=2;
    g4.x=2;g4.y=-2;
    g5.x=2;g5.y=0;
    g6.x=2;g6.y=2;
    goals.clear();
    goals.push_back(g1);
    goals.push_back(g2);
    goals.push_back(g3);
    goals.push_back(g4);
    goals.push_back(g5);
    goals.push_back(g6);
    for(int i=0;i<goals.size();i++){//依次绘制球洞
        glBegin(GL_POLYGON);
        glColor3f(1.0f, 1.0f, 1.0f);
        for (int j = 0; j < 100; j++){//用多边形近似绘制圆
            glVertex2f(goals[i].x+r*cos(2*pi/n*j), goals[i].y+r*sin(2*pi/n*j));	
        }
        glEnd();
        glFlush();
    }
    vector<Line2D>sol;
    vector<Line2D>path2;
    findPath(whtbl,billiards,goals,0.05,sol,path2,0.8,2.9);//根据白球、目标球、球洞的位置计算可行球路，保存在sol与path中
    for(int i=0;i<sol.size();i++){//分别绘制入射与出射球路sol,path2
        glBegin(GL_LINES);
        glColor3f(1, 1, 0);
        glVertex3f(sol[i].startPoint.x, sol[i].startPoint.y, 0);
        glVertex3f(sol[i].endPoint.x, sol[i].endPoint.y, 0);
        glEnd();
        glBegin(GL_LINES);
        glColor3f(1, 1, 0);
        glVertex3f(path2[i].startPoint.x, path2[i].startPoint.y, 0);
        glVertex3f(path2[i].endPoint.x, path2[i].endPoint.y, 0);
        glEnd();
    }
    glutSwapBuffers();
}
void reshape (int w, int h)
{
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) w/(GLfloat) h, 1.0, 100.0);
    gluLookAt(0, 0, 5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'x':
        case 27: 
        exit(0);
        break;
        default:
        break;
    }
}

int main(int argc, char **argv){
    Mat im=imread(argv[1]);
    Mat gray;
    cvtColor(im, gray, COLOR_BGR2GRAY);
    //Mat corners;
    vector<KeyPoint> keypoints;
    
    Mat grad_x, grad_y; 
    Mat abs_grad_x, abs_grad_y;
    Mat grad;
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;
    Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    imshow( "Sobel", grad );
    Ptr<FeatureDetector> fast=FastFeatureDetector::create(50);
    fast->detect(grad, keypoints);
    drawKeypoints(grad,keypoints,grad,Scalar(255,0,0),DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow("fastFeatures",grad);
    waitKey(0);
    //以上是有关特征点探测的部分调试代码
    //以下是配合球路计算算法仿真的代码
    /*srand(time(NULL));
    whtbl.x=0;//将白球置于球桌中心处
    whtbl.y=0;
    billiards.resize(13);//13个目标球
    colorTable.resize(billiards.size());
    srand((int)time(0));
    for(int i=0;i<billiards.size();i++){//目标球位置随机化
        billiards[i].x=((double)rand())/RAND_MAX*4-2;
        billiards[i].y=((double)rand())/RAND_MAX*4-2;
    }
    srand((int)time(0));//生成随机颜色表，用于绘制不同的球
    for(int i=0;i<colorTable.size();i++){
        colorTable[i].r=((double)rand())/RAND_MAX;
        colorTable[i].g=((double)rand())/RAND_MAX;
        colorTable[i].b=((double)rand())/RAND_MAX;
    }
    glutInit(&argc, argv);//调用OpenGL进行绘制
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Billiards");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(display);
    glutMainLoop();*/
    return 0;
}