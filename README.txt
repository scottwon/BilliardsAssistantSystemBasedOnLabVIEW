分工：
算法设计、研究、调测，原型程序设计：王思恢
labview接口，硬件调测与实现：黄韩
labview代码实现：陈未殊

主程序在final.vi中，运行即可（需要连接投影仪，摄像头）

各类vi文件的作用（按字母顺序）：
detect_goal			实现球洞检测
dot_line_dis			点到直线距离
draw_circle_image			在image格式（IMAQ中的组件）下，绘画圆
draw_track6_close&dis&angle_resist	实现在投影仪下，球路轨迹的绘画（采用picture格式）
draw_track7_image			实现在前面板中，球路轨迹的绘画（采用image格式）
draw_track7_image_diplay		与draw_track7_image功能一样，只不过这个用来做单独模块的展示，上一个是用作子vi
edge_cordinate			将检测到的边缘信息转换成坐标信息
FINAL				主程序
foot_on_line			检测点到线段所在直线的垂线的垂足是否落在线段上
hit_pos				计算白球撞击点
inner_product			计算内积
integer7_overlayDraw		用作image格式下球路的展示，但此功能也已经直接封装到了FINAL中，所以此vi可忽略
p2p_distance			两点间距离
white_ball_detect4_for_function	白球检测
去除背景fin_forfunctionuse		图像二值化
调用方.vi				调用图片显示vi，用于投影仪显示，但此vi功能也已经封装到FINAL中，所以此vi可忽略
图片显示.vi			显示传来图片

