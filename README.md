# Billiards Assistant System Based On LabVIEW
This is the course project for 'Virtual Experimental System Based on Labview'.

This is a teamwork and my teammates are Weishu Chen and Han Huang.

My contributions in the project are:

1. To find out the feasible algorithms for ball detection. In fact, OpenCV offers the function HoughCircle which is adequate for our task of detecting the balls.

2. To design the algorithms for stick detection and table edge detection. I have tried the Sobel operator, the Canny edge detector, the LSD algorithm and the EDLine algorithm and none of them is practically competent for our tasks. So I designed my own algorithms that can produce favorable outcome.

3. To design the algorithm to calculate the path of the balls.

4. To build the 'prototype'. I tested my ideas and algorithms in C++ first, and then my teammates transplant the code to the LabVIEW platform.

My work is in the folder 'ProtoCodes' and W. Chen's work is in the folder 'LabVIEW'. H. Huang helped us a lot in the aspect of hardware, brainstorming, testing/debuging and deployment.
