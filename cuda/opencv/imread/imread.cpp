#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

int im_read(){
	cv::Mat img = cv::imread("../img/fruit.jpg");
	cv::imshow("fruit",img);
	cv::waitKey();
	printf("rows: %d\n",img.rows);
	printf("cols: %d\n",img.cols);
}

int main(){
	im_read();
}
