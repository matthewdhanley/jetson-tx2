#include <iostream>
#include <opencv2/highgui/highgui.hpp>

int main(){
    // CV_8UC3 specifies that we want the image to be represented as 8-bit unsigned integers with 3 channels
    cv::Mat img(512, 512, CV_8UC3, cv::Scalar(0));

    cv::putText(img,"Hello World", cv::Point(10, img.rows/2),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118,185,0),2);

    cv::imshow("hello world", img);
    cv::waitKey();

    return 0;
}