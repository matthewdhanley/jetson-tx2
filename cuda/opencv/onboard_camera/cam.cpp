#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/videoio.hpp>
#include <iostream>


int main(){

//const char* gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! \
nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! \
videoconvert ! video/x-raw, format=(string)BGR ! \
appsink";

cv::VideoCapture cap(0);

if (!cap.isOpened()){
	std::cout<<"failed to open camera."<<std::endl;
	return -1;
}

unsigned int w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
unsigned int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
unsigned int pixels = w * h;

cv::namedWindow("My Cam Prev", CV_WINDOW_AUTOSIZE);
cv::Mat frame_in(w, h, CV_8UC3);

while(1){
	cv::imshow("My Cam Prev", frame_in);
	cv::waitKey(1);
}

cap.release();

return 0;
}
