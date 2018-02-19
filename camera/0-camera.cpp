#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    const char* gst;
    gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! \
                nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
                videoconvert ! video/x-raw, format=(string)BGR ! \
                appsink";

    // open the camera with the gst string
    cv::VideoCapture cap(gst);

    // error handling
    if(!cap.isOpened())
    {
        std::cout<<"Failed to open camera."<<std::endl;
        return -1;
    }


    unsigned int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    unsigned int pixels = width*height;
    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<std::endl;

    cv::namedWindow("MyCameraPreview", CV_WINDOW_AUTOSIZE);

    // create Mat item with width, height from camera and make it color
    cv::Mat frame_in(width, height, CV_8UC3);



    while(1)
    {
        // error handling
        if (!cap.read(frame_in)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

        // show the image
        else {
            cv::imshow("MyCameraPreview",frame_in);
            if(cv::waitKey(100) >= 0){
                printf("Exiting.");
                break;
            }
        }
    }

    cap.release();

    return 0;
}