#include <iostream>
#include <opencv2/opencv.hpp>
#include "CameraConfig.h.in"
#include "include/cuda_kernels.h"
//#include "custom.h"
#include <cuda.h>

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

    cv::Mat frame_out(width,height,CV_8UC1);

    // gpu memory allocation
    unsigned int *d_Pin, *d_Pout;
    cudaMalloc((void **) &d_Pin, sizeof(unsigned int)*pixels*3);
    cudaMalloc((void **) &d_Pout, sizeof(unsigned int)*pixels);

    int numThreads = 32;
    int numBlocks = (pixels + numThreads -1)/numThreads;


    while(1)
    {
        // error handling
        if (!cap.read(frame_in)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

            // show the image
        else {
            //cv::imshow("MyCameraPreview",frame_in);
            cudaMemcpy(d_Pin, frame_in.data, pixels*3*sizeof(unsigned int), cudaMemcpyHostToDevice);
            ch1(d_Pout, d_Pin, width, height, numBlocks, numThreads);
            if(cv::waitKey(1) >= 0){
                printf("Exiting.\n");
                break;
            }
        }
    }

    cap.release();

    return 0;
}
