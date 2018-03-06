//
// Created by matt on 3/4/18.
//

#include "gpu_add.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <stdlib.h> //needed for rand()
#include <stdio.h> //needed for printf()
#include <time.h>


int main()
{
    const char* gst;
    gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! \
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

//    cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("After Greyscale", CV_WINDOW_AUTOSIZE);
//    cv::namedWindow("After Blur", CV_WINDOW_AUTOSIZE);


    // create Mat item with width, height from camera and make it color
    cv::Mat frame_in(cv::Size(width, height), CV_8UC3);

    // gpu memory allocation
    unsigned char *d_Pin, *d_Pout, *d_grey;

    // host allocation
    unsigned char *h_Pout = (unsigned char *) malloc(pixels*sizeof(char));
    unsigned char *h_Pout2 = (unsigned char *) malloc(pixels*sizeof(char));




    cudaMalloc((void **) &d_Pin, sizeof(unsigned char)*pixels*3);
    cudaMalloc((void **) &d_grey, sizeof(unsigned char)*pixels);
    cudaMalloc((void **) &d_Pout, sizeof(unsigned char)*pixels);

    int numThreadsX = 4;
    dim3 numThreads(numThreadsX,numThreadsX);
    int numBlocksX = (width + numThreadsX -1)/numThreadsX;
    int numBlocksY = (height + numThreadsX - 1)/numThreadsX;
    dim3 numBlocks(numBlocksX, numBlocksY);

    printf("numBlocks: %d\n",numBlocks.x + numBlocks.y);
    printf("total threads: %d\n",(numBlocks.x * numBlocks.y)*numThreads.x*numThreads.y);


    while(1)
    {
        // error handling
        if (!cap.read(frame_in)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

            // show the image
        else {
//            cv::imshow("Original",frame_in);

            cudaMemcpy(d_Pin, frame_in.data, pixels*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
            cuda_grey_and_thresh(d_Pout, d_Pin, width, height, numBlocks, numThreads);
            cudaMemcpy(h_Pout, d_Pout, sizeof(unsigned char)*pixels, cudaMemcpyDeviceToHost);

            cv::Mat frame_out(cv::Size(width,height),CV_8UC1,h_Pout);

            cv::imshow("After Greyscale",frame_out);

//            cudaMemcpy(d_grey, frame_out.data, sizeof(unsigned char), cudaMemcpyHostToDevice);
//            cuda_blur(d_Pout, d_grey, width, height, numBlocks, numThreads);
//            cudaMemcpy(h_Pout2, d_grey, sizeof(unsigned char)*pixels, cudaMemcpyDeviceToHost);
//
//            cv::Mat frame_out2(cv::Size(width,height),CV_8UC1,h_Pout);
//            cv::imshow("After Blur",frame_out2);


            if(cv::waitKey(1) >= 0){
                printf("Exiting.\n");
                break;
            }
        }
    }

    cap.release();

    return 0;
}