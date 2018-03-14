//
// Created by matt on 3/4/18.
//

#include "gpu_kernels.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <stdlib.h> //needed for rand()
#include <stdio.h> //needed for printf()
#include <time.h>


void cpu_greyscale(cv::Mat src, cv::Mat dst){
    int num_rows = src.rows;
    int num_cols = src.cols;

    // loop through all the pixels
    for(int i=0; i<num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            // You can now access the pixel value with cv::Vec3b
            unsigned char r = src.at<cv::Vec3b>(i, j)[0];
            unsigned char g = src.at<cv::Vec3b>(i, j)[1];
            unsigned char b = src.at<cv::Vec3b>(i, j)[2];
            dst.at<uchar>(i, j) = 0.21f * r + 0.71f * g + 0.07f * b;
        }
    }
}


void cpu_blur(cv::Mat src, cv::Mat dst){
    int num_rows = src.rows;
    int num_cols = src.cols;

    // size of kernel (i.e. 3 = -3 -> 3 so size of 7 px)
    unsigned char ksize = 3;
    for(int i=ksize; i<num_rows-ksize; i++) {
        for (int j = ksize; j < num_cols-ksize; j++) {
            unsigned int pixsum = 0;
            unsigned int numpx = 0;
            // You can now access the pixel value with cv::Vec3b
            for (int k=-ksize; k < ksize+1; k++){
                for (int l=-ksize; l < ksize+1; l++) {
                    numpx++;
                    pixsum += (unsigned int) src.at<uchar>(i + k, j + l);
                }
            }
            dst.at<uchar>(i,j) = (unsigned char) (pixsum / numpx);
        }
    }
}


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

    // create a window
    cv::namedWindow("Output", CV_WINDOW_AUTOSIZE);


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

    // --------------------------------CALCULATING BLOCKS AND THREADS-----------------------

    // count available devices
    int dev_count;
    cudaGetDeviceCount(&dev_count);

    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        printf("maxThreadsPerBlock: %d\n",dev_prop.maxThreadsPerBlock);
        printf("multiProcessorCount: %d\n",dev_prop.multiProcessorCount);
        printf("wrapSize: %d\n",dev_prop.warpSize);
        printf("regsPerBlock: %d\n", dev_prop.regsPerBlock);
        printf("maxThreadsPerMultiProcessor: %d\n",dev_prop.maxThreadsPerMultiProcessor);

    }

    int numThreadsX = 32;
    dim3 numThreads(numThreadsX,numThreadsX);
    int numBlocksX = (width  + numThreadsX - 1)/numThreadsX;
    int numBlocksY = (height + numThreadsX - 1)/numThreadsX;
    printf("\nnumBlocksX: %d\n",numBlocksX);
    printf("numBlocksY: %d\n",numBlocksY);
    dim3 numBlocks(numBlocksX, numBlocksY);

    printf("numBlocks: %d\n",numBlocks.x + numBlocks.y);
    printf("total threads: %d\n",(numBlocks.x * numBlocks.y)*numThreads.x*numThreads.y);

    char keystroke = 'g';  // default keystroke (GPU)
    char key_prev = 'g'; // default previous keystroke(GPU)
    
    while(1)
    {
        // error handling
        if (!cap.read(frame_in)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

        else {
            // get the keystroke
            char tmp_key = (char) cv::waitKey(1);
            // check if a key has been pressed
            if (tmp_key != 255){
                key_prev = keystroke;
                keystroke = tmp_key;
                tmp_key = 255;
            }

            switch(keystroke) {
                case 'q' : {
                    // ------------------QUIT CASE------------------------------------------------------------
                    printf("Exiting.\n");
                    exit(0);
                }
                case 'c' : {
                    // ------------------CPU CASE------------------------------------------------------------
                    // make some containers
                    cv::Mat frame_out_grey(cv::Size(width, height), CV_8UC1);
                    cv::Mat frame_out_blur(cv::Size(width, height), CV_8UC1);

                    // apply the operaions
                    cpu_greyscale(frame_in, frame_out_grey);
                    cpu_blur(frame_out_grey, frame_out_blur);
                    cv::putText(frame_out_blur, "CPU", cv::Point(30,30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250), 1, CV_AA);
                    cv::imshow("Output", frame_out_blur);
                    break;
                }
                case 'g' : {
                    // ------------------GPU CASE------------------------------------------------------------
                    // copy memory to the device
                    cudaMemcpy(d_Pin, frame_in.data, pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

                    // execute kernel
                    cuda_grey_and_blur(d_Pout, d_Pin, width, height, numBlocks, numThreads);

                    // copy memory back to host
                    // todo - figure out how to display w/o copying back
                    cudaMemcpy(h_Pout, d_Pout, sizeof(unsigned char) * pixels, cudaMemcpyDeviceToHost);

                    // create container for new image and insert the data
                    cv::Mat frame_out(cv::Size(width, height), CV_8UC1, h_Pout);

                    cv::putText(frame_out, "GPU", cv::Point(30,30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250), 1, CV_AA);

                    // show the image
                    cv::imshow("Output", frame_out);
                    break;
                }
            }
        }
    }

    // free the camera!
    cap.release();

    return 0;
}