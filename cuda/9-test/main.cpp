//
// Created by matt on 04/01/18.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "macros.h"
#include "structs.h"



// --------------------------------------cpu functions------------------------------------------------------------------
// todo - add CPU mask

// -------------------------------------- HELPER FUNCTIONS -------------------------------------------------------------


// -----------MAIN-------------------MAIN-----------------MAIN----------------MAIN-------------MAIN---------------------

int main(){

    // ----------------getting the camera stream-----------------------------------
    const char* gst;
    gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! \
                nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
                videoconvert ! video/x-raw, format=(string)BGR ! \
                appsink";

    // open the camera with the gst string
    // If you want to use a webcam, change this to cap(1) (at least on the jetson tx2)
    cv::VideoCapture cap(gst);

    // error handling
    if(!cap.isOpened()) {
        printf("Failed To Open Camera");
        return -1;
    }

    // get info about the picture
    unsigned int width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    unsigned int pixels = width * height;

    // print info about the picture
    printf("Frame Size: %d x %d\n %d pixels\n", width, height, pixels);

    // create opnecv Mat container for image
    cv::Mat frame_in;
    cv::Mat frame_in_hsv;

    // grab the first frame, this will also automatically size the container
    cap >> frame_in;

    // create windows for different filters
    cv::namedWindow("Source", CV_WINDOW_AUTOSIZE);  // this window will hold the source image
    cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);  // this window will hold the resultant image

    // create bounding values
    mask.hue_max = 255;
    mask.hue_min = 0;
    mask.sat_max = 255;
    mask.sat_min = 0;
    mask.val_max = 255;
    mask.val_min = 0;



    // ------------------- ALLOCATE GPU MEMORY ------------------------------------
    unsigned char *d_Pin_hsv, *d_Pin_rgb, *d_Pout;  // init pointers
    gpuErrchk(
            cudaMalloc((void **) &d_Pin_hsv, sizeof(unsigned char) * pixels * CHANNELS) // allocate space on device for input
    );

    gpuErrchk(
            cudaMalloc((void **) &d_Pin_rgb, sizeof(unsigned char) * pixels * CHANNELS) // allocate space on device for input
    );

    gpuErrchk(
            cudaMalloc((void **) &d_Pout, sizeof(unsigned char) * pixels * CHANNELS) // allocate space on device for output
    );




    // ------------------- ALLOCATE CPU MEMORY -------------------------------------
    // note that input image has space allocated already by frame_in
    unsigned char *h_Pout = (unsigned char *) malloc(pixels * sizeof(char) * CHANNELS); // allocate space for output


    // ----------------------------GPU POLLING--------------------------------------------------------------------------
    // todo - use this info to optomize the program for any GPU
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

    // -----------------------------BLOCKS AND THREADS------------------------------------------------------------------
    int numThreadsX = BLOCK_SIZE;
    dim3 numThreads(numThreadsX, numThreadsX);
    int numBlocksX = (width)/numThreadsX;
    int numBlocksY = (height)/numThreadsX;
    dim3 numBlocks(numBlocksX, numBlocksY);

    // create container for block centroids
    centroid c;
    centroid *d_c;
    unsigned int *d_non_zero;
    gpuErrchk(
            cudaMalloc((void **) &d_c, sizeof(centroid) * numBlocksX * numBlocksY) // allocate space on device for centroids
    );
    gpuErrchk(
            cudaMalloc((void **) &d_non_zero, sizeof(int) * numBlocksX * numBlocksY) // allocate space on device for centroids
    );


    printf("\nnumBlocksX: %d\n",numBlocksX);
    printf("numBlocksY: %d\n",numBlocksY);
    printf("numBlocks: %d\n",numBlocks.x * numBlocks.y);
    printf("threads per block: %d\n", numThreads.x * numThreads.y);
    printf("total threads: %d\n",(numBlocks.x * numBlocks.y)*numThreads.x*numThreads.y);


    printf("\n\n\n\n===========================================================\n"
                   " TOGGLE WITH \"0\" on your keyboard!\n"
                   "\"q\" to quit\n"
                   "\"c\" for CPU (Not yet...)\n"
                   "===========================================================\n\n\n\n");

    // init keystroke variables
    char keystroke = '0';  // default keystroke (GPU)
    bool trackbar = 0;

    while(1){

        //---------------------Keyboard Input Handling------------------------------------------------------------------
        // get the keystroke
        char tmp_key = (char) cv::waitKey(1);

        // check if a key has been pressed
        if (tmp_key != 255){
            keystroke = tmp_key;
            tmp_key = 255;
        }

        if (keystroke == 't'){
            trackbar = !trackbar;
            cv::destroyWindow("Result");
            cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);  // this window will hold the resultant image
            keystroke = '0';

        } // toggle trackbar on or off



        //---------------------Read in the image------------------------------------------------------------------------
        // direct image from camera to cv::Mat object
        cap >> frame_in;

        // convert color to hsv
        cv::cvtColor(frame_in, frame_in_hsv, CV_BGR2HSV);

        switch(keystroke) {
            case 'q' : {
                // ------------------QUIT CASE--------------------------------------------------------------------------
                printf("Exiting.\n");

                // free everything
                cap.release();
                cudaFree(d_Pin_hsv);
                cudaFree(d_Pin_rgb);
                cudaFree(d_Pout);
                free(h_Pout);

                // exit the program
                exit(0);
            }

            case 'c' : {
                // ------------------CPU CASE---------------------------------------------------------------------------
                // apply the operations
                printf("CPU...");
                break;
            }

            case '0' : {
                // ------------------ GPU Color Mask -------------------------------------------------------------------

                if (trackbar){
                    cv::createTrackbar("Hue Min", "Result", &(mask.hue_min), 255);
                    cv::createTrackbar("Hue Max", "Result", &(mask.hue_max), 255);
                    cv::createTrackbar("Sat Min", "Result", &(mask.sat_min), 255);
                    cv::createTrackbar("Sat Max", "Result", &(mask.sat_max), 255);
                    cv::createTrackbar("Val Min", "Result", &(mask.val_min), 255);
                    cv::createTrackbar("Val Max", "Result", &(mask.val_max), 255);

                }

                // copy memory from host to device
                gpuErrchk(
                        cudaMemcpy(d_Pin_hsv, frame_in_hsv.data, pixels * sizeof(unsigned char) * CHANNELS,
                                   cudaMemcpyHostToDevice)
                );

                gpuErrchk(
                        cudaMemcpy(d_Pin_rgb, frame_in.data, pixels * sizeof(unsigned char) * CHANNELS,
                                   cudaMemcpyHostToDevice)
                );

                // execute gpu kernel
                cuda_mask(d_Pout, d_Pin_hsv, d_Pin_rgb, width, height, mask, d_c, d_non_zero, numBlocks, numThreads);

                gpuErrchk(
                        cudaGetLastError() // check for errors
                );

                // make sure all the threads finished
                gpuErrchk(
                        cudaThreadSynchronize()
                );

                // copy memory back to host from device
                gpuErrchk(
                        cudaMemcpy(h_Pout, d_Pout, sizeof(unsigned char) * pixels * CHANNELS, cudaMemcpyDeviceToHost)
                );

                gpuErrchk(
                        cudaMemcpy(&c, d_c, sizeof(centroid), cudaMemcpyDeviceToHost)
                );

                cv::circle(frame_in, cv::Point(c.x, c.y), 5, cv::Scalar(0, 255, 0), -1);

                // create container for new image and insert the data
                cv::Mat frame_out(cv::Size(width, height), CV_8UC3, h_Pout);

                // place some text on the image
                cv::putText(frame_out, "GPU Mask", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1, CV_AA);

                // show the images
                cv::imshow("Source", frame_in);
                cv::imshow("Result", frame_out);
                break;
            }

            default: {
                cv::Mat frame_out(cv::Size(width, height), CV_8UC3, h_Pout);
                cv::putText(frame_out, "INCORRECT KEYSTROKE", cv::Point(height/2, width/2),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 3, cv::Scalar(200, 200, 250), 1, CV_AA);
                // show the images
                cv::imshow("Source", frame_in);
                cv::imshow("Result", frame_out);
                break;
            }
        }
    }
}