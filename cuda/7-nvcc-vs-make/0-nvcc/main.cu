//
// Created by matt on 04/01/18.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// -----------------CONSTANT MEMORY-------------------------------------
#define BLOCK_SIZE 32

// number of channels in image
#define CHANNELS 3

// image props
#define WIDTH 1280
#define HEIGHT 720


// error checking macro
// wrap api calls with this
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// -------------------------------------- STRUCTS ----------------------------------------------------------------------
struct colorMaskHSV {
    int hue_min;
    int hue_max;
    int sat_min;
    int sat_max;
    int val_min;
    int val_max;
};

struct centroid {
    float x;
    float y;
};

// -------------------------------------- GLOBALS ----------------------------------------------------------------------
colorMaskHSV mask;


// -------------------------------------- GPU KERNELS ------------------------------------------------------------------

/***********************************************************************************************************************
 * Name:    gpu_colorMask
 * Author:  Matthew Hanley
 * Purpose: Mask an image to isolate a blob of color, quickly.
 *
 * Inputs:
 * @param Pout      - binary output mask
 * @param Pin       - HSV input image, 3 channels (Hue, Saturation, Value)
 * @param width     - width of the image in pixels
 * @param height    - height of the image in pixels
 * @param colorMask - struct containing bounding values
 *
 **********************************************************************************************************************/
__global__ void gpu_colorMask(unsigned char *Pout, unsigned char* Pin_hsv, unsigned char* Pin_rgb,
                              int width, int height, colorMaskHSV colorMask, centroid* c, unsigned int* countNonZeroBlocks) {

    __shared__ unsigned int momentsX[BLOCK_SIZE * BLOCK_SIZE];      /* used for storing moments, making array of zeros */
    __shared__ unsigned int momentsY[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ unsigned int countNonZero[BLOCK_SIZE * BLOCK_SIZE];

    int col = threadIdx.x + blockIdx.x * blockDim.x;                /* index into row of image */
    int row = threadIdx.y + blockIdx.y * blockDim.y;                /* index into column of image */
    unsigned int t = threadIdx.y * blockDim.x + threadIdx.x;        /* index into the block */
    unsigned int b = blockIdx.y * gridDim.x + blockIdx.x;           /* index blocks into grid */


    if (col < width && row < height)                                /* Make sure within image, this will cause some control*/
                                                                    /* divergence at the edges of the image*/
    {
        unsigned int i = (row * width + col) * CHANNELS;            /* unique index into image */
        unsigned char h = Pin_hsv[i];                               /* Grab values of hue, saturation, and value */
        unsigned char s = Pin_hsv[i + 1];
        unsigned char v = Pin_hsv[i + 2];

        if (h <= colorMask.hue_max && h >= colorMask.hue_min &&     /* check if pixel should be masked. */
            s <= colorMask.sat_max && s >= colorMask.sat_min &&
            v <= colorMask.val_max && v >= colorMask.val_min) {
            Pout[i] = Pin_rgb[i];                                   /* masking rgb image */
            Pout[i + 1] = Pin_rgb[i + 1];
            Pout[i + 2] = Pin_rgb[i + 2];
            momentsX[t] = col;                                      /* assigning weights to moment */
            momentsY[t] = row;
            countNonZero[t] = 1;                                    /* saving that not zero for average sum later */
        } else                                                      /* assign black values! */
        {
            Pout[i] = 0;
            Pout[i + 1] = 0;
            Pout[i + 2] = 0;
            momentsX[t] = 0;
            momentsY[t] = 0;
            countNonZero[t] = 0;                                    /* saving that zero for average sum later */
        }
    }
    __syncthreads();

    // now want to do some reduction to find center of mass of each block
    // the initial stride needs to be a power of 2
    for (unsigned int stride = BLOCK_SIZE * BLOCK_SIZE / 2; stride >= 1; stride = stride >> 1) {
        __syncthreads();

        // todo - make sure this isn't janky...

        if (row * width + col + stride < width * height) {          /* make pixels within image */
            if (t < stride) {                                       /* reducing size */
                momentsX[t] += momentsX[t + stride];
                momentsY[t] += momentsY[t + stride];
                countNonZero[t] += countNonZero[t + stride];        /* counting non zero */
            }
        }
    }
    __syncthreads();
    if (t == 0 && b < gridDim.x * gridDim.y) {
        if (countNonZero[0] > 0) {
            c[b].x = float(momentsX[0]) / float(countNonZero[0]);
            c[b].y = float(momentsY[0]) / float(countNonZero[0]);
            countNonZeroBlocks[b] = 1;
        }
        else{
            c[b].x = 0.0;
            c[b].y = 0.0;
            countNonZeroBlocks[b] = 0;
        }
    }
    __syncthreads();

    // now c is composed of centroids for each block. Time to do a similar reducing algorithm on c. It should be local
    // in memory since it was the last data worked with
    unsigned int stride = gridDim.x * gridDim.y;

    // bitwise method for finding next highest power of 2
    // see https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    stride--;
    stride |= stride >> 1;
    stride |= stride >> 2;
    stride |= stride >> 4;
    stride |= stride >> 8;
    stride |= stride >> 16;
    stride++;

    stride /= 2;

    for ( ; stride >= 1; stride = stride >> 1){
        __syncthreads();

        if (t == 0 && b < stride && (b + stride) < gridDim.x * gridDim.y){
            c[b].x += c[b + stride].x;
            c[b].y += c[b + stride].y;
            countNonZeroBlocks[b] += countNonZeroBlocks[b + stride];
        }
    }
    __syncthreads();

    // average it out.
    if (row == 0 && col == 0){
        c[0].x = c[0].x/float(countNonZeroBlocks[0]);
        c[0].y = c[0].y/float(countNonZeroBlocks[0]);
    }
}
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
                gpu_colorMask <<< numBlocks, numThreads >>> (d_Pout, d_Pin_hsv, d_Pin_rgb, width, height, mask, d_c, d_non_zero);
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