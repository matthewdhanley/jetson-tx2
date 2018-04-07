//
// Created by matt on 3/13/18.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// ----------------------------gpu functions----------------------------------------------------------------------------

__global__ void ch1(unsigned char* Pout, unsigned char* Pin, int width, int height) {

    int channels = 3;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    // check if pixel within range
    if (col < width && row < height){
        int gOffset = row * width + col;
        int rgbOffset = gOffset * channels;
        unsigned char r = Pin[rgbOffset  ];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];
        Pout[gOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

__global__ void gpu_blur(unsigned char* Pout, unsigned char* Pin, int width, int height){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int k_size = 3;

    if (col < width && row < height){
        int pixVal = 0;
        int pixels = 0;

        for(int blurRow = -k_size; blurRow < k_size+1; blurRow++){
            for(int blurCol = -k_size; blurCol < k_size+1; blurCol++){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width){
                    pixVal += Pin[curRow * width + curCol];
                    pixels++;
                }
            }
        }

        Pout[row * width + col] = (unsigned char) (pixVal / pixels);
    }
}

__global__ void gpu_grey_and_blur(unsigned char* Pout, unsigned char* Pin, int width, int height){

    int channels = 3;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    // check if pixel within range
    if (col < width && row < height){
        int gOffset = row * width + col;
        int rgbOffset = gOffset * channels;
        unsigned char r = Pin[rgbOffset  ];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];
        Pout[gOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
    __syncthreads();

    unsigned char k_size = 1;
    int pixVal = 0;
    int pixels = 0;
    if (col < width && row < height){
        for(int blurRow = -k_size; blurRow < k_size+1; ++blurRow){
            for(int blurCol = -k_size; blurCol < k_size+1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width){
                    pixVal += Pout[curRow * width + curCol];
                    pixels++;
                }
            }
        }
    }
    __syncthreads();
    if (col < width && row < height) {
        Pout[row * width + col] = (unsigned char) (pixVal / pixels);
    }
}


__global__ void gpu_grey_and_thresh(unsigned char* Pout, unsigned char* Pin, int width, int height){

    int channels = 3;
    unsigned char thresh = 157;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    // check if pixel within range
    if (col < width && row < height){
        int gOffset = row * width + col;
        int rgbOffset = gOffset * channels;
        unsigned char r = Pin[rgbOffset  ];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];
        unsigned char gval = 0.21f*r + 0.71f*g + 0.07f*b;

        if(gval > thresh){
            Pout[gOffset] = 255;
        }
        else {
            Pout[gOffset] = 0;
        }
    }
}

// --------------------------------------cpu functions------------------------------------------------------------------
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


void cpu_thresh(cv::Mat src, cv::Mat dst){
    unsigned char thresh = 157;

    int num_rows = src.rows;
    int num_cols = src.cols;

    for(int i = 0; i < num_rows; i++){
        for(int j = 0; j < num_cols; j++){
            if (src.at<uchar>(i,j) < thresh){
                dst.at<uchar>(i,j) = 0;
            }
            else{
                dst.at<uchar>(i,j) = 255;
            }
        }
    }
}

// --------------------------------------image buffer-------------------------------------------------------------------
// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr) {
    unsigned char *ptr = NULL;

    // this line enables allocating pinned host memory that is accessible to the device
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // this allocates space on the host that can be accessed from the kernel without needing memcpy
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);

    // passes the device a pointer corresponding to the mapped, pinned host buffer created above
    cudaHostGetDevicePointer(devicePtr, ptr, 0);
    return ptr;
}

// -----------MAIN-------------------MAIN-----------------MAIN----------------MAIN-------------MAIN---------------------

int main(){
    const char* gst;
    gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! \
                nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
                videoconvert ! video/x-raw, format=(string)BGR ! \
                appsink";

    // open the camera with the gst string
    cv::VideoCapture cap(gst);

    // error handling
    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        return -1;
    }

    // create container for image
    cv::Mat frame_in;

    // get info about the picture
    unsigned int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    unsigned int pixels = width*height;

    // print info about the picture
    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<std::endl;

    // create windows for different filters
    cv::namedWindow("Source");
    cv::namedWindow("ch1");
    cv::namedWindow("grey_and_blur");
    cv::namedWindow("grey_and_thresh");
    
    // grab the first frame
    cap >> frame_in;

    // we will be pointing the image buffers to these pointers
    unsigned char *sourceDataDevice, *ch1DataDevice, *grey_and_blurDataDevice, *grey_and_threshDataDevice;

    // Now we need to create the cv::Mat objects at the pointers declared above.
    // These objects will be referenced by both the host and the device
    // The image data will start at the location of the pointer
    //
    // we are telling mat three things:
    // 1. the size of the image with frame_in.size()
    // 2. the type of image we are creating (i.e. 8 unsigned bytes with three channels)
    // 3. where to store the image. We are creating an "image buffer" in a location that can be accessed by both the
    //    host and the device

    // for the input image. use the cv::Mat::copyTo() function to move the incoming frame to source
    cv::Mat source  (frame_in.size(),
                     CV_8UC3,
                     createImageBuffer(frame_in.size().width * frame_in.size().height * 3, &sourceDataDevice));

    cv::Mat grey (frame_in.size(),
                  CV_8U,
                  createImageBuffer(frame_in.size().width * frame_in.size().height, &ch1DataDevice));


    cv::Mat greyblur (frame_in.size(),
                      CV_8U,
                      createImageBuffer(frame_in.size().width * frame_in.size().height, &grey_and_blurDataDevice));

    cv::Mat greythresh (frame_in.size(),
                      CV_8U,
                      createImageBuffer(frame_in.size().width * frame_in.size().height, &grey_and_threshDataDevice));


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
    int numThreadsX = 32;
    dim3 numThreads(numThreadsX,numThreadsX);
    int numBlocksX = (width  + numThreadsX - 1)/numThreadsX;
    int numBlocksY = (height + numThreadsX - 1)/numThreadsX;
    dim3 numBlocks(numBlocksX, numBlocksY);

    printf("\nnumBlocksX: %d\n",numBlocksX);
    printf("numBlocksY: %d\n",numBlocksY);
    printf("numBlocks: %d\n",numBlocks.x * numBlocks.y);
    printf("total threads: %d\n",(numBlocks.x * numBlocks.y)*numThreads.x*numThreads.y);


    // init keystroke variables
    char keystroke = 'g';  // default keystroke (GPU)

    while(1){

        //---------------------Keyboard Input Handling------------------------------------------------------------------
        // get the keystroke
        char tmp_key = (char) cv::waitKey(1);

        // check if a key has been pressed
        if (tmp_key != 255){
            keystroke = tmp_key;
            tmp_key = 255;
        }


        //---------------------Read in the image------------------------------------------------------------------------
        // direct image from camera to cv::Mat object
        cap >> frame_in;

        switch(keystroke) {
            case 'q' : {
                // ------------------QUIT CASE--------------------------------------------------------------------------
                printf("Exiting.\n");
                cap.release();
                exit(0);
            }
            case 'c' : {
                // ------------------CPU CASE---------------------------------------------------------------------------
                // make some containers
                cv::Mat frame_out_grey(cv::Size(width, height), CV_8UC1);
                cv::Mat frame_out_blur(cv::Size(width, height), CV_8UC1);
                cv::Mat frame_out_thresh(cv::Size(width, height), CV_8UC1);

                // apply the operaions
                cpu_greyscale(frame_in, frame_out_grey);
                cpu_blur(frame_out_grey, frame_out_blur);
                cpu_thresh(frame_out_grey, frame_out_thresh);
                cv::putText(frame_in, "CPU", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 200, 250), 1, CV_AA);
                cv::imshow("Source", frame_in);
                cv::imshow("ch1", frame_out_grey);
                cv::imshow("grey_and_blur", frame_out_blur);
                cv::imshow("grey_and_thresh", frame_out_thresh);
                break;
            }
            case 'g' : {

                // clone the incoming image to the mapped memory
                frame_in.copyTo(source);

                ch1 <<< numBlocks, numThreads >>> (ch1DataDevice, sourceDataDevice, width, height);
                gpu_grey_and_blur <<< numBlocks, numThreads >>>
                                                  (grey_and_blurDataDevice, sourceDataDevice, width, height);
                gpu_grey_and_thresh <<< numBlocks, numThreads >>>
                                                    (grey_and_threshDataDevice, sourceDataDevice, width, height);
                cudaThreadSynchronize();
                cv::putText(source, "GPU", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 200, 250), 1, CV_AA);
                cv::imshow("Source", source);
                cv::imshow("ch1", grey);
                cv::imshow("grey_and_blur", greyblur);
                cv::imshow("grey_and_thresh", greythresh);
                break;
            }
        }

    }

}