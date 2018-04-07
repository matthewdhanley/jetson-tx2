//
// Created by matt on 3/18/18.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// -------------------------------------CONSTANT MEMORY---------------------------------------------------------------
// size of mask dimensions
#define MASK_SIZE 5

// constant memory on host to save kernel for fast access
__constant__ unsigned char M[MASK_SIZE * MASK_SIZE];

// ----------------------------gpu kernels----------------------------------------------------------------------------

__global__ void gpu_convolve2d(unsigned char* Pout, unsigned char* Pin, int width, int height, int mask_sum) {
    // index into the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    unsigned char n = MASK_SIZE / 2;

    __shared__ unsigned char Pout_ds[16*16];

    // load tiles into shared memory
    if (x < width && y < height){
        Pout_ds[threadIdx.y * blockDim.y + threadIdx.x] = Pin[i];
    }

    __syncthreads();  // execution barrier


    // remember that we are tiling with blocks size equal to tile size
    // we need to create bounds for the tiles
    int x_tile_start = blockIdx.x * blockDim.x;
    int x_tile_stop = (blockIdx.x + 1) * blockDim.x;

    int y_tile_start = blockIdx.y * blockDim.y;
    int y_tile_stop = (blockIdx.y + 1) * blockDim.y;

    int x_start_point = x - MASK_SIZE / 2;
    int y_start_point = y - MASK_SIZE / 2;

    int P_sum = 0;

    for (int j = 0; j < MASK_SIZE; j++){
        for (int k = 0; k < MASK_SIZE; k++) {
            int x_index = x_start_point + j;
            int y_index = y_start_point + k;

            // check that we are in bounds of the picture
            if (x_index >= 0 && x_index < width && y_index >= 0 && y_index < height){

                // check for skirt cells
                // if it is a skirt, load it from global memory. It should then be cached in L2 which
                // will allow it to be accessed quickly as an internal cell in another kernel
                if ((x_index >= x_tile_start) &&
                    (y_index >= y_tile_start) &&
                    (x_index < x_tile_stop) &&
                    (y_index < y_tile_stop)){

                    // grab pic values from shared memory
                    P_sum += Pout_ds[(threadIdx.y * blockDim.y - n + k) + (threadIdx.x - n + j)] * M[(n + k) * MASK_SIZE + (n + j)];
                }
                else{
                    // grab from global memory
                    P_sum += Pin[(y + k) * width + (x + j)] * M[(n + k) * MASK_SIZE + (n + j)];
//                    P_sum = 157;
                }
            }

        }
    }

    if (x < width && y < height) {
//        Pout[i] = Pout_ds[threadIdx.y * blockDim.y + threadIdx.x];
        Pout[i] = P_sum / mask_sum;
//        Pout[i] = (unsigned char) threadIdx.x * threadIdx.y;
    }

    if (x % 32 == 0 || y % 32 == 0){
        Pout[i] = 255;
    }
}

// --------------------------------------cpu functions------------------------------------------------------------------
void cpu_convolve2d(cv::Mat src, cv::Mat dst, char *mask){
   printf("CPU\n");
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

    // create a mask for convolving!
    char convolve_mask5x5[25] = {
            1, 1, 1, 1, 1,
            1, 2, 2, 2, 1,
            1, 2, 3, 2, 1,
            1, 2, 2, 3, 1,
            1, 1, 1, 1, 1
    };

//    unsigned char convolve_mask5x5[25] = {
//            1, 1, 0, 1, 1,
//            1, 2, 0, 2, 1,
//            0, 0, 0, 0, 0,
//            1, 2, 0, 2, 1,
//            1, 1, 0, 1, 1
//    };


    int mask_sum;
    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++){
        mask_sum += convolve_mask5x5[i];
    }

    // copy the mask from the host to the device constant memory
    cudaMemcpyToSymbol(M, convolve_mask5x5, MASK_SIZE * MASK_SIZE * sizeof(char));

    // getting the camera stream----------------------------------------------------------------------------------------
    const char* gst;
    gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! \
                nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
                videoconvert ! video/x-raw, format=(string)BGR ! \
                appsink";

    // open the camera with the gst string
    cv::VideoCapture cap(gst);

    // error handling
    if(!cap.isOpened()) {
        std::cout << "Failed to open camera." << std::endl;
        return -1;
    }

    // create container for image
    cv::Mat frame_in;

    // get info about the picture
    unsigned int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    unsigned int pixels = width * height;

    // print info about the picture
//    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<std::endl;
    printf("Frame Size: %d x %d\n %d pixels\n", width, height, pixels);

    // create windows for different filters
    cv::namedWindow("Source");
    cv::namedWindow("Result");
    
    // grab the first frame
    cap >> frame_in;

    // we will be pointing the image buffers to these pointers
    unsigned char *sourceDataDevice, *convolveDataDevice;

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
                     CV_8U,
                     createImageBuffer(frame_in.size().width * frame_in.size().height, &sourceDataDevice));

    cv::Mat convolve (frame_in.size(),
                  CV_8U,
                  createImageBuffer(frame_in.size().width * frame_in.size().height, &convolveDataDevice));



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
    int numThreadsX = 16;
    dim3 numThreads(numThreadsX, numThreadsX);
    int numBlocksX = (width)/numThreadsX;
    int numBlocksY = (height)/numThreadsX;
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
        // convert color to bw
        cv::cvtColor(frame_in, source, CV_BGR2GRAY);

        switch(keystroke) {
            case 'q' : {
                // ------------------QUIT CASE--------------------------------------------------------------------------
                printf("Exiting.\n");
                cap.release();
                exit(0);
            }

            case 'c' : {
                // ------------------CPU CASE---------------------------------------------------------------------------
                // apply the operations
                cpu_convolve2d(source, convolve, convolve_mask5x5);


                cv::putText(convolve, "CPU", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1, CV_AA);

                cv::imshow("Source", source);
                cv::imshow("Result", convolve);
                break;
            }

            case 'g' : {

                gpu_convolve2d <<< numBlocks, numThreads >>> (convolveDataDevice, sourceDataDevice, width, height, mask_sum);
                cudaThreadSynchronize();

                cv::putText(convolve, "GPU", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1, CV_AA);

                cv::imshow("Source", source);
                cv::imshow("Result", convolve);
                break;
            }
        }

    }

}