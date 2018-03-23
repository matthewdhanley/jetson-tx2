//
// Created by matt on 3/18/18.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// -----------------CONSTANT MEMORY-------------------------------------
#define BLOCK_SIZE 32

// size of mask dimensions
#define MASK_SIZE 5


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


// constant memory on host to save kernel for fast access
__constant__ float M[MASK_SIZE * MASK_SIZE];

// -------------------------------------------- GPU KERNELS ------------------------------------------------------------

// CONVOLUTION
__global__ void gpu_convolve2d(unsigned char* Pout, unsigned char* Pin,
                               int width, int height) {

    // index into the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    // commonly used operation so only do it once and store it in register
    unsigned char n = MASK_SIZE / 2;

    // shared memory to store tile
    __shared__ float Pout_ds[BLOCK_SIZE * BLOCK_SIZE];

    // load tiles into shared memory
    if (x < width && y < height) {
//        Pout_ds[threadIdx.y * blockDim.y + threadIdx.x] = Pin[i];
        Pout_ds[threadIdx.y * blockDim.x + threadIdx.x] = float(Pin[i]);
    }

    __syncthreads();  // execution barrier


    // remember that we are tiling with blocks size equal to tile size
    // we need to create bounds for the tiles

    float P_sum = 0.0;

    // loop through rows of mask
    for (int j = -n; j < n + 1; j++) {
        // loop through columns of mask
        for (int k = n; k < n + 1; k++) {

            // get vars, delete if too many registers being used.
            // index into the tile
            int x_index = threadIdx.x + k;
            int y_index = threadIdx.y + j;

            // check that we are in bounds of the shared memory
            if (x_index > -1 &&
                    x_index < blockDim.x &&
                    y_index > -1 &&
                    y_index < blockDim.y) {

                // accumulate with weight from mask "M"
                P_sum += Pout_ds[y_index * blockDim.x + x_index] * M[(n + j)*MASK_SIZE + n + k];
            }
                // check that we're within bounds of the image, otherwise just make the pixel zero
            else if(x + k > -1 &&
                    x + k < width &&
                    y + j > -1 &&
                    y + j < height){

                // grab from global memory (probably cached since other threads are likely pulling this data)
                // accumulate wieth weight from mask "M"
                P_sum += float(Pin[(y+j)*width + x + k]) * M[(n + j)*MASK_SIZE + n + k];
            }
        }
    }

    // make sure that the pixel is within the range of the picture
    if (x < width && y < height) {
        Pout[i] = (unsigned char) P_sum;
    }
}


// ------------------------------------more naive version of the blur --------------------------------------------------
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

// KERNEL TO PLAY WITH
__global__ void gpu_test(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int i = row * width + col;

    if (row < height && col < width) {
        Pout[i] = Pin[i];
    }

}

// --------------------------------------cpu functions------------------------------------------------------------------

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


// -----------MAIN-------------------MAIN-----------------MAIN----------------MAIN-------------MAIN---------------------

int main(){

    float bottom = 7.0;
    // ------------------------------- create a mask for convolving! ---------------------------------------------------
    float convolve_mask5x5[25] = {
            1.f/bottom, 1.f/bottom, 1.f/bottom, 1.f/bottom, 1.f/bottom,
            1.f/bottom, 2.f/bottom, 2.f/bottom, 2.f/bottom, 1.f/bottom,
            1.f/bottom, 2.f/bottom, 3.f/bottom, 2.f/bottom, 1.f/bottom,
            1.f/bottom, 2.f/bottom, 2.f/bottom, 2.f/bottom, 1.f/bottom,
            1.f/bottom, 1.f/bottom, 1.f/bottom, 1.f/bottom, 1.f/bottom
    };

    // copy the mask from the host to the device constant memory
    cudaMemcpyToSymbol(M, convolve_mask5x5, MASK_SIZE * MASK_SIZE * sizeof(float));

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
    unsigned int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    unsigned int pixels = width * height;

    // print info about the picture
    printf("Frame Size: %d x %d\n %d pixels\n", width, height, pixels);

    // create opnecv Mat container for image
    cv::Mat frame_in;

    // grab the first frame, this will also automatically size the container
    cap >> frame_in;

    // create windows for different filters
    cv::namedWindow("Source");  // this window will hold the source image
    cv::namedWindow("Result");  // this window will hold the resultant image


    // ------------------- ALLOCATE GPU MEMORY ------------------------------------
    unsigned char *d_Pin, *d_Pout;  // init pointers
    gpuErrchk( cudaMalloc((void **) &d_Pin, sizeof(unsigned char)*pixels) );  // allocate space on device for input
    gpuErrchk( cudaMalloc((void **) &d_Pout, sizeof(unsigned char)*pixels) ); // allocate space on device for output


    // ------------------- ALLOCATE CPU MEMORY -------------------------------------
    // note that input image has space allocated already by frame_in
    unsigned char *h_Pout = (unsigned char *) malloc(pixels*sizeof(char)); // allocate space for output


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

    printf("\nnumBlocksX: %d\n",numBlocksX);
    printf("numBlocksY: %d\n",numBlocksY);
    printf("numBlocks: %d\n",numBlocks.x * numBlocks.y);
    printf("threads per block: %d\n", numThreads.x * numThreads.y);
    printf("total threads: %d\n",(numBlocks.x * numBlocks.y)*numThreads.x*numThreads.y);


    printf("\n\n\n\n===========================================================\n"
                   " TOGGLE WITH \"1\", \"2\", and \"3\" on your keyboard!\n"
                   "\"q\" to quit\n"
                   "\"c\" for CPU\n"
                   "===========================================================\n\n\n\n");

    // init keystroke variables
    char keystroke = '1';  // default keystroke (GPU)

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
        cv::cvtColor(frame_in, frame_in, CV_BGR2GRAY);

        switch(keystroke) {
            case 'q' : {
                // ------------------QUIT CASE--------------------------------------------------------------------------
                printf("Exiting.\n");

                // free everything
                cap.release();
                cudaFree(d_Pin);
                cudaFree(d_Pout);
                free(h_Pout);

                // exit the program
                exit(0);
            }

            case 'c' : {
                // ------------------CPU CASE---------------------------------------------------------------------------
                // apply the operations
                cv::Mat frame_out(cv::Size(width, height), CV_8UC1);
                cpu_blur(frame_in, frame_out);
                // place some text on the image
                cv::putText(frame_out, "CPU Blur", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1, CV_AA);

                // show the images
                cv::imshow("Source", frame_in);
                cv::imshow("Result", frame_out);
                break;
            }

            case '1' : {
                // ------------------ GPU CONVOLVE ---------------------------------------------------------------------
                // copy memory from host to device
                gpuErrchk( cudaMemcpy(d_Pin, frame_in.data, pixels * sizeof(unsigned char), cudaMemcpyHostToDevice) );

                // execute gpu kernel
                gpu_convolve2d <<< numBlocks, numThreads >>> (d_Pout, d_Pin, width, height);
                gpuErrchk( cudaPeekAtLastError() );

                // make sure all the threads finished
                gpuErrchk( cudaThreadSynchronize() );

                // copy memory back to host from device
                gpuErrchk( cudaMemcpy(h_Pout, d_Pout, sizeof(unsigned char) * pixels, cudaMemcpyDeviceToHost) );

                // create container for new image and insert the data
                cv::Mat frame_out(cv::Size(width, height), CV_8UC1, h_Pout);

                // place some text on the image
                cv::putText(frame_out, "GPU Convolve", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1, CV_AA);

                // show the images
                cv::imshow("Source", frame_in);
                cv::imshow("Result", frame_out);
                break;
            }

            case '2' : {
                // ------------------ GPU Naive Blur -------------------------------------------------------------------
                // copy memory from host to device
                gpuErrchk( cudaMemcpy(d_Pin, frame_in.data, pixels * sizeof(unsigned char), cudaMemcpyHostToDevice) );

                // execute gpu kernel
                gpu_blur <<< numBlocks, numThreads >>> (d_Pout, d_Pin, width, height);

                gpuErrchk( cudaGetLastError() );

                // make sure all the threads finished
                gpuErrchk( cudaThreadSynchronize() );

                // copy memory back to host from device
                gpuErrchk( cudaMemcpy(h_Pout, d_Pout, sizeof(unsigned char) * pixels, cudaMemcpyDeviceToHost) );

                // create container for new image and insert the data
                cv::Mat frame_out(cv::Size(width, height), CV_8UC1, h_Pout);

                // place some text on the image
                cv::putText(frame_out, "GPU Naive Blur", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1, CV_AA);

                // show the images
                cv::imshow("Source", frame_in);
                cv::imshow("Result", frame_out);
                break;
            }

            case '3' : {
                // ------------------ GPU Naive Blur -------------------------------------------------------------------
                // copy memory from host to device
                gpuErrchk( cudaMemcpy(d_Pin, frame_in.data, pixels * sizeof(unsigned char), cudaMemcpyHostToDevice) );

                // execute gpu kernel
                gpu_test <<< numBlocks, numThreads >>> (d_Pout, d_Pin, width, height);

                gpuErrchk( cudaGetLastError() );

                // make sure all the threads finished
                gpuErrchk( cudaThreadSynchronize() );

                // copy memory back to host from device
                gpuErrchk( cudaMemcpy(h_Pout, d_Pout, sizeof(unsigned char) * pixels, cudaMemcpyDeviceToHost) );

                // create container for new image and insert the data
                cv::Mat frame_out(cv::Size(width, height), CV_8UC1, h_Pout);

                // place some text on the image
                cv::putText(frame_out, "GPU Test", cv::Point(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1, CV_AA);

                // show the images
                cv::imshow("Source", frame_in);
                cv::imshow("Result", frame_out);
                break;
            }

            default: {
                cv::Mat frame_out(cv::Size(width, height), CV_8UC1, h_Pout);
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