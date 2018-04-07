//
// Created by matt on 3/4/18.
//

#include <cuda_runtime_api.h>
#include <cuda.h>

extern "C" void cuda_ch1(unsigned char* Pout, unsigned char* Pin, int width, int height, dim3 blocks, dim3 block_size);
extern "C" void cuda_blur(unsigned char* Pout, unsigned char* Pin, int width, int height, dim3 numBlocks, dim3 numThreads);
extern "C" void cuda_grey_and_blur(unsigned char* Pout, unsigned char* Pin, int width, int height, dim3 numBlocks, dim3 numThreads);
extern "C" void cuda_grey_and_thresh(unsigned char* Pout, unsigned char* Pin, int width, int height, dim3 numBlocks, dim3 numThreads);
