//
// Created by matt on 4/7/18.
//
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "main.h"

#ifndef JETSON_TX2_GPU_KERNELS_H
#define JETSON_TX2_GPU_KERNELS_H
extern "C" void cuda_mask(bool *mask, unsigned char* Pin_hsv, int width, int height, colorMaskHSV colorMask,
                          dim3 numBlocks, dim3 numThreads);

extern "C" void cuda_erode(bool *mask, int width, int height, dim3 numBlocks, dim3 numThreads);

extern "C" void cuda_applyMask(unsigned char* Pout, unsigned char* Pin_rgb, bool*mask, int width, int height,
                               dim3 numBlocks, dim3 numThreads);

extern "C" void cuda_gpu_centroid(bool* mask, int width, int height, int val, int gridStride, centroid* c,
                             unsigned int* countNonZeroBlocks, dim3 numBlocks, dim3 numThreads);

#endif //JETSON_TX2_GPU_KERNELS_H
