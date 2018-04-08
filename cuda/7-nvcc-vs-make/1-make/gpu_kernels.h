//
// Created by matt on 4/7/18.
//
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "main.h"

#ifndef JETSON_TX2_GPU_KERNELS_H
#define JETSON_TX2_GPU_KERNELS_H
extern "C" void cuda_mask(unsigned char *Pout, unsigned char* Pin_hsv, unsigned char* Pin_rgb,
                          int width, int height, colorMaskHSV colorMask, centroid* c, unsigned int* countNonZeroBlocks,
                          dim3 numBlocks, dim3 numThreads);
#endif //JETSON_TX2_GPU_KERNELS_H
