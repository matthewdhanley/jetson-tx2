//
// Created by matt on 3/4/18.
//

#include <cuda_runtime_api.h>
#include <cuda.h>

extern "C" void cuda_big_add(int *a, int *b, int *c, unsigned int N, dim3 numBlocks, dim3 numThreads);