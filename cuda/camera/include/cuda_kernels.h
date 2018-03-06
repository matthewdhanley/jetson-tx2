#include <cuda_runtime_api.h>
#include <cuda.h>


extern "C" void cuda_ch1(unsigned int* Pout, unsigned int* Pin, int width, int height, dim3 blocks, dim3 block_size);