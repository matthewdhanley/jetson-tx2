// GPU Kernel
__global__ void big_add(int *a, int *b, int *c, unsigned int N){
    // init thread id
    int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    // stride is for big arrays, i.e. bigger than threads we have
    int stride = blockDim.x * gridDim.x;

    // do the operations
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += stride;
    }
}

extern "C" void cuda_big_add(int *a, int *b, int *c, unsigned int N, dim3 numBlocks, dim3 numThreads){
    big_add <<< numBlocks, numThreads >>> (a, b, c, N);
}