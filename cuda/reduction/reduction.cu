/* This program uses the method of reduction to add all elements of an array
 *
 */
#include <stdio.h>

// =========== GLOBALS =========================
const int N = 200; // number of elements in array

// this needs to be a power of 2
const int threadsPerBlock = 256;

// Calculate number of blocks needed
const int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

// GPU Kernel
__global__ void reduce(int *a, int *res){
    // create shared memory for the threads in the block
    __shared__ int cache[threadsPerBlock];

    // get the thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // index into the cache for this block
    int cacheIndex = threadIdx.x;

    // set the value in cache
    cache[cacheIndex] = a[tid];

    __syncthreads(); //synchronize threads before continuing

    int i = blockDim.x/2; // only want first half to do work
    while( i != 0 ){
        if (cacheIndex < i) // make sure we are not doing bogus add

            // add the current index and ith element
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads(); // we want all threads to finish
        i /= 2;
    }
    if (cacheIndex == 0) // only one thread needs to do this
        *res = cache[0];
}

int main(void){
    // initialize pointers
    int *a, *res;
    int *d_a, *d_res;

    // allocate cpu memory
    a = (int*)malloc(N*sizeof(int));
    res = (int*)malloc(sizeof(int));

    // allocate memory on GPU
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_res, sizeof(int));

    // fill in "a" array
    for (int i=0; i<N; i++){
        a[i] = 2;
    }

    // copy from host to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, sizeof(int), cudaMemcpyHostToDevice);

    // run kernel
    reduce<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_res);

    // copy memory from gpu to cpu
    cudaMemcpy(res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

    // print reslut
    printf("Sum: %d\n", *res);

    // clean up
    cudaFree(d_a);
    cudaFree(d_res);
    free(a);
    free(res);

}
