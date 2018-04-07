#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// =========== GLOBALS =========================
const int m = 50; // number of elements in array
const int n = 10; // number of elements in array
const int k = 10; // number of elements in array
const int sizeZ = 64; // number of threads in z direction. NEEDS TO BE CONST GLOBAL

// GPU Kernel
__global__ void matmul(int *a, int *b, int *result){

    // init the thread ids
    int tidx, tidy, tidz, cacheIndex;
    tidx = blockIdx.x * blockDim.x + threadIdx.x;
    tidy = blockIdx.y * blockDim.y + threadIdx.y;
    tidz = threadIdx.z;
    cacheIndex = threadIdx.z;

    // set up the shared memory for each block
    __shared__ int cache[sizeZ];

    while (tidz < n){
        cache[cacheIndex] += a[tidx * n + tidz] * b[tidz * k + tidy];
        tidz += blockDim.z;
    }

    // make sure all threads are finished
    __syncthreads();

    // reduction
    int i = blockDim.z/2;
    while(i != 0){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }

        // make sure all threads are caught up.
        __syncthreads();

        i /= 2; // iterate i
    }

    // "master" thread sets value of result
    if(cacheIndex == 0){
        result[tidx * k + tidy] = cache[0];
        if(sizeZ % 2 == 1){
            result[tidx * k + tidy] += cache[sizeZ-1];
        }
    }

    __syncthreads();
}

int main(void){
    // initialize pointers
    int *a, *b, *res;       // host
    int *d_a, *d_b, *d_res; // device

    // allocate cpu memory (host)
    a = (int*)malloc(m * k * sizeof(int));
    b = (int*)malloc(n * k * sizeof(int));
    res = (int*)malloc(m * k * sizeof(int));

    // allocate memory on GPU
    cudaMalloc(&d_a, m * n * sizeof(int));
    cudaMalloc(&d_b, n * k * sizeof(int));
    cudaMalloc(&d_res, m * k * sizeof(int));

    // set up rng
    srand(time(NULL));

    // fill in "a" array
    for (int i=0; i<m * n; i++){
        a[i] = rand() % 100; // some number less than 100
//        a[i] = 1;
    }
    // fill in "b" array
    for (int i=0; i<n * k; i++){
        b[i] = rand() % 100; // some number less than 100
//        b[i] = 1;
    }

    // copy from host to device
    printf("m * n: %d\n",m * n);
    printf("n * k: %d\n",k * n);

    cudaMemcpy(d_a, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * k * sizeof(int), cudaMemcpyHostToDevice);

    const dim3 threadsPerBlock(1, 1, sizeZ);

    printf("threadsPerBlock.z: %d\n", threadsPerBlock.z);

    // Calculate number of blocks needed
    const dim3 blocksPerGrid(m, k);
    printf("blocksPerGrid.x: %d\n", m);

    // run kernel
    matmul<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_b, d_res);

    // copy memory from gpu to cpu
    cudaMemcpy(res, d_res, m * k * sizeof(int), cudaMemcpyDeviceToHost);

    // print reslut
    printf("Top Left of Result:\n[[%d %d %d . . .\n [%d %d %d . . .]\n [%d %d %d . . .]\n  . . . . . .]]\n",
           res[0], res[1], res[2],
           res[k], res[k + 1], res[k + 2],
           res[2 * k], res[2 * k + 1], res[2 * k + 2]);

    // clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
}
