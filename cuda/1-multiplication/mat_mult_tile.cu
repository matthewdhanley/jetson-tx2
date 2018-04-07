#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define TILE_WIDTH 16

__global__ void gpu_matrix_mult_one(int *a, int *b, int *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // get the row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // get the column
    int sum = 0; // initialize the sum

    if( col < k && row < m) // check to make sure that the thread needs to compute
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void gpu_matrix_mult_two(int *d_M, int *d_N, int *d_P, int m, int n, int k)
{

    // shared memory for tiling
    __shared__ int Mds [TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds [TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // recall that TILE_WIDTH = blockDim
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int pval = 0;

    // this loop is iterating through cols of M and rows of N
    // recall that n is the shared inner dimension, that's why we're using it
    // to define our loop size
    for (int ph = 0; ph < n / TILE_WIDTH; ph++){
        // boundary check for shared Mds
        if (row < k && ph * TILE_WIDTH + tx < m){
            // saving tile from M
            /* indexing thought exercise:
             * "row * k" gets us to our desired row in M
             * adding "ph * TILE_WIDTH" moves our tile over to the desired tile location
             * adding "tx" moves us to the desired location within the tile
             * */
            Mds[ty][tx] = d_M[row * k + ph * TILE_WIDTH + tx];
        }
        // boundary check
        if (ph*TILE_WIDTH + ty < k && col < m){
            // saving tile from N
            /* indexing thought exercise:
             * "ph * TILE_WIDTH" moves the tile "down" to the desired location
             * adding "ty" gets us to the desired location within the tile
             * multiplying by "k" does the magic (remember row major order)
             * adding col moves the tile to the desired column*/
            Nds[ty][tx] = d_N[(ph * TILE_WIDTH + ty) * k + col];
        }

        __syncthreads();  // execution barrier

        for (int j = 0; j < TILE_WIDTH; j++){
            // performing part of inner product
            pval += Mds[ty][j] * Nds[j][tx];
        }

        __syncthreads();
    }
    if (row < k && col < m){
        d_P[row * k + col] = pval;
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k; // init matrix dimensions
    printf("---------------------------------------------\n");
    printf("We will be multiplying two matrices\n");
    printf("The first will be of size m x n\n");
    printf("The second will be of size n x k\n");
    printf("I will have you choose these dimensions!\n");
    printf("---------------------------------------------\n\n");
    printf("Input m:\n");
    scanf("%d", &m);
    printf("\nInput n:\n");
    scanf("%d", &n);
    printf("\nInput k:\n");
    scanf("%d", &k);
    printf("\n");


    // Initialize pointers
    int *cpu_a, *cpu_b, *cpu_result, *cpu_c;

    // Allocate memory to the pointers on the host
    cudaMallocHost((void **) &cpu_a, sizeof(int)*m*n); // matrix a
    cudaMallocHost((void **) &cpu_b, sizeof(int)*n*k); // matrix b
    cudaMallocHost((void **) &cpu_c, sizeof(int)*m*k); // cpu memory for gpu result
    cudaMallocHost((void **) &cpu_result, sizeof(int)*m*k); // cpu result

    // Generate the matrices
    // cpu_a
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cpu_a[i * n + j] = rand () % 1024;
        }
    }

    // cpu_b
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            cpu_b[n * k + j] = rand () % 1024;
        }
    }


    // variable to keep track of time
    float gpu_time;

    // create start and stop events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ******************************************************************************
    // ================================ GPU =========================================
    // ******************************************************************************

    // the final matrix will have size m x k
    // we need to spawn enough threads to compute all of the entries

    unsigned int grid_rows = (m + TILE_WIDTH - 1) / TILE_WIDTH;
    unsigned int grid_cols = (k + TILE_WIDTH - 1) / TILE_WIDTH;

    printf("---------------------------------------------\n");
    printf("The resulting matrix will be of size %d x %d\n", m, k);
    printf("\nI am launching a grid size of %d x %d blocks\n", grid_rows, grid_cols);
    printf("Each block will be %d x %d threads\n",TILE_WIDTH,TILE_WIDTH);
    printf("This will give you %d x %d available threads\n",grid_rows*TILE_WIDTH,grid_cols*TILE_WIDTH);
    printf("---------------------------------------------\n\n");
    printf("Press ENTER to begin computation on GPU (w/o tiling)...\n");
    getchar();
    getchar();

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    // Allocate memory space on the device
    int *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void **) &gpu_a, sizeof(int)*m*n);
    cudaMalloc((void **) &gpu_b, sizeof(int)*n*k);
    cudaMalloc((void **) &gpu_c, sizeof(int)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(gpu_a, gpu_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, gpu_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    // Launch kernel
    // Kernels will always be launched using triple brackets
    // the first input in the triple brackets is the dimension of the grid
    // the second is the dimension of the block
    gpu_matrix_mult_one<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c, m, n, k);

    // Transefer results from device to host
    cudaMemcpy(cpu_c, gpu_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d * %dx%d on GPU WITHOUT tiling: %f ms.\n\n", m, n, n, k, gpu_time);

    printf("Press ENTER to begin computation on GPU (w/tiling)...\n");
    getchar();

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

    // copy matrix A and B from host to device memory
    cudaMemcpy(gpu_a, gpu_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, gpu_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    // Launch kernel
    // Kernels will always be launched using triple brackets
    // the first input in the triple brackets is the dimension of the grid
    // the second is the dimension of the block
    gpu_matrix_mult_two<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c, m, n, k);

    // Transefer results from device to host
    cudaMemcpy(cpu_c, gpu_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d * %dx%d on GPU WITH tiling: %f ms.\n\n", m, n, n, k, gpu_time);

    // fin
    return 0;
}