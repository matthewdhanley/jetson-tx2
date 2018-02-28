#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16

void native_mat_mult(int *a, int *b, int *result, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; h++)
            {
                tmp += a[i * n + h] * b[h * k + j];
            }
            result[i * k + j] = tmp;
        }
    }

}

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k)
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

int main(int argc, char const *argv[])
{
	int m, n, k;
	printf("Input m n k::\n");
	scanf("%d %d %d", &m, &n, &k);


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
	float cpu_time, gpu_time;

    // create start and stop events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // ******************************************************************************
    // ================================ GPU =========================================
    // ******************************************************************************

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

    // calculate grid size needed
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernel
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c, m, n, k);

    // Transefr results from device to host
    cudaMemcpy(cpu_c, gpu_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d * %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_time);

    // cpu - start recording time at 0
	cudaEventRecord(start, 0);

    // multiply matricies
	native_mat_mult(cpu_a, cpu_b, cpu_result, m, n, k);

    // stop hte timer and put result into cpu_time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_time, start, stop);

    // display timing result
	printf("Time on mat mult of %dx%d * %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_time);

    // free host memory
	cudaFreeHost(cpu_a);
	cudaFreeHost(cpu_b);
	cudaFreeHost(cpu_result);

    // fin
	return 0;
}
