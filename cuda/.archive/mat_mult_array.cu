#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16

//============================= CPU ===========================================
void native_mat_mult(int a[m][n], int **b, int **result, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; h++)
            {
                tmp += a[h][i] * b[j][h];
            }
            result[i][j] = tmp;
        }
    }

}

//============================= GPU ===========================================
// To be run on the GPU, hence the __global__ tag
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
	//int *cpu_a, *cpu_b, *cpu_result, *cpu_c;
    int cpu_a[m][n], cpu_b[n][k], cpu_result[n][k], cpu_c[n][k];

//    // Allocate memory to the pointers on the host
//	cudaMallocHost((void **) &cpu_a, sizeof(int)*m*n);
//	cudaMallocHost((void **) &cpu_b, sizeof(int)*n*k);
//    cudaMallocHost((void **) &cpu_c, sizeof(int)*m*k);
//	cudaMallocHost((void **) &cpu_result, sizeof(int)*m*k);

    // Generate the matrices (with random numbers)
    // cpu_a
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cpu_a[i][j] = rand () % 1024;
		}
	}

    // cpu_b
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
                cpu_b[i][j] = rand () % 1024;
        }
    }


    // variable to keep track of time
	float cpu_time, gpu_time;

    // create start and stop events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // ================================ GPU =========================================

    // the final matrix will have size m x k
    // we need to spawn enough threads to compute all of the entries

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("---------------------------------------------\n");
    printf("The resulting matrix will be of size %d x %d\n", m, k);
    printf("\nI am launching a grid size of %d x %d blocks\n", grid_rows, grid_cols);
    printf("Each block will be %d x %d threads\n",BLOCK_SIZE,BLOCK_SIZE);
    printf("This will give you %d x %d available threads\n",grid_rows*BLOCK_SIZE,grid_cols*BLOCK_SIZE);
    printf("---------------------------------------------\n\n");
    printf("Press ENTER to begin computation on GPU...\n");
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
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernel
    // Kernels will always be launched using triple brackets
    // the first input in the triple brackets is the dimension of the grid
    // the second is the dimension of the block
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c, m, n, k);

    // Transefer results from device to host
    cudaMemcpy(cpu_c, gpu_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d * %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_time);

    printf("Press ENTER to begin computation on CPU...\n");
    getchar();

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
