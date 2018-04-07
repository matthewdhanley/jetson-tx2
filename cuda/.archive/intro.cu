#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>

//============================= GPU Kernel ====================================
__global__
void d_add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

//============================= CPU ===========================================
// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  // loop through array and add element by element
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}


// CPU WRAPPER
int add_cpu(void){
  int N = 1<<25; // 1M elements
  printf("Number of elements: %d\n",N);

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  clock_t begin = clock();
  add(N, x, y);
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("CPU Time: %f seconds\n", time_spent);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}

// GPU WRAPPER
int add_gpu(void){

  int N = 1<<25;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  clock_t begin = clock();

  d_add<<<numBlocks, blockSize>>>(N, x, y);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("GPU Time: %f seconds\n", time_spent);


  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}


int main(void)
{
  add_cpu();
  add_gpu();
}
