#include <stdio.h>

/*
The "__global__" tag tells nvcc that the function will execute on the device
but will be called from the host. Notice that we must use pointers!
*/
__global__
void add_vec( int *a, int *b, int *c){
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
  printf("%d\n",blockIdx.x);
}

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = rand();
}

#define N 512

// Main program
int main(void){

  //host memory != device memory, must allocate differently
  //device pointers point to GPU Memory
  //host pointers point to CPU memory
  int *a, *b, *c;                //host copies
  int *dev_a, *dev_b, *dev_c; //device copies
  int size = N * sizeof( int );   //size of an interger

  //allocate space on device
  cudaMalloc( (void**)&dev_a, size );
  cudaMalloc( (void**)&dev_b, size );
  cudaMalloc( (void**)&dev_c, size );

  //allocate cpu memory
  a = (int*)malloc( size );
  b = (int*)malloc( size );
  c = (int*)malloc( size );


  //generate random numbers in arrays
  random_ints(a,N);
  random_ints(b,N);

  // now we need the values to be copied to the device
  cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

  // launch the add_int kernel on the GPU with N blocks
  add_vec<<<N,1>>>(dev_a, dev_b, dev_c);

  //now we want the values back on the CPU
  cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost );

  //be free!
  free(a);
  free(b);
  free(c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);


  // your basic hello world program
  printf("Hello, World!\n");
  return 0;
}
