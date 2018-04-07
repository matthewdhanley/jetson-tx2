#include <stdio.h>


/*
The "__global__" tag tells nvcc that the function will execute on the device
but will be called from the host. Notice that we must use pointers!
*/
__global__
void add_int( int *a, int *b, int *c){
  *c = *a + *b;
  printf("blockIdx: %d\n",blockIdx.x);
}

// Main program
int main(void){

  //host memory != device memory, must allocate differently
  //device pointers point to GPU Memory
  //host pointers point to CPU memory
  int a, b, c;                //host copies
  int *dev_a, *dev_b, *dev_c; //device copies
  int size = sizeof( int );   //size of an interger

  //allocate space on device
  cudaMalloc( (void**)&dev_a, size );
  cudaMalloc( (void**)&dev_b, size );
  cudaMalloc( (void**)&dev_c, size );

  a = 2; //storing values in host
  b = 7;

  // now we need the values to be copied to the device
  cudaMemcpy( dev_a, &a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( dev_b, &b, size, cudaMemcpyHostToDevice );

  // launch the add_int kernel on the GPU
  add_int<<<3,1>>>(dev_a, dev_b, dev_c);

  //now we want the values back on the CPU
  cudaMemcpy( &c, dev_c, size, cudaMemcpyDeviceToHost );

  printf("C: %d\n",c);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);


  // your basic hello world program
  printf("Hello, World!\n");
  return 0;
}
