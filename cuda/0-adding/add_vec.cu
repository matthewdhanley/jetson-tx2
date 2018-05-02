/*
 * Name: add_vec.cu
 * Description: "Hello World" CUDA program to add two vectors
 */
#include <stdio.h>


//============================= GPU Kernel ====================================
/*
The "__global__" tag tells nvcc that the function will execute on the device
but will be called from the host. Notice that we must use pointers!
*/
/*
 * function: add_vec
 * purpose: add two vectors on GPU
 * PARAMETERS:
 *  a - first array
 *  b - second array
 *  c - output array
 */
__global__
void add_vec( int *a, int *b, int *c){
    // index into array using blockIdx.x and add that element
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

//============================= CPU ===========================================
/*
 * Name: random_ints
 * purpose: generate array of random numbers
 * PARAMETERS:
 *  a - pointer to array of size N
 *  N - length of array
 */
void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = rand();
}

// defining global variable "N" which defines size of arrays.
#define N 32

//=============================================================================
//============================= MAIN ==========================================
int main(void){

    //host memory != device memory, must allocate differently
    //device pointers point to GPU Memory
    //host pointers point to CPU memory
    int *a, *b, *c;                //host copies
    int *dev_a, *dev_b, *dev_c;    //device copies
    int size = N * sizeof( int );  //size of an interger

    //allocate space on device
    cudaMalloc( (void**) &dev_a, size );
    cudaMalloc( (void**) &dev_b, size );
    cudaMalloc( (void**) &dev_c, size );

    //allocate cpu memory
    a = (int*) malloc( size );
    b = (int*) malloc( size );
    c = (int*) malloc( size );

    // generate random numbers in input arrays
    random_ints(a,N);
    random_ints(b,N);

    // now we need the values to be copied to the device
    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

    // launch the add_int kernel on the GPU
    // using block size of N with 1 block
    add_vec <<< N, 1 >>> (dev_a, dev_b, dev_c);

    //now we want the values back on the CPU
    cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost );

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
