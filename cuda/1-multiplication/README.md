# Matix Multiplication Example

## Naive implementation
When running a basic matrix multiplication example, the average time to multiply two random 500x500 element arrays on the gpu was _78.2 ms_ whereas on the CPU the time was on the order of _13000 ms_!


This is the kernel that achieved these stats with a blocksize of 16
``` C
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
```
## Tiling
Next, we will implement the tiling algorithm. Here is the new kernel. This brought our average time for 500x500 matrix multiplications down to _32.9 ms_. Much better! But can we do better yet?
``` C
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
```
## Intelligent Resource Management
Using intelligent resource management, I was able to bring the time of multiplying 500x500 element matricies down to _31.9 ms_. Not an incredible speedup.
Things we need to take advantage of
1. Registers
2. Shared Memory
3. Minimizing flops
4. Minimizing global memory accesses

https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
