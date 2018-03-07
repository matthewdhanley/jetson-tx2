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

__global__ void gpu_matrix_mult_two(float *d_M, float *d_N, float *d_P, int m, int n, int k){

    // shared memory for tiling
    __shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // recall that TILE_WIDTH = blockDim
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float pval = 0;

    // this loop is iterating through cols of M and rows of N
    // recall that n is the shared inner dimension, that's why we're using it
    // to define our loop size
    for (int ph = 0; ph < ceil(n / (float) TILE_WIDTH); ph++){
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
