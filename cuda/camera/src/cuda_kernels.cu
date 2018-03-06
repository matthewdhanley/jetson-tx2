__global__ void ch1(unsigned int* Pout, unsigned int* Pin, int width, int height) {
    int channels = 3;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    // check if pixel within range
    if (col < width && row < height){
        int gOffset = row*width + col;
        int rgbOffset = gOffset * channels;
        unsigned char r = Pin[rgbOffset  ];
        unsigned char b = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];
        Pout[gOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}


extern "C" void cuda_ch1(unsigned int* Pout, unsigned int* Pin, int width, int height, dim3 numBlocks, dim3 numThreads)
{
    grayscale <<< numBlocks, numThreads >>> (Pout, Pin, width, height);
}
