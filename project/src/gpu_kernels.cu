// -------------------------------------- GPU KERNELS ------------------------------------------------------------------
#include <device_launch_parameters.h>
#include "../include/main.h"

/***********************************************************************************************************************
 * Name:    gpu_colorMask
 * Author:  Matthew Hanley
 * Purpose: Mask an image to isolate a blob of color, quickly.
 *
 * Inputs:
 * @param Pout      - binary output mask
 * @param Pin       - HSV input image, 3 channels (Hue, Saturation, Value)
 * @param width     - width of the image in pixels
 * @param height    - height of the image in pixels
 * @param colorMask - struct containing bounding values
 *
 **********************************************************************************************************************/
__global__ void gpu_colorMask(bool *mask, unsigned char* Pin_hsv, int width, int height, colorMaskHSV colorMask) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;                /* index into row of image */
    int row = threadIdx.y + blockIdx.y * blockDim.y;                /* index into column of image */
    int j = row * width + col;
    int i = j * CHANNELS;            /* unique index into image */
    if (col < width-1 && row < height-1 && col > 0 && row > 0)                                /* Make sure within image, this will cause some control*/
        /* divergence at the edges of the image*/
    {

        unsigned char h = Pin_hsv[i];                               /* Grab values of hue, saturation, and value */
        unsigned char s = Pin_hsv[i + 1];
        unsigned char v = Pin_hsv[i + 2];

        if (h < colorMask.hue_max+1 && (int)h > colorMask.hue_min-1 &&     /* check if pixel should be masked. */
            s < colorMask.sat_max+1 && (int)s > colorMask.sat_min-1 &&
            v < colorMask.val_max+1 && (int)v > colorMask.val_min-1)
        {
            mask[j] = 1;
        } else                                                      /* assign black values! */
        {
            mask[j] = 0;
        }
    }
    else if (col == width - 1 || row == height - 1 || row == 0 || col == 0){
        mask[j] = 0; // mask the edges of the image for now. throws off the moment.
    }
}

__global__ void gpu_erodeMask(bool *mask, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;                /* index into row of image */
    int row = threadIdx.y + blockIdx.y * blockDim.y;                /* index into column of image */
    int maxj= width * height;
    int j = row * width + col;
    if (col < width && row < height){
        // erode
        if (mask[j] && j+width < maxj && j-width >= 0) { // if mask[j] is equal to one
            if (!mask[j + 1] || !mask[j - 1] || !mask[j + width] || !mask[j - width]) { // if one of the surrounding pixels is zero
                mask[j] = 0; // make j equal to zero aka erode.
            }
        }
    }
}

// stride should be the next highest power of two from block_size*block_size
__global__ void gpu_centroid(bool* mask, int width, int height, int blockStride, int gridStride, centroid* c,
                                unsigned int* countNonZeroBlocks){
    __shared__ unsigned int momentX[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ unsigned int momentY[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ unsigned int countNonZero[BLOCK_SIZE * BLOCK_SIZE];
    unsigned int t = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int tmax = blockDim.x * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;                /* index into row of image */
    int row = threadIdx.y + blockIdx.y * blockDim.y;                /* index into column of image */
    int i = row * width + col;                                      /* index into mask */

    // load mask into shared memory
    if (col < width && row < height){                               /* make sure within bounds of image */
        countNonZero[t] = (unsigned int) mask[i];                   /* load mass */
        momentX[t] = countNonZero[t] * col;                         /* create pixel moments */
        momentY[t] = countNonZero[t] * row;
    }


    // perform reduction algorithm to accum moments and mass
    for( ; blockStride >= 1; blockStride = blockStride >> 1){
        __syncthreads();
        if(t + blockStride < tmax && row < height && col < width) {
            momentX[t] += momentX[t + blockStride];
            momentY[t] += momentY[t + blockStride];
            countNonZero[t] += countNonZero[t + blockStride];
        }
    }
    __syncthreads();

    unsigned int b = blockIdx.y * gridDim.x + blockIdx.x;           /* index blocks into grid */


    if (t == 0 && b < gridDim.x * gridDim.y) {
            if (countNonZero[0] > 0) {
                c[b].x = float(momentX[0]) / float(countNonZero[0]);
                c[b].y = float(momentY[0]) / float(countNonZero[0]);
                countNonZeroBlocks[b] = 1;
            }
            else{
                c[b].x = 0.0;
                c[b].y = 0.0;
                countNonZeroBlocks[b] = 0;
            }
        }

    __syncthreads();

    // now c is composed of centroids for each block. Time to do a similar reducing algorithm on c. It should be local
    // in memory since it was the last data worked with


    for ( ; gridStride >= 1; gridStride = gridStride >> 1){
        __syncthreads();

        if (t == 0 && b < gridStride && (b + gridStride) < gridDim.x * gridDim.y){
            c[b].x += c[b + gridStride].x;
            c[b].y += c[b + gridStride].y;
            countNonZeroBlocks[b] += countNonZeroBlocks[b + gridStride];
        }
    }

    __syncthreads();

    // average it out. only want top left thread to perform.
    if (row == 0 && col == 0){
        c[0].x = c[0].x/float(countNonZeroBlocks[0]);
        c[0].y = c[0].y/float(countNonZeroBlocks[0]);
    }

}

__global__ void gpu_applyMask(unsigned char* Pout, unsigned char* Pin_rgb, bool* mask, int width, int height){
    int col = threadIdx.x + blockIdx.x * blockDim.x;                /* index into row of image */
    int row = threadIdx.y + blockIdx.y * blockDim.y;                /* index into column of image */
    int j = row * width + col;
    int i = j * CHANNELS;
    if (col < width && row < height){
        if (mask[j]){
            Pout[i] = Pin_rgb[i];
            Pout[i+1] = Pin_rgb[i+1];
            Pout[i+2] = Pin_rgb[i+2];
        }
        else{
            Pout[i] = 0;
            Pout[i+1] = 0;
            Pout[i+2] = 0;
        }
    }
}


extern "C" void cuda_mask(bool *mask, unsigned char* Pin_hsv,int width, int height, colorMaskHSV colorMask,
                          dim3 numBlocks, dim3 numThreads)
{
    gpu_colorMask <<< numBlocks, numThreads >>> (mask, Pin_hsv, width, height, colorMask);
}

extern "C" void cuda_erode(bool *mask, int width, int height, dim3 numBlocks, dim3 numThreads)
{
    gpu_erodeMask <<< numBlocks, numThreads >>> (mask, width, height);
}

extern "C" void cuda_applyMask(unsigned char* Pout, unsigned char* Pin_rgb, bool*mask, int width, int height,
                                dim3 numBlocks, dim3 numThreads)
{
    gpu_applyMask <<< numBlocks, numThreads >>> (Pout, Pin_rgb, mask, width, height);
}

extern "C" void cuda_gpu_centroid(bool* mask, int width, int height, int blockStride, int gridStride, centroid* c,
                                             unsigned int* countNonZeroBlocks, dim3 numBlocks, dim3 numThreads)
{
    gpu_centroid <<< numBlocks, numThreads >>> (mask, width, height, blockStride, gridStride, c, countNonZeroBlocks);
}