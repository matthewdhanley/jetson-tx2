// -------------------------------------- GPU KERNELS ------------------------------------------------------------------
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
__global__ void gpu_colorMask(unsigned char *Pout, unsigned char* Pin_hsv, unsigned char* Pin_rgb,
                              int width, int height, colorMaskHSV colorMask, centroid* c, unsigned int* countNonZeroBlocks) {

    __shared__ unsigned int momentsX[BLOCK_SIZE * BLOCK_SIZE];      /* used for storing moments, making array of zeros */
    __shared__ unsigned int momentsY[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ unsigned int countNonZero[BLOCK_SIZE * BLOCK_SIZE];

    int col = threadIdx.x + blockIdx.x * blockDim.x;                /* index into row of image */
    int row = threadIdx.y + blockIdx.y * blockDim.y;                /* index into column of image */
    unsigned int t = threadIdx.y * blockDim.x + threadIdx.x;        /* index into the block */
    unsigned int b = blockIdx.y * gridDim.x + blockIdx.x;           /* index blocks into grid */


    if (col < width && row < height)                                /* Make sure within image, this will cause some control*/
        /* divergence at the edges of the image*/
    {
        unsigned int i = (row * width + col) * CHANNELS;            /* unique index into image */
        unsigned char h = Pin_hsv[i];                               /* Grab values of hue, saturation, and value */
        unsigned char s = Pin_hsv[i + 1];
        unsigned char v = Pin_hsv[i + 2];

        if (h <= colorMask.hue_max && h >= colorMask.hue_min &&     /* check if pixel should be masked. */
            s <= colorMask.sat_max && s >= colorMask.sat_min &&
            v <= colorMask.val_max && v >= colorMask.val_min) {
            Pout[i] = Pin_rgb[i];                                   /* masking rgb image */
            Pout[i + 1] = Pin_rgb[i + 1];
            Pout[i + 2] = Pin_rgb[i + 2];
            momentsX[t] = col;                                      /* assigning weights to moment */
            momentsY[t] = row;
            countNonZero[t] = 1;                                    /* saving that not zero for average sum later */
        } else                                                      /* assign black values! */
        {
            Pout[i] = 0;
            Pout[i + 1] = 0;
            Pout[i + 2] = 0;
            momentsX[t] = 0;
            momentsY[t] = 0;
            countNonZero[t] = 0;                                    /* saving that zero for average sum later */
        }
    }
    __syncthreads();

    // now want to do some reduction to find center of mass of each block
    // the initial stride needs to be a power of 2
    for (unsigned int stride = BLOCK_SIZE * BLOCK_SIZE / 2; stride >= 1; stride = stride >> 1) {
        __syncthreads();

        // todo - make sure this isn't janky...

        if (row * width + col + stride < width * height) {          /* make pixels within image */
            if (t < stride) {                                       /* reducing size */
                momentsX[t] += momentsX[t + stride];
                momentsY[t] += momentsY[t + stride];
                countNonZero[t] += countNonZero[t + stride];        /* counting non zero */
            }
        }
    }
    __syncthreads();
    if (t == 0 && b < gridDim.x * gridDim.y) {
        if (countNonZero[0] > 0) {
            c[b].x = float(momentsX[0]) / float(countNonZero[0]);
            c[b].y = float(momentsY[0]) / float(countNonZero[0]);
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
    unsigned int stride = gridDim.x * gridDim.y;

    // bitwise method for finding next highest power of 2
    // see https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    stride--;
    stride |= stride >> 1;
    stride |= stride >> 2;
    stride |= stride >> 4;
    stride |= stride >> 8;
    stride |= stride >> 16;
    stride++;

    stride /= 2;

    for ( ; stride >= 1; stride = stride >> 1){
        __syncthreads();

        if (t == 0 && b < stride && (b + stride) < gridDim.x * gridDim.y){
            c[b].x += c[b + stride].x;
            c[b].y += c[b + stride].y;
            countNonZeroBlocks[b] += countNonZeroBlocks[b + stride];
        }
    }
    __syncthreads();

    // average it out.
    if (row == 0 && col == 0){
        c[0].x = c[0].x/float(countNonZeroBlocks[0]);
        c[0].y = c[0].y/float(countNonZeroBlocks[0]);
    }

}

extern "C" void cuda_mask(unsigned char *Pout, unsigned char* Pin_hsv, unsigned char* Pin_rgb,
                          int width, int height, colorMaskHSV colorMask, centroid* c, unsigned int* countNonZeroBlocks,
                          dim3 numBlocks, dim3 numThreads){
    gpu_colorMask <<< numBlocks, numThreads >>> (Pout, Pin_hsv, Pin_rgb, width, height, colorMask, c, countNonZeroBlocks);
}