//
// Created by matt on 4/8/18.
//
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#ifndef MAIN_H
#define MAIN_H
    // -------------------------------------- STRUCTS ----------------------------------------------------------------------
    struct colorMaskHSV {
        int hue_min;
        int hue_max;
        int sat_min;
        int sat_max;
        int val_min;
        int val_max;
    };

    struct centroid {
        float x;
        float y;
    };

    #define BLOCK_SIZE 32
    // number of channels in image
    #define CHANNELS 3
    // image props
    #define WIDTH 1280
    #define HEIGHT 720


    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
#endif //MAIN_H
