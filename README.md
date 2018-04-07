# Jetson-TX2 (and CUDA) Independent Study

## Cloning the repository
Independent study at the University of Colorado Boulder about the NVIDIA Jetson TX2
To install this project onto your linux machine, use git
Simply type into the command line:
```
git clone https://github.com/matthewdhanley/jetson-tx2.git
```

Once it's there, you can run the following command to update it within the jetson-tx2 directory:
```
git pull origin master
```

or if you simply want to overwrite all files
```
git fetch --all
git reset --hard origin/master
```

I have included a shell script so you can simply run:
```
./git.sh
```
to update all your files (will overwrite any changes).

That's it!

## Directory Listing
* [Jetson-TX2](https://github.com/matthewdhanley/jetson-tx2)<br>
  * [getting_started](./getting_started)
     * Instructions to getting started with the Jetson-TX2 single board computer
  * [cuda](./cuda)
     * Example CUDA Programs
     * [0-adding](./cuda/0-adding)
        * Examples of Adding
        * [add_int.cu](./cuda/0-adding/add_int.cu)
           * The most basic example of adding on the GPU. Simply adds
        together two integers
        * [add_vec.cu](./cuda/0-adding/add_vec.cu)
           * Hello World Example - Simple example of adding two vectors
           using the GPU
        * [adding.cu](./cuda/0-adding/adding.cu)
           * Comparison of using GPU and CPU for adding. Logs results
           to a file for future analysis. Inside this file, you can change
           both the number of elements in the array as well as the number
           of iterations run
     * [1-multiplication](./cuda/1-multiplication)
        * [mat_mult.cu](./cuda/1-multiplication/mat_mult.cu)
           * Simple implementation of multiplication on GPU. User inputs
           matrix dimensions.
        * [mat_mult_tile.cu](./cuda/1-multiplication/book_example_naive.cu)
           * Example of matrix multipliation using tiling methods. User inputs
           matrix dimensions
     * [2-opencv](./cuda/2-opencv)
        * Examples of OpenCV implementation in C++
        * [0-hello_world](./cuda/2-opencv/0-hello_world)
           * Very basic example of OpenCV in C++. It is a good test to see if
           OpenCV is working on your device
        * [1-cv_mat](./cuda/2-opencv/1-cv_mat)
           * Example of how to use the Mat object in OpenCV
        * [2-imread](./cuda/2-opencv/2-imread)
           * Example of using imread to get images from files
        * [3-onboard_camera](./cuda/2-opencv/3-onboard_camera)
           * Example of getting images from the Jetson-TX2 onboard camera
        * [4-blur](./cuda/2-opencv/4-blur)
           * Example of using OpenCV's gpu functions. Doesn't currently
           compile on my machine
     * [3-cmake_examle](./cuda/3-cmake_example)
        * Example of using cmake to compile code
        * [adding.cu](./cuda/3-cmake_example/adding.cu)
           * Simple CUDA adding program
        * [CMakeLists.txt](./cuda/3-cmake_example/CMakeLists.txt)
           * Example ```CMakeLists.txt file``` for compiling CUDA code
     * [4-camera](./cuda/4-camera)
        * [0-camera](./cuda/4-camera/0-camera)
           * Example of using a camera with Jetson TX2
        * [1-fast-image-processing](./cuda/4-camera/1-fast-image-processing)
           * Kernels for greyscale, blurring, and thresholding. Allows for
           switching between CPU and GPU
        * [2-convolution](./cuda/4-camera/2-convolution)
           * Example of convolution of a mask on an image. Can switch between
           CPU and GPU
        * [3-compare](./cuda/4-camera/3-compare)
           * Compares different blur methods on GPU and CPU
     * [5-cpp-cuda-integration](./cuda/5-cpp-cuda-integration)
        * [0-make-example](./cuda/5-cpp-cuda-integration/0-make-example)
           * Very simple example of using CUDA and ```make```
        * [1-make-opencv-example](./cuda/5-cpp-cuda-integration/0-make-opencv-example)
           * Example of including OpenCV with ```make```
        * [2-particle-nvcc-example](./cuda/5-cpp-cuda-integration/2-particle-nvcc-example)
           * This is not my code, but shows an example of linking files together using ```make```
     * [6-reduction](./cuda/6-reduction)
        * [reduction.cu](./cuda/6-reduction/reduction.cu)
           * Example of using reduction to add a vector on the GPU
        * [mat_mult_reduction.cu](./cuda/6-reduction/mat_mult_reduction.cu)
           * Example of using reduction and matrix multiplication
  * [latex](./latex)
     * Examples of using LaTeX on linux
     * [0-sample-report](./latex/0-sample-report)
     * [1-cuda-cheatsheet](./latex/1-cuda-cheatsheet)
  * [project](./project)
     * Start of the project
     * [main.cu](./project/main.cu)
        * Program that tracks colors with ability to manually adjust mask
        with OpenCV slider bars
  * [python](./python)
     * [opencv](./python/opencv)
        * Example of using OpenCV in Python on Jetson TX2
     * [usb-cam](./python/usb-cam)
        * Test for using USB cam with Python and OpenCV
  * [git.sh](./git.sh)
     * Shell script to simplify pulling process and avoid merging.
  * [README.md](./README.md)
     * Markdown code to this "wiki"

