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
     * [4-camera](./cuda/4-camera)
     * [5-cpp-cuda-integration](./cuda/5-cpp-cuda-integration)
     * [6-reduction](./cuda/6-reduction)
