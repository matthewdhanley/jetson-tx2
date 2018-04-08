cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# define the name of the project
project(1-make)

# find the required packages, here it's opencv and cuda
find_package(CUDA 8.0 REQUIRED)
find_package(OpenCV REQUIRED)

# specify where include directories are.
# OpenCV_INCLUDE_DIRS is a variable that is populated when we run
include_directories(${OpenCV_INCLUDE_DIRS})

# creating an executable from our c++ code and header
add_executable(1-make main.cpp main.h)

# creating CUDA "library" from our CUDA code.
CUDA_ADD_LIBRARY(
        1-make_gpu
        gpu_kernels.cu
        gpu_kernels.h
        )

# linking libraries to our executable
target_link_libraries(1-make 1-make_gpu)
target_link_libraries(1-make ${OpenCV_LIBS})
target_link_libraries(1-make ${CUDA_LIBRARIES})

# prints out this message because this cost me hours of trouble shooting
message("!!!!!!!!!!!!!!!DISABLE ROS OPENCV!!!!!!!!!!!!")
