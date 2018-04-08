cmake_minimum_required(VERSION 2.8)
project(1-make-exe)

set(source_files ../gpu_kernels.cu ../main.cpp)
add_executable(1-make ${source_files})
