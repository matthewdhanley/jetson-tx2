cmake_minimum_required (VERSION 2.8)

project(1-blur)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gpu_blur gpu_blur.cpp)

target_link_libraries(gpu_blur ${OpenCV_LIBS})
