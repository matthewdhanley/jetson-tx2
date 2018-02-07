cmake_minimum_required (VERSION 2.8)

project(imread)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imread imread.cpp)

target_link_libraries(imread ${OpenCV_LIBS})
