cmake_minimum_required (VERSION 2.8)

project(cam)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cam cam.cpp)

target_link_libraries(cam ${OpenCV_LIBS})
