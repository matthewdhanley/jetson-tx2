cmake_minimum_required (VERSION 2.8)

project(0-camera)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(0-camera 0-camera.cpp)

target_link_libraries(0-camera ${OpenCV_LIBS})

