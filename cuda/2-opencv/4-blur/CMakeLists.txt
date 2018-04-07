cmake_minimum_required (VERSION 2.8)

project(blur)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(blur blur.cpp)

target_link_libraries(blur ${OpenCV_LIBS})
