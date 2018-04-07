cmake_minimum_required (VERSION 2.8)

project(cv-mat)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv-mat cv-mat.cpp)

target_link_libraries(cv-mat ${OpenCV_LIBS})
