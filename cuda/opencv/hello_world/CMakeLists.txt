cmake_minimum_required (VERSION 2.8)

project(hello_world)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_hello hello_world.cpp)

target_link_libraries(cv_hello ${OpenCV_LIBS})
