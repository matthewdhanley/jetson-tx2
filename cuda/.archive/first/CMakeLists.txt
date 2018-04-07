# this cmake file was created following the cmake tutorial from
# cmake.org/cmake-tutorial

# minimum cmake version
cmake_minimum_required (VERSION 2.8)

# project name
project(camera)

# project version number
set (CAMERA_VERSION_MAJOR 0)
set (CAMERA_VERSION_MINOR 1)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
        "${PROJECT_SOURCE_DIR}/CameraConfig.h.in"
        "${PROJECT_BINARY_DIR}/CameraConfig.h"
)

# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")

# CUDA PACKAGE
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER clang++)

# COMPILE CU FILES
file(GLOB CUDA_FILES "src/" *.cu)
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30; -std=c++11")
CUDA_COMPILE(CU_O ${CUDA_FILES})

# SETUP FOR CPP FILES
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# look into subdirectory for custom library
# it needs to have its own CMakeLists.txt file
# its file will have this one line:
# add_library(Subdirectory custom.cpp)
# now we add it here
#include_directories("${PROJECT_SOURCE_DIR}/Subdirectory")
#add_subdirectory(Subdirectory)

# adds Subdirectory to list of included libs (defined by me)
#set(INCLUDE_LIBS ${INCUDE_LIBS} Subdirectory)


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
set(INCLUDE_LIBS ${INCLUDE_LIBS} ${OpenCV_LIBS})


cuda_add_executable(camera ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CU_O})

# link the libraries in INCUDE_LIBS variable
target_link_libraries(camera ${INCLUDE_LIBS})

