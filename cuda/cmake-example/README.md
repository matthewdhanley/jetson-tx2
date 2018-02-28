# Compiling with CMake
1. Write your CUDA program. In this example, I am just going to use a simple vector adding example.
2. Create a `CMakeLists.txt` file.
```
$ touch CMakeLists.txt
```
3. Open CMakeLists.txt in your favorite editer.
4. Do something like this in your `CMakeLists.txt` file:
```
# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
 
# Specify binary name and source file to build it from
cuda_add_executable(
    adding
    adding.cu)
```
5. Build it with `cmake`
```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
$ ./hellocuda
```

You can also use `make` instead of `cmake --build .`
6. Run it!
```
$ ./adding.cu
```
