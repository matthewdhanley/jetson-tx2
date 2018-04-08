# NVCC vs Makefile

## First, some base knowledge
A common misconception is that *compiling* creates an executable. This
is not true. *Compiling* creates an *object* file that is in machine
code, like Assembly. You can't run this yet!!!

It is common to create a C++ project that consists of many source files. When
these are compiled, multiple object files are created. It is the *Linking*
step that allows these files to be put together and executed.

**Compiling** - Taking source files (.c, .cpp, .cu, .cc, .cxx, etc) and
creating an 'object' file.

**Linking** - Creation of single executable file from multiple object files.
This is where errors start to arise. If a function is underdefined during
*compilation*, the compiler will likely not complain. If the function is
underdefined during *linking*, the linker will complain.

Why seperate these processes? It allows for larger projects. It takes a while
to compile large projects, but if just one file is changed, it allows for
just this one file to be compiled.

## make

The ```make``` tool is a dependency manager. It takes a bunch of source
files, object files, libraries, headers, etc and turns them into
into an executable.

Make is useful because it allows for modularity, easily. This makes programs,
especially large programs, easier to design, implement, and maintain.

### cmake
```cmake``` is a tool that is used to build ```Makefiles``` (build system generator)
* ```cmake``` is a scripting language
* ```CMakeLists.txt``` is the entry point for the build system generator
   * The top directory should hold a ```CMakeLists.txt``` file
   * Additional directories can be added by putting ```add_subdirectory()```
   into CMakeLists.
      * Note that subdirectories must also have ```CMakeLists.txt``` files.
* Commands in CMake
   * Syntax: ```command_name(space seperated list of strings)```
   * Can be used to set variables, change behavior of other commands, create build targets,
   modify build targets, etc.
* Variables
   * set with ```set()``` command
   * reference with ```${}```
   * Example that would print "hello world.":
   ```
   set(hello world)
   message(STATUS "hello, ${hello}")
   ```
   * ```cmake``` variables != environment variables (like in a Makfile)
* Comments
   * ```# single line comment```
* It is possible to set custom commands
   * Lookup ```function()``` and ```macro()```
* Targets and properties
   * Constructors
      * ```add_executable()``` - creates an executable from the specified file
      * ```add_library()``` - adds user library
      * ```target_link_libraries()``` - Links libraries to executable

[1-make example](./1-make)
``` CMake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# define the name of the project
project(1-make)

# find the required packages, here it's opencv and cuda
find_package(CUDA 8.0 REQUIRED)
find_package(OpenCV REQUIRED)

# specify where include directories are.
# OpenCV_INCLUDE_DIRS is a variable that is populated when we run
include_directories(${OpenCV_INCLUDE_DIRS})

# creating an executable from our c++ code and header
add_executable(1-make main.cpp main.h)

# creating CUDA "library" from our CUDA code.
CUDA_ADD_LIBRARY(
        1-make_gpu
        gpu_kernels.cu
        gpu_kernels.h
        )

# linking libraries to our executable
target_link_libraries(1-make 1-make_gpu)
target_link_libraries(1-make ${OpenCV_LIBS})
target_link_libraries(1-make ${CUDA_LIBRARIES})
```


## NVCC
If you wanted to compile the code in [0-nvcc](./0-nvcc), you could type
```make``` then execute the output executable. However you might wonder, what is happening
here?

The ```Makefile``` is wrapping the ```nvcc``` commands to simplify the
compilation and linking steps for the user. You *could* type out the
following lines every time if you really wanted to.
```
nvcc -I. -I/usr/local/cuda/include `pkg-config --cflags opencv` -c main.cu -o main.o
nvcc main.o -o tracker -L/usr/local/cuda/lib64 -lcudart -lcuda `pkg-config --libs opencv`
```
or even longer yet
```
nvcc main.cu -o tracker -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 `pkg-config opencv --cflags --libs` -lcudart -lcuda
```

Your next question is what is ``` `pkg-config opencv --cflags` ``` and
``` `pkg-config opencv --libs` ```? ```pkg-config``` is a command line
tool that allows for easier compilation and linking of external programs.
It provides linking and compilation flags for the compiler/linker. Try
it in your command line!
```
pkg-config opencv --cflags
```
outputs
```
-I/usr/include/opencv
```
Not bad. This could easily be added to the `nvcc` command. However, that
library is not the same for everyone. Things get more complicated with
the OpenCV library. Try this:
```
pkg-config opencv --libs
```
outputs for me:
```
pkg-config opencv --libs
-lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_dnn -lopencv_ml -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev
```
Wow! That's a lot of libraries. You can pick and choose which ones you need, but
your ```nvcc``` command now would look like the following:
```
nvcc main.cu -o tracker -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/usr/include/opencv -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_dnn -lopencv_ml -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev  -lcudart -lcuda
```
No thanks.

Now you're thinking, "What the heck is going on with all these flags anyways?" I'll
try to explain.

With most commands, you can use ```man``` in the command line to try to get
an idea of how to use it. However, ```nvcc``` does not have a ```man```
page. You can get similar information from ```man gcc```.

I will explain the the directives in the commands above. Note that there
are *hundreds* of flags that you can supply.

The first argument in a basic ```nvcc``` call is the source code file
name, i.e. ```main.cu```. Simple enough.

The second arguement seen above, ```-o``` is used to specify the name
of the output file. Here, ```tracker``` is the name of the outputted
executable.

```-I``` tells ```nvcc``` where to look for header files. So if you have
header files in the top of the project directory, ```-I.``` tells ```nvcc```
to look in the current directory.

```-L``` tells ```nvcc``` where to look for libraries that are specified.
```-L``` does not include the libraries, just tells ```nvcc``` where to look.
To include libraries, use ```-l``` as seen with ```-lcuda``` above. With that
directive, ```nvcc``` will first look for the ```cuda``` library in all the standard
system directories then directories specified with ```-L```, in this case
```/user/local/cuda/lib64```.

So to make things more clear, if I have the following include in my CUDA
program
``` C++
#include <opencv2/highgui/highgui.hpp>
```
I need to make sure that I have ```-I/usr/include/opencv``` and ```-lopencv_highghi``` included in my
nvcc call.

That's about it for the basics. Other commands include ```-c``` which just
*compiles* the code and doesn't *link* it. This can be used to compile
many different source files. Then you can link them together later with ```nvcc```. This
scheme can be seen in the ```Makefile``` to make it easy to implement future source
files. It *could* all be done in one line.


#### References
https://www.cprogramming.com/compilingandlinking.html
