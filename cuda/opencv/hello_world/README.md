# Simple "Hello World" example with OpenCV on Jetson TX2

If you have done the JetPack installation, you should have all you need to run OpenCV on your TX2. This project shows a very, very simple implementation of OpenCV using C++. This program does not use any CUDA kernals, but is a good starting place for using CV2 on the Jetson with C++.

## The Code
```c++
#include <opencv2/highgui/highgui.hpp>

int main(){
    cv::Mat img(512, 512, CV_8UC3, cv::Scalar(0));

    cv::putText(img,"Hello World", cv::Point(10, img.rows/2),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118,185,0),2);

    cv::imshow("hello world", img);

    cv::waitKey();

    return 0;
}
```

## The Code Explained
This line of code is telling the program that we need to use the [highgui.hpp File Reference](https://docs.opencv.org/3.1.0/d4/dd5/highgui_8hpp.html).
```c++
#include <opencv2/highgui/highgui.hpp>
```
Next, in our main function, we want to create an image. This is done using cv::Mat. Mat is a general matrix class provied by OpenCV. It works fantastically as an image container. Here, we are creating a three dimensional array with dimensions 512 x 512 x 3 where each entry is represented by an 8 bit unsigned integer (specified by CV_8UC3).
```c++
cv::Mat img(512, 512, CV_8UC3, cv::Scalar(0));
```
In the spirit of the "Hello World" program, let's add "Hello World" to the image.
```c++
cv::putText(img,"Hello World", cv::Point(10, img.rows/2),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118,185,0),2);
```
And now we want to show the image in a named window called "hello world"
```c++
cv::imshow("hello world", img);
```
Finally, we wait for user input on the keyboard
```c++
cv::waitKey();
```

## Building the project
The best way to build this project is to use cmake. If you haven't already, download and install cmake
```
sudo apt-get install cmake
```
Now, we need to make a CMakeLists.txt file. Create the file by typing `touch CMakeLists.txt` in the directory where your program lives. Open the file with your favorite text editor. Let's dig into what is needed in the CMakeLists.txt file.

```
cmake_minimum_required (VERSION 2.8)

project(hello_world)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_hello hello_world.cpp)

target_link_libraries(cv_hello ${OpenCV_LIBS})
```
First, we want to tell cmake that the minimum version of cmake to use is v2.8.
```
cmake_minimum_required (VERSION 2.8)
```
Next, we need to tell cmake what the project name is. In this case, it's `hello_world`
```
project(hello_world)
```
Now we need to tell cmake that we are using OpenCV. This creates the variables `OpenCV_INCUDE_DIRS` and `OpenCV_LIBS`.
```
find_package(OpenCV REQUIRED)
```
Now tell cmake that these directories need to be included
```
include_directories(${OpenCV_INCLUDE_DIRS})
```
And then add the executable file (binary) from the .cpp File
```
add_executable(cv_hello hello_world.cpp)
```
Finally, link the OpenCV libraries to the program
```
target_link_libraries(cv_hello ${OpenCV_LIBS})
```
Now your CMakeLists.txt file is ready to go!

## Compiling and running
1. Make a "build" directory and change into it
```
mkdir build; cd build
```
2. Use `cmake` to compile your project
```
cmake ..
```
3. Make your project
```
make
```
4. Run the executable
```
./hello_world
```
You're done!
