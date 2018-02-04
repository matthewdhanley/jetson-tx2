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

