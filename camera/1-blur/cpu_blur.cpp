#include "opencv2/highgui/highgui.hpp"
// highgui - an interface to video and image capturing.
#include <opencv2/imgproc/imgproc.hpp> // For dealing with images
#include <iostream>
//#include "camera.h" // including my helper file

//using namespace cv;
// Namespace where all the C++ OpenCV functionality resides.

int main()
{
    const char* gst;
    gst = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! \
                nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
                videoconvert ! video/x-raw, format=(string)BGR ! \
                appsink";

    cv::VideoCapture cap(gst);
    // cap is the object of class video capture that tries to capture Bumpy.mp4
    if ( !cap.isOpened() ) // isOpened() returns true if capturing has been initialized.
    {
        printf("Cannot open the stream\n");
        return -1;
    }

    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video

    cv::namedWindow("Blur",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

    while(1)
    {
        cv::Mat frame;
        // Mat object is a basic image container. frame is an object of Mat.
        if (!cap.read(frame)) // if not success, break loop
            // read() decodes and captures the next frame.
        {
            printf("Error getting image from camera\n");
            break;
        }
        cv::blur(frame,frame,cv::Size(10,10)); // To blur the image.
        cv::imshow("Blur", frame);
        // first argument: name of the window.
        // second argument: image to be shown(Mat object).

        if(cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
    }
    return 0;
}
// END OF PROGRAM