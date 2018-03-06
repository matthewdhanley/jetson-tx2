# Using the Built In Camera

## What is a CSI Camera and Why Use it?
CSI cameras allow for high resolution video in conjunction with high frame rates. A CSI optimizes getting images from the camera to the CPU for computation. They are similar to the cameras on today's cell phones. They also give you low level control. A good writeup on the advantages of CSI cameras with the Jetson is available [here](http://petermoran.org/csi-cameras-on-tx2/).


## How to get video off the CSI Camera
`gstreamer` is the method that will be used here to get data from the onboard CSI camera of the TX2 development board. In short, `gstreamer` pipelines data from the camera from input to output. The `0-camera.cpp` file has a line with an example of a gstreamer pipeline:
```c++
    const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
```

The important thing that this pipeline is doing is converting the raw image data from the CSI camera to BGR colorspace that can then be used by OpenCV. This is important because it allows the hardware modules to convert the video rather than the CPU. Investigating this pipeline will make it apparent that it is possible to change the height and width of the input image from 1280 and 720 respectively. It can also be seen the the framerate can be changed. If you find that your video is upside down, change the flip method to flip-method=6.