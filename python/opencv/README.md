# Note:
I had to download and run [buildOpenCVTX2](www.jetsonhacks.com/2017/04/05/build-opencv-nvidia-jetson-tx2/) from JetsonHacks
```
git clone https://github.com/jetsonhacks/buildOpenCVTX2.git
cd buildOpenCVTX2
./buildOpenCVTX2.sh
cd ~/opencv/build
sudo make install
```

I then went back and ran this to enable some additional CUDA options.
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.1.0/modules \
    -D BUILD_EXAMPLES=ON ..
```
and ran `sudo make install` again for good measure.	
