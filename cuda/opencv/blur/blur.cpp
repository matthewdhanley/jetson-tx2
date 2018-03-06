using namespace std;
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <time.h>

int blur_cpu(){
	cv::Mat img = cv::imread("../img/fruit.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	printf("rows: %d\n",img.rows);
	printf("cols: %d\n",img.cols);
	
	cv::Mat dst;

	int N_BLUR = 1<<15;
	printf("Blurring image %d times on CPU\n",N_BLUR);
        clock_t begin = clock();	
        for (int i=0; i < N_BLUR; i++){
		cv::blur(img,dst, cv::Size(9,9));
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("CPU TIME: %f\n", time_spent);

}


int blur_gpu(){
	cv::Mat img = cv::imread("../img/fruit.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat dst_host;
	int N_BLUR = 1<<15;
	printf("Blurring image %d times on GPU\n",N_BLUR);
        cv::gpu::GpuMat dst,src;
	src.upload(img);	
        clock_t begin = clock();		
        for (int i=0; i < N_BLUR; i++){
		cv::gpu::blur(src, dst, cv::Size(9,9));
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("GPU TIME: %f\n", time_spent);
}


int main(){
	blur_cpu();
	blur_gpu();
}
