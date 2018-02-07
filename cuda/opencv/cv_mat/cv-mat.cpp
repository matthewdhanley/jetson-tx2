#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int draw_circle(){
	cv::Mat image0; //constructor for image0. Contains no data yet.
	image0.create(480, 640, CV_8UC1); //Creating an image that is grayscale
					  //480x640 pixels
					  //8 unsigned ints with one channel
					  //linear memory, row major order
	
	image0.setTo(0);  //fills image with zeros (black)

	cv::Point center(image0.cols / 2, image0.rows / 2); //center point
	int radius = image0.rows / 2; //radius of circle (half the width)

	cv::circle(image0, //image to draw on
            center, //location of center
	    radius, //radius of circle
            128, //color (intensity since we are in grayscale)
            3); //thickness

	cv::imshow("image0 display",image0); //display the result
	cv::waitKey(); //wait for user input.
	return 0;
}

int draw_circle_square(){
	cv::Mat image0; //constructor for image0. Contains no data yet.
	image0.create(480, 640, CV_8UC1); //Creating an image that is grayscale
					  //480x640 pixels
					  //8 unsigned ints with one channel
					  //linear memory, row major order
	
	image0.setTo(0);  //fills image with zeros (black)

	cv::Point center(image0.cols / 2, image0.rows / 2); //center point
	int radius = image0.rows / 2; //radius of circle (half the width)

	cv::circle(image0, //image to draw on
            center, //location of center
	    radius, //radius of circle
            128, //color (intensity since we are in grayscale)
            3); //thickness

	cv::Mat image1 = image0;
	//note here that we are adding a rectangle to image1, but it appears on image0
	cv::rectangle(image1,
	    center - cv::Point(radius,radius), //bottom left
	    center + cv::Point(radius,radius), //top right
	    255, //white
            3); //thickness

	cv::imshow("image0 display",image0); //display the result
	cv::waitKey(); //wait for user input.
	return 0;
}

int gradient(){
	cv::Mat image0; //constructor for image0. Contains no data yet.
	image0.create(480, 640, CV_8UC1); //Creating an image that is grayscale
					  //480x640 pixels
					  //8 unsigned ints with one channel
					  //linear memory, row major order
	
	image0.setTo(0);  //fills image with zeros (black)

	cv::Point center(image0.cols / 2, image0.rows / 2); //center point
	int radius = image0.rows / 2; //radius of circle (half the width)

	cv::circle(image0, //image to draw on
            center, //location of center
	    radius, //radius of circle
            128, //color (intensity since we are in grayscale)
            3); //thickness

	cv::Mat image1 = image0;
	//note here that we are adding a rectangle to image1, but it appears on image0
	cv::rectangle(image1,
	    center - cv::Point(radius,radius), //bottom left
	    center + cv::Point(radius,radius), //top right
	    255, //white
            3); //thickness

	cv::Mat image2;
	cv::cvtColor(image1, image2, CV_GRAY2BGR);

	int inscribed_radius = radius / sqrt(2);

	cv::Rect rect(
	         center - cv::Point(inscribed_radius, inscribed_radius),
	         center - cv::Point(inscribed_radius, inscribed_radius));
	
	cv::Mat roi = image2(rect); //using our rectangle to index into image 2
        roi.setTo(cv::Scalar(0, 185, 250)); //set eqach channel of the Mat object
	
	for (int y = 0; y < image1.rows; y++){
		uchar *row = image1.ptr<uchar>(y); //need to specify type with template
		for (int x = 0; x<image1.cols; x++){
			if (row[x] == 128)
				row[x] = x * y * 255 / image1.total(); //gradient
		}
	}
	cv::imshow("image0 display",image0); //display the result
	cv::waitKey(); //wait for user input.
	return 0;
}

int color_gradient(){
	cv::Mat image0; //constructor for image0. Contains no data yet.
	image0.create(480, 640, CV_8UC1); //Creating an image that is grayscale
					  //480x640 pixels
					  //8 unsigned ints with one channel
					  //linear memory, row major order
	
	image0.setTo(0);  //fills image with zeros (black)

	cv::Point center(image0.cols / 2, image0.rows / 2); //center point
	int radius = image0.rows / 2; //radius of circle (half the width)

	cv::circle(image0, //image to draw on
            center, //location of center
	    radius, //radius of circle
            128, //color (intensity since we are in grayscale)
            3); //thickness

	cv::Mat image1 = image0;
	//note here that we are adding a rectangle to image1, but it appears on image0
	cv::rectangle(image1,
	    center - cv::Point(radius,radius), //bottom left
	    center + cv::Point(radius,radius), //top right
	    255, //white
            3); //thickness

	cv::Mat image2;
	cv::cvtColor(image1, image2, CV_GRAY2BGR);

	int inscribed_radius = radius / sqrt(2);

	cv::Rect rect(
	         center - cv::Point(inscribed_radius, inscribed_radius),
	         center + cv::Point(inscribed_radius, inscribed_radius));
	
	cv::Mat roi = image2(rect); //using our rectangle to index into image 2
        roi.setTo(cv::Scalar(0, 185, 118)); //set eqach channel of the Mat object
	
	for (int y = 0; y < image2.rows; y++){
		cv::Vec3b *row = image2.ptr<cv::Vec3b>(y); //need to specify type with template
		for (int x = 0; x<image2.cols; x++){
			if (row[x][1] == 185)
				row[x] = cv::Vec3b(0, x * y * 255 / image1.total(), 118); //gradient
		}
	}
	cv::imshow("image2 display",image2); //display the result
	cv::waitKey(); //wait for user input.
	return 0;
}

int main(){
	draw_circle(); //call draw circle function
	draw_circle_square(); //draw circle square
	gradient(); //make gradient
	color_gradient();
}
