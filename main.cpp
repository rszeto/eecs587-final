#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencvUtil.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	int threshold = 75;
	char* imLoc = "/home/szetor/shared/legendary/image_0001.png";
	char* imLoc2 = "/home/szetor/shared/legendary/image_0002.png";
	Mat image, image2;
	image = imread(imLoc, CV_LOAD_IMAGE_COLOR);
	image2 = imread(imLoc2, CV_LOAD_IMAGE_COLOR);
	namedWindow("display", CV_WINDOW_NORMAL);
	Mat image3 = Mat::zeros(image.rows, image.cols, CV_8UC1);

	range xRange = {0, 359};
	range yRange = {0, 359};

	diff(image3, image, image2, yRange, xRange, threshold);

	imshow("display", image3);
	waitKey(0);

	Mat components = Mat::zeros(image.rows, image.cols, CV_8UC1);
	int** comps = getConnectedComponents(image3);
	for(int i = 0; i < image.rows; i++) {
		for(int j = 0; j < image.cols; j++) {
			components.at<uchar>(i, j) = comps[i][j];
		}
	}

	imshow("display", components);
	waitKey(0);
	
	return 0;
}