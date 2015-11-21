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

	Rect_<int> opRange(0, 0, image.cols, image.rows);
	Mat diffs = Mat::zeros(image.rows, image.cols, CV_8UC1);
	diff(diffs, image, image2, opRange, threshold);
	namedWindow("display", CV_WINDOW_NORMAL);
	imshow("display", diffs);
	waitKey(0);

	Mat components = Mat::zeros(diffs.size(), CV_32SC1);
	getConnectedComponents(components, diffs);
	for(int i = 0; i < image.rows; i++) {
		for(int j = 0; j < image.cols; j++) {
			cout << (int)components.at<int>(i, j) << ",";
		}
		cout << endl;
	}
	
	return 0;
}