#include <iostream>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencvUtil.hpp"

using namespace std;
using namespace cv;

RNG rng(12345);

void animateDiffs() {
	int threshold = 75;
	Mat prevImage;
	Mat curImage;
	char imLoc[256];
	namedWindow("curImage", CV_WINDOW_NORMAL);
	namedWindow("diffImage", CV_WINDOW_NORMAL);
	for(int i = 1; i <= 500; i++) {
		if(curImage.data) {
			prevImage = curImage.clone();
		}
		sprintf(imLoc, "/home/szetor/shared/data/MOT/ADL-Rundle-3/image_%04d.png", i);
		curImage = imread(imLoc, CV_LOAD_IMAGE_COLOR);
		if(prevImage.data) {
			Mat diffImage = Mat::zeros(curImage.size(), CV_8UC1);
			Rect_<int> opRange(0, 0, curImage.cols, curImage.rows);
			diff(diffImage, curImage, prevImage, opRange, threshold);
			imshow("diffImage", diffImage);
		}
		imshow("curImage", curImage);
		waitKey(25);
	}
}

void findBlobs() {
	int threshold = 75;
	char* imLoc = "/home/szetor/shared/legendary/image_0001.png";
	char* imLoc2 = "/home/szetor/shared/legendary/image_0002.png";
	Mat image, image2;
	image = imread(imLoc, CV_LOAD_IMAGE_COLOR);
	image2 = imread(imLoc2, CV_LOAD_IMAGE_COLOR);

	Rect_<int> opRange(0, 0, image.cols, image.rows);
	Mat diffs = Mat::zeros(image.rows, image.cols, CV_8UC1);
	diff(diffs, image, image2, opRange, threshold);
	imwrite("output/diffs.png", diffs);

	Mat components = Mat::zeros(diffs.size(), CV_32SC1);
	getConnectedComponents(components, diffs, opRange);
	for(int i = 0; i < image.rows; i++) {
		for(int j = 0; j < image.cols; j++) {
			cout << (int)components.at<int>(i, j) << ",";
		}
		cout << endl;
	}
}

void findNearbyContours() {
	int threshold = 75;
	char* imLoc = "/home/szetor/shared/legendary/image_0001.png";
	char* imLoc2 = "/home/szetor/shared/legendary/image_0002.png";
	Mat image, image2;
	image = imread(imLoc, CV_LOAD_IMAGE_COLOR);
	image2 = imread(imLoc2, CV_LOAD_IMAGE_COLOR);

	Rect_<int> opRange(0, 0, image.cols, image.rows);
	Mat diffs = Mat::zeros(image.rows, image.cols, CV_8UC1);
	diff(diffs, image, image2, opRange, threshold);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	getContours(contours, hierarchy, diffs);
	
	char contourSaveLoc[256];
	for(int i = 0; i < contours.size(); i++)
	{
		Mat cloned;
		cvtColor(diffs, cloned, CV_GRAY2RGB);
		Scalar green = Scalar(0, 255, 0);
		for(int j = 0; j < contours.size(); j++) {
			if(closeContours(contours[i], contours[j], 10)) {
				Scalar randColor = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
				drawContours(cloned, contours, j, randColor);
			}
		}
		drawContours(cloned, contours, i, green);
		sprintf(contourSaveLoc, "output/contours/contours%04d.png", i);
		imwrite(contourSaveLoc, cloned);
	}
	imwrite("output/diffs.png", diffs);
}

int main(int argc, char** argv) {
	// animateDiffs();
	// findBlobs();
	findNearbyContours();

	return 0;
}