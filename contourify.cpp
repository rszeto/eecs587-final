#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	// Load image
	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat paddedImage;
	copyMakeBorder(image, paddedImage, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	// Get contours of local image
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(paddedImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(-1,-1));

	Mat saveImage = Mat::zeros(image.size(), CV_8UC1);
	for(int c = 0; c < contours.size(); c++) {
		vector<Point> contour = contours[c];
		for(int i = 0; i < contour.size(); i++) {
			saveImage.at<uchar>(contour[i]) = 255;
		}
	}

	string newFileName(argv[1]);
	newFileName.replace(newFileName.find(".png"), 4, "_c.png");
	imwrite(newFileName, saveImage);
	cout << "Contour version of " << argv[1] << " saved at " << newFileName << endl;
}