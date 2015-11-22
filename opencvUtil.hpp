#ifndef __OPENCV_UTIL_HPP
#define __OPENCV_UTIL_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "uf.hpp"

#define DEFAULT_NUM_CC 256

using namespace std;
using namespace cv;

typedef struct range {
	int a; // Start
	int b; // End
} range;

void diff(Mat& result, const Mat& frameA, const Mat& frameB, const Rect_<int> opRange, double threshold);
void getConnectedComponents(Mat& componentLabels, const Mat& components, const Rect_<int> opRange);
void getContours(vector<vector<Point> >& contours, vector<Vec4i>& hierarchy, const Mat& image);
bool closeContours(const vector<Point>& contA, const vector<Point>& contB, double threshold);

#endif