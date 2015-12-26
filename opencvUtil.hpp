#ifndef __OPENCV_UTIL_HPP
#define __OPENCV_UTIL_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "uf.hpp"

#define DEFAULT_NUM_CC 256

using namespace std;
using namespace cv;

static vector<Point> borderSearchOrder;

typedef struct simplePoint {
	double x;
	double y;
} simplePoint;

void diff(Mat& result, const Mat& frameA, const Mat& frameB, const Rect_<int> opRange, double threshold);
void getConnectedComponents(Mat& componentLabels, const Mat& components, const Rect_<int> opRange);
void getContours(vector<vector<Point> >& contours, vector<Vec4i>& hierarchy, const Mat& image);
bool closeContours(const vector<Point>& contA, const vector<Point>& contB, double threshold);
vector<Point> findBorder(Mat image, Point start);
vector<Point> findBorder2(Mat image, Point start, int goodLabel);
void drawContour(Mat image, vector<Point> contour, Scalar color);
simplePoint simplifyPoint(Point2d p);

#endif