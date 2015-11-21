#ifndef __OPENCV_UTIL_HPP
#define __OPENCV_UTIL_HPP

#include <opencv2/core/core.hpp>
#include <iostream>

#include "uf.hpp"

#define DEFAULT_NUM_CC 256

using namespace std;
using namespace cv;

typedef struct range {
	int a; // Start
	int b; // End
} range;

void diff(Mat& result, const Mat& frameA, const Mat& frameB, const range rangeVert, const range rangeHoriz, double threshold);
int** getConnectedComponents(const Mat& components);

#endif