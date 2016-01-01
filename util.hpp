#ifndef __UTIL_HPP
#define __UTIL_HPP

#include <iostream>
#include <getopt.h>
#include <opencv2/core/core.hpp>

cv::Point linToTup(int k, int n);
int tupToLin(int i, int j, int n);
int triMatSize(int rows, int cols);

int handleOpts(int argc, char** argv, bool& displayImages, bool& verbose, char*& imLoc);

void balanceWork(int* numPointsArr, int* workArr, int n, int p);

#endif