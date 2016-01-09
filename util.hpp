#ifndef __UTIL_HPP
#define __UTIL_HPP

#include <cstdlib>
#include <iostream>
#include <getopt.h>
#include <opencv2/core/core.hpp>

cv::Point linToTup(int k, int n);
int tupToLin(int i, int j, int n);
int triMatSize(int rows, int cols);

int handleOpts(int argc, char** argv, bool& displayImages, bool& verbose, char*& imLoc, double& threshold);

void balanceWork(int* numPointsArr, int n, int p);
void calculateWork(int* numPointsArr, int* workArr, int p);

#endif