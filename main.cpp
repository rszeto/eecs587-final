#include <iostream>
#include <cstdio>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencvUtil.hpp"

using namespace std;
using namespace cv;

RNG rng(12345);

int getNumFrames(char* imDir) {
	char cmd[512];
	sprintf(cmd, "ls %s/image_[0-9]*.png | wc -l", imDir);

	// Read command output. Obtained from http://stackoverflow.com/a/125866
	FILE *pipeOut = popen(cmd, "r");
	assert(pipeOut);
	char buffer[256];
	char *line_p = fgets(buffer, sizeof(buffer), pipeOut);
	pclose(pipeOut);

	return atoi(line_p);
}

void animateDiffs(char* imDir) {
	int totalNumFrames = getNumFrames(imDir);
	int threshold = 75;

	char buffer[256];
	sprintf(buffer, "%s/mean.png", imDir);
	Mat meanImg = imread(buffer, CV_LOAD_IMAGE_COLOR);
	if(!meanImg.data) {
		cerr << "Mean image does not exist" << endl;
		exit(1);
	}
	Rect_<int> opRange(0, 0, meanImg.cols, meanImg.rows);

	namedWindow("curImg");
	namedWindow("diffImg");
	sprintf(buffer, "mkdir %s/diffs", imDir);
	system(buffer);
	Scalar green = Scalar(0, 255, 0);
	for(int frameNum = 1; frameNum <= totalNumFrames; frameNum++) {
		sprintf(buffer, "%s/image_%04d.png", imDir, frameNum);
		Mat curImg = imread(buffer, CV_LOAD_IMAGE_COLOR);
		Mat diffImg = Mat::zeros(meanImg.rows, meanImg.cols, CV_8UC1);
		diff(diffImg, meanImg, curImg, opRange, threshold);

		imshow("curImg", curImg);
		imshow("diffImg", diffImg);

		sprintf(buffer, "%s/diffs/image_%04d.png", imDir, frameNum);
		imwrite(buffer, diffImg);
		waitKey(10);
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

void groupContours(char* imDir) {
	int totalNumFrames = getNumFrames(imDir);
	int threshold = 75;

	char buffer[256];
	sprintf(buffer, "%s/mean.png", imDir);
	Mat meanImg = imread(buffer, CV_LOAD_IMAGE_COLOR);
	if(!meanImg.data) {
		cerr << "Mean image does not exist" << endl;
		exit(1);
	}
	Rect_<int> opRange(0, 0, meanImg.cols, meanImg.rows);

	namedWindow("curImg");
	namedWindow("boundBoxImg");
	Scalar green = Scalar(0, 255, 0);
	for(int frameNum = 1; frameNum <= totalNumFrames; frameNum++) {
		sprintf(buffer, "%s/image_%04d.png", imDir, frameNum);
		Mat curImg = imread(buffer, CV_LOAD_IMAGE_COLOR);
		Mat diffImg = Mat::zeros(meanImg.rows, meanImg.cols, CV_8UC1);
		diff(diffImg, meanImg, curImg, opRange, threshold);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		getContours(contours, hierarchy, diffImg);

		UF contourClasses(contours.size());

		for(int i = 0; i < contours.size(); i++) {
			for(int j = i+1; j < contours.size(); j++) {
				if(closeContours(contours[i], contours[j], 10)) {
					contourClasses.merge(i, j);
				}
			}
		}

		// Combine contour clusters
		map<int, vector<Point> > contourClusters;
		for(int i = 0; i < contours.size(); i++) {
			for(int j = 0; j < contours[i].size(); j++) {
				contourClusters[contourClasses.find(i)].push_back(contours[i][j]);
			}
		}

		Mat boundBoxImg;
		cvtColor(diffImg, boundBoxImg, CV_GRAY2RGB);
		for(map<int, vector<Point> >::iterator it = contourClusters.begin(); it != contourClusters.end(); it++) {
			vector<Point> meh = it->second;
			Rect boundingBox = boundingRect(meh);
			rectangle(boundBoxImg, boundingBox, green);
			rectangle(curImg, boundingBox, green);
		}


		imshow("curImg", curImg);
		imshow("boundBoxImg", boundBoxImg);
		sprintf(buffer, "%s/diffs/image_%04d.png", imDir, frameNum);
		imwrite(buffer, boundBoxImg);
		waitKey(10);
	}
}

void averageImage(char* imDir) {
	int totalNumFrames = getNumFrames(imDir);

	char buffer[256];
	sprintf(buffer, "%s/image_0001.png", imDir);
	Mat curImage = imread(buffer, CV_LOAD_IMAGE_COLOR);
	long*** total = new long**[curImage.rows];
	for(int y = 0; y < curImage.rows; y++) {
		total[y] = new long*[curImage.cols];
		for(int x = 0; x < curImage.cols; x++) {
			total[y][x] = new long[3];
			for(int c = 0; c < 3; c++) {
				total[y][x][c] = 0;
			}
		}
	}

	for(int i = 1; i <= totalNumFrames; i++) {
		sprintf(buffer, "%s/image_%04d.png", imDir, i);
		curImage = imread(buffer, CV_LOAD_IMAGE_COLOR);
		for(int y = 0; y < curImage.rows; y++) {
			for(int x = 0; x < curImage.cols; x++) {
				for(int c = 0; c < 3; c++) {
					total[y][x][c] += (long)curImage.at<Vec3b>(y, x)[c];
				}
			}
		}
	}

	Mat avgImage = Mat::zeros(curImage.size(), CV_8UC3);
	for(int y = 0; y < curImage.rows; y++) {
		for(int x = 0; x < curImage.cols; x++) {
			for(int c = 0; c < 3; c++) {
				uchar avg = total[y][x][c]/totalNumFrames;
				avgImage.at<Vec3b>(y, x)[c] = avg;
			}
		}
	}
	sprintf(buffer, "%s/mean.png", imDir);
	imwrite(buffer, avgImage);

	// Free total array
	for(int y = 0; y < curImage.rows; y++) {
		for(int x = 0; x < curImage.cols; x++) {
			delete [] total[y][x];
		}
		delete total[y];
	}
	delete[] total;
}



int main(int argc, char** argv) {
	// animateDiffs(argv[1]);
	// findBlobs();
	groupContours(argv[1]);
	// averageImage(argv[1]);

	return 0;
}