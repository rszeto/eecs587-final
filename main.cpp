#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <map>
#include <getopt.h>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "util.hpp"

using namespace std;
using namespace cv;

RNG rng(12345);
/*
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
	// int totalNumFrames = getNumFrames(imDir);
	int totalNumFrames = 5;
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
	char* blobsLoc = "/home/szetor/shared/data/MOT/AVG-TownCentre/diffs/image_0001.png";
	Mat blobs = imread(blobsLoc, CV_LOAD_IMAGE_GRAYSCALE);
	Rect_<int> opRange(0, 0, blobs.cols, blobs.rows);

	Mat components = Mat::zeros(blobs.size(), CV_32SC1);
	getConnectedComponents(components, blobs, opRange);

	double minComp, maxComp;
	minMaxLoc(components, &minComp, &maxComp);
	vector<Scalar> colors;
	colors.push_back(Scalar(0, 0, 0));
	for(int i = 1; i <= maxComp; i++)
		colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255)));

	Mat final = Mat::zeros(blobs.size(), CV_8UC3);
	for(int y = 0; y < blobs.rows; y++) {
		for(int x = 0; x < blobs.cols; x++) {
			for(int c = 0; c < 3; c++) {
				final.at<Vec3b>(y, x)[c] = colors[(int)components.at<int>(y, x)][c];
			}
		}
		// cout << endl;
	}
	namedWindow("final", CV_WINDOW_KEEPRATIO);
	imshow("final", final);
	waitKey(0);
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

void findBorder() {
	char* imLoc = "/home/szetor/shared/binary2.png";
	Mat image = imread(imLoc, CV_LOAD_IMAGE_GRAYSCALE);
	Mat final = imread(imLoc, CV_LOAD_IMAGE_COLOR);
	Scalar green = Scalar(0, 255, 0);
	Scalar blue = Scalar(255, 0, 0);
	Scalar red = Scalar(0, 0, 255);

	vector<vector<Point> > contours;
	vector<Point> starts;
	starts.push_back(Point(23, 20));
	starts.push_back(Point(74, 27));
	starts.push_back(Point(118, 88));
	
	for(int i = 0; i < starts.size(); i++) {
		contours.push_back(findBorder(image, starts[i]));
		vector<Point> contour = findBorder(image, starts[i]);
		drawContour(final, contour, red);
	}
	namedWindow("display", CV_WINDOW_KEEPRATIO);
	imshow("display", final);
	waitKey(0);
}

void findBorder2() {
	Mat C = (Mat_<int>(5,5) << 0,0,0,0,0,0,1,2,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0);
	vector<Point> border = findBorder2(C, Point(1, 1), 1);
	for(int i = 0; i < border.size(); i++) {
		C.at<int>(border[i]) = 9;
	}
	for(int i = 0; i < C.rows; i++) {
		for(int j = 0; j < C.cols; j++) {
			cout << C.at<int>(i, j) << " ";
		}
		cout << endl;
	}
}
*/

int mpiMain(int argc, char** argv) {
	bool displayImages = false;
	bool verbose = false;
	char* imLoc = NULL;
	int optsRet = handleOpts(argc, argv, displayImages, verbose, imLoc);
	if(optsRet != 0) {
		return optsRet;
	}

	MPI_Init(&argc, &argv);
	
	// Get number of procs
	int p;
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	// Get square root of number of procs
	int sqP = sqrt(p);

	// Get rank
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Get row and column of this proc
	int rRow = rank / sqP;
	int rCol = rank % sqP;
	
	// Get locations of the original and contour images
	string origImageLoc(imLoc);
	string contImageLoc(imLoc);
	contImageLoc.replace(contImageLoc.find(".png"), 4, "_c.png");
	// Make sure both images exist
	ifstream origF(origImageLoc.c_str());
	ifstream contF(contImageLoc.c_str());
	if(!origF.good() || !contF.good()) {
		origF.close();
		contF.close();
		if(rank == 0) {
			fprintf(stderr, "Could not find both %s and %s, exiting\n",
					origImageLoc.c_str(), contImageLoc.c_str());
		}
		MPI_Finalize();
		return 0;
	}
	origF.close();
	contF.close();

	// Load contour image
	Mat contImage = imread(contImageLoc.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	// Max distance that points in the same cluster can be
	double thresh = 3.0;

	// Get width and height of image on most procs
	int normBlockWidth = contImage.cols / sqP;
	int normBlockHeight = contImage.rows / sqP;
	int localBlockWidth = normBlockWidth;
	int localBlockHeight = normBlockHeight;
	// Adjust width or height if this is the last proc
	if(rRow == sqP-1)
		localBlockHeight = contImage.rows - (sqP-1)*normBlockHeight;
	if(rCol == sqP-1)
		localBlockWidth = contImage.cols - (sqP-1)*normBlockWidth;

	// Extract local region
	Rect localROI(rCol*normBlockWidth, rRow*normBlockHeight, localBlockWidth, localBlockHeight);
	Mat localImage = contImage(localROI);

	if(verbose && localImage.rows < 15 && localImage.cols < 15) {
		for(int r = 0; r < p; r++) {
			if(rank == r) {
				cout << localImage << endl;
				cout << "=====" << rank << endl;
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	// Populate the contour points to use for clustering
	vector<Point> localPoints;
	// Correct offset of local region
	Point offset(rRow*normBlockHeight, rCol*normBlockWidth);
	for(int i = 0; i < localImage.rows; i++) {
		for(int j = 0; j < localImage.cols; j++) {
			if(localImage.at<uchar>(i, j) == 255)
				localPoints.push_back(Point(i, j) + offset);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double startTime;
	if(rank == 0) {
		startTime = MPI_Wtime();
	}

	// How many points this proc owns
	int numLocalPoints = localPoints.size();
	// Array keeping track of how many points each proc owns
	int numLocalPointsArr[p];
	// Array keeping track of how many points are owned by procs before this one
	int cumNumLocalPointsArr[p];
	// Array keeping track of which proc each point is on
	vector<int> pointToProc;
	// Total number of points
	int totalNumPoints = 0;
	// Populate above vars
	MPI_Allgather(&numLocalPoints, 1, MPI_INT, numLocalPointsArr, 1, MPI_INT, MPI_COMM_WORLD);
	for(int i = 0; i < p; i++) {
		cumNumLocalPointsArr[i] = (i == 0 ? 0 : cumNumLocalPointsArr[i-1] + numLocalPointsArr[i-1]);
		pointToProc.insert(pointToProc.end(), numLocalPointsArr[i], i);
	}
	totalNumPoints = pointToProc.size();
	if(verbose) {
		if(rank == 0) {
			cout << "Num points: " << totalNumPoints << endl;
			cout << "Num pairs: " << totalNumPoints*(totalNumPoints-1)/2 << endl;
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// Set cluster indexes for local points
	int clusterIndexes[numLocalPoints];
	for(int i = 0; i < numLocalPoints; i++) {
		int globalI = cumNumLocalPointsArr[rank] + i;
		clusterIndexes[i] = globalI;
	}

	// Init send to lower ranked procs
	float localPointsArr[2*numLocalPoints];
	for(int i = 0; i < numLocalPoints; i++) {
		localPointsArr[2*i] = localPoints[i].x;
		localPointsArr[2*i+1] = localPoints[i].y;
	}
	MPI_Request sendReqs[rank];
	for(int i = 0; i < rank; i++) {
		MPI_Isend(localPointsArr, 2*numLocalPoints, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &sendReqs[i]);
	}

	// Calculate distances between points that this proc owns
	// Note that we only store the upper triangular part of matrix, excluding the diagonal
	// Example:
	// xxxxxx
	// 0xxxxx
	// 00xxxx
	int distsWidth = totalNumPoints-cumNumLocalPointsArr[rank]-1;
	Mat dists = Mat(triMatSize(numLocalPoints, distsWidth), 1, CV_32F, numeric_limits<float>::max());
	for(int i = 0; i < numLocalPoints; i++) {
		for(int j = i+1; j < numLocalPoints; j++) {
			dists.at<float>(tupToLin(i, j-1, distsWidth)) = norm(localPoints[i]-localPoints[j]);
		}
	}

	// Request from higher ranked procs
	for(int otherRank = rank+1; otherRank < p; otherRank++) {
		float otherPointsArr[2*numLocalPointsArr[otherRank]];
		MPI_Recv(otherPointsArr, 2*numLocalPointsArr[otherRank], MPI_FLOAT, otherRank, 0, MPI_COMM_WORLD, NULL);
		// Reconstruct other points
		vector<Point> otherPoints;
		for(int i = 0; i < numLocalPointsArr[otherRank]; i++) {
			otherPoints.push_back(Point(otherPointsArr[2*i], otherPointsArr[2*i+1]));
		}
		// Calculate distance of other points from points on this proc
		for(int i = 0; i < numLocalPoints; i++) {
			for(int j = 0; j < otherPoints.size(); j++) {
				int whateverJ = j + cumNumLocalPointsArr[otherRank] - cumNumLocalPointsArr[rank] - 1;
				dists.at<float>(tupToLin(i, whateverJ, distsWidth)) = norm(localPoints[i]-otherPoints[j]);
			}
		}
	}

	// Print work size information
	if(verbose) {
		int localWork = dists.rows;
		
		int allDistsSize[p];
		MPI_Gather(&localWork, 1, MPI_INT, allDistsSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		if(rank == 0) {
			for(int r = 0; r < p; r++) {
				fprintf(stdout, "Work size for rank %d: %d\n", r, allDistsSize[r]);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// Merge clusters
	double iterTime = MPI_Wtime();
	for(int loopVar = 0; loopVar < totalNumPoints*(totalNumPoints-1)/2; loopVar++) {

		if(verbose && rank == 0 && loopVar % 200 == 0) {
			double curTime = MPI_Wtime();
			cout << "Iteration " << loopVar << "/" << totalNumPoints*(totalNumPoints-1)/2
					<< " (" << curTime-iterTime << "s)" << endl;
			iterTime = MPI_Wtime();
		}

		// Find the smallest distance
		double minDist;
		Point minLocIdx;
		minMaxLoc(dists, &minDist, NULL, &minLocIdx);

		// Cast minDist to float
		float minDistF = (float)minDist;
		// Handle case where matrix is empty
		if(minLocIdx == Point(-1, -1)) {
			minDistF = numeric_limits<float>::max();
		}

		struct {
			float val;
			int rank;
		} in, out;
		in.val = minDistF;
		in.rank = rank;
		MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
		// Quit if min value is too high
		if(out.val > thresh) {
			break;
		}

		if(rank == out.rank) {
			dists.at<float>(minLocIdx) = numeric_limits<float>::max();
		}

		// Convert dists index to location in full matrix
		Point minLoc = linToTup(minLocIdx.y, distsWidth);
		
		// Get global position of minimum
		int globalPos[2];
		globalPos[0] = cumNumLocalPointsArr[rank] + minLoc.x;
		globalPos[1] = cumNumLocalPointsArr[rank] + minLoc.y + 1;
		MPI_Bcast(globalPos, 2, MPI_INT, out.rank, MPI_COMM_WORLD);

		// Get which procs own the points whose clusters should be updated
		int rankA = pointToProc[globalPos[0]];
		int rankB = pointToProc[globalPos[1]];
		int clusterA, clusterB;
		// Get the cluster of the first point
		if(rank == rankA) {
			clusterA = clusterIndexes[globalPos[0]-cumNumLocalPointsArr[rank]];
		}
		MPI_Bcast(&clusterA, 1, MPI_INT, rankA, MPI_COMM_WORLD);
		// Get the cluster of the second point
		if(rank == rankB) {
			clusterB = clusterIndexes[globalPos[1]-cumNumLocalPointsArr[rank]];
		}
		MPI_Bcast(&clusterB, 1, MPI_INT, rankB, MPI_COMM_WORLD);
		// If the points were in different clusters, update
		if(clusterA != clusterB) {
			// Update cluster indexes
			for(int i = 0; i < numLocalPoints; i++) {
				if(clusterIndexes[i] == clusterA || clusterIndexes[i] == clusterB) {
					clusterIndexes[i] = min(clusterA, clusterB);
				}
			}
		}
	}

	// Gather cluster indexes from all procs
	int allClusterIndexes[totalNumPoints];
	MPI_Gatherv(clusterIndexes, numLocalPoints, MPI_INT, allClusterIndexes, numLocalPointsArr, cumNumLocalPointsArr, MPI_INT, 0, MPI_COMM_WORLD);
	// Gather points
	float allPointsArr[2*totalNumPoints];
	int dNumLocalPointsArr[p], dCumNumLocalPointsArr[p];
	for(int i = 0; i < p; i++) {
		dNumLocalPointsArr[i] = 2*numLocalPointsArr[i];
		dCumNumLocalPointsArr[i] = 2*cumNumLocalPointsArr[i];
	}
	MPI_Gatherv(localPointsArr, 2*numLocalPoints, MPI_FLOAT, allPointsArr, dNumLocalPointsArr, dCumNumLocalPointsArr, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0) {
		// Stop timing
		double endTime = MPI_Wtime();
		// Print info
		cout << "Num procs: " << p << endl;
		cout << "Total time (s): " << endTime-startTime << endl;

		// Convert allPointsArr to points vector
		vector<Point> allPoints;
		for(int i = 0; i < totalNumPoints; i++) {
			allPoints.push_back(Point(allPointsArr[2*i], allPointsArr[2*i+1]));
		}

		// Color the clusters
		Mat origImage = imread(origImageLoc.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		Mat final;
		cvtColor(origImage, final, CV_GRAY2RGB);
		map<int, Vec3b> clusterColors;
		for(int i = 0; i < totalNumPoints; i++) {
			int clusterIndex = allClusterIndexes[i];
			Point point = allPoints[i];
			if(clusterColors.count(clusterIndex) == 0) {
				Vec3b color(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
				clusterColors.insert(pair<int, Vec3b>(clusterIndex, color));
			}
			Vec3b color = clusterColors[clusterIndex];
			final.at<Vec3b>(point.x, point.y) = color;
		}

		if(displayImages) {
			namedWindow("final", CV_WINDOW_KEEPRATIO);
			imshow("final", origImage);
			waitKey(0);
			imshow("final", final);
			waitKey(0);
		}
	}

	MPI_Finalize();

	return 0;
}

int main(int argc, char** argv) {
	return mpiMain(argc, argv);
}
