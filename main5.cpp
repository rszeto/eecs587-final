#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <map>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "util.hpp"

using namespace std;
using namespace cv;

RNG rng(12345);

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

	// Find all contour points
	vector<Point> allPoints;
	for(int i = 0; i < contImage.rows; i++) {
		for(int j = 0; j < contImage.cols; j++) {
			if(contImage.at<uchar>(i, j) == 255)
				allPoints.push_back(Point(i, j));
		}
	}

	// Figure out how many points each proc should have
	int numLocalPointsArr[p];
	int cumNumLocalPointsArr[p];
	balanceWork(numLocalPointsArr, allPoints.size(), p);
	for(int r = 0; r < p; r++) {
		cumNumLocalPointsArr[r] = (r == 0 ? 0 : cumNumLocalPointsArr[r-1] + numLocalPointsArr[r-1]);
	}

	// Assign local points
	vector<Point> localPoints;
	for(int i = 0; i < numLocalPointsArr[rank]; i++) {
		localPoints.push_back(allPoints[cumNumLocalPointsArr[rank] + i]);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double startTime;
	if(rank == 0) {
		startTime = MPI_Wtime();
	}

	// How many points this proc owns
	int numLocalPoints = numLocalPointsArr[rank];
	// Array keeping track of which proc each point is on
	vector<int> pointToProc;
	// Total number of points
	int totalNumPoints = 0;
	for(int i = 0; i < p; i++) {
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
