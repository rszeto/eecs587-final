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

	// Create cartesian topology over procs
	MPI_Comm MPI_COMM_CART;
	MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){sqP,sqP}, (int[]){false,false}, true, &MPI_COMM_CART);

	// Get rank
	int rank;
	MPI_Comm_rank(MPI_COMM_CART, &rank);
	// Get row and column of this proc
	int coords[2];
	MPI_Cart_coords(MPI_COMM_CART, rank, 2, coords);
	int rRow = coords[0];
	int rCol = coords[1];
	
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

	// Figure out how many procs need to be communicated with based on threshold
	int maxProcDistVert = ceil(thresh/normBlockHeight);
	int maxProcDistHoriz = ceil(thresh/normBlockWidth);

	// Find the procs to communicate with
	vector<int> connectedProcsUpper;
	vector<int> connectedProcsLower;
	for(int i = -maxProcDistVert; i <= maxProcDistVert; i++) {
		for(int j = -maxProcDistHoriz; j <= maxProcDistHoriz; j++) {
				// Skip horizontal out of bounds
			if(rRow+i >= 0 && rRow+i < sqP) {
					// Skip vertical out of bounds
				if(rCol+j >= 0 && rCol+j < sqP) {
					int adjProc;
					MPI_Cart_rank(MPI_COMM_CART, (int[]){rRow+i,rCol+j}, &adjProc);
					if(adjProc > rank)
						connectedProcsUpper.push_back(adjProc);
					else if(adjProc < rank)
						connectedProcsLower.push_back(adjProc);
				}
			}
		}
	}

	// Extract local region
	Rect localROI(rCol*normBlockWidth, rRow*normBlockHeight, localBlockWidth, localBlockHeight);
	Mat localImage = contImage(localROI);

	if(verbose && localImage.rows < 15 && localImage.cols < 15) {
		for(int r = 0; r < p; r++) {
			if(rank == r) {
				cout << localImage << endl;
				cout << "=====" << rank << endl;
			}
			MPI_Barrier(MPI_COMM_CART);
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

	MPI_Barrier(MPI_COMM_CART);
	double startTime;
	if(rank == 0) {
		startTime = MPI_Wtime();
	}

	// How many points this proc owns
	int numLocalPoints = localPoints.size();
	// Array keeping track of how many points each proc owns
	int numLocalPointsArr[p];
	// Cumulated version of above array (ie. cumulation over all procs)
	int cumNumLocalPointsArr[p];
	// Array keeping track of which proc each point is on
	vector<int> pointToProc;
	// Total number of points
	int totalNumPoints = 0;
	// Populate above vars
	MPI_Allgather(&numLocalPoints, 1, MPI_INT, numLocalPointsArr, 1, MPI_INT, MPI_COMM_CART);
	for(int i = 0; i < p; i++) {
		cumNumLocalPointsArr[i] = (i == 0 ? 0 : cumNumLocalPointsArr[i-1] + numLocalPointsArr[i-1]);
		pointToProc.insert(pointToProc.end(), numLocalPointsArr[i], i);
	}
	totalNumPoints = pointToProc.size();
	if(verbose && rank == 0) {
		cout << "Num points: " << totalNumPoints << endl;
		cout << "Num pairs: " << totalNumPoints*(totalNumPoints-1)/2 << endl;
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
	MPI_Request sendReqs[connectedProcsLower.size()];
	for(int i = 0; i < connectedProcsLower.size(); i++) {
		MPI_Isend(localPointsArr, 2*numLocalPoints, MPI_FLOAT, connectedProcsLower[i], 0, MPI_COMM_CART, &sendReqs[i]);
	}

	// Figure out width of dists matrix, which keeps track of distances between points owned
	// by this proc and points owned by higher procs
	int numLocalAndUpperPts = 0;
	vector<int> cumNumLocalAndUpperPts;
	// Vector mapping column indexes to global ones
	vector<int> localAndUpperIdxToGlobal;
	// Add points on this processor
	cumNumLocalAndUpperPts.push_back(numLocalAndUpperPts);
	numLocalAndUpperPts += numLocalPoints;
	for(int i = 0; i < numLocalPoints; i++)
		localAndUpperIdxToGlobal.push_back(cumNumLocalPointsArr[rank]+i);
	// Add points from upper procs
	for(int r = 0; r < connectedProcsUpper.size(); r++) {
		int otherRank = connectedProcsUpper[r];
		cumNumLocalAndUpperPts.push_back(numLocalAndUpperPts);
		numLocalAndUpperPts += numLocalPointsArr[otherRank];
		for(int i = 0; i < numLocalPointsArr[otherRank]; i++)
			localAndUpperIdxToGlobal.push_back(cumNumLocalPointsArr[otherRank]+i);
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

	// Request from higher connected procs
	for(int r = 0; r < connectedProcsUpper.size(); r++) {
		int otherRank = connectedProcsUpper[r];
		float otherPointsArr[2*numLocalPointsArr[otherRank]];
		MPI_Recv(otherPointsArr, 2*numLocalPointsArr[otherRank], MPI_FLOAT, otherRank, 0, MPI_COMM_CART, NULL);
		// Reconstruct other points
		vector<Point> otherPoints;
		for(int i = 0; i < numLocalPointsArr[otherRank]; i++) {
			otherPoints.push_back(Point(otherPointsArr[2*i], otherPointsArr[2*i+1]));
		}
		// Calculate distance of other points from points on this proc
		for(int i = 0; i < numLocalPoints; i++) {
			int start = cumNumLocalAndUpperPts[1+r];
			for(int j = 0; j < numLocalPointsArr[otherRank]; j++) {
				int whateverJ = start+j-1;
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
		MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_CART);
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
		if(rank == out.rank) {
			globalPos[0] = localAndUpperIdxToGlobal[minLoc.x];
			globalPos[1] = localAndUpperIdxToGlobal[minLoc.y+1];
		}
		MPI_Bcast(globalPos, 2, MPI_INT, out.rank, MPI_COMM_CART);

		// Get which procs own the points whose clusters should be updated
		int rankA = pointToProc[globalPos[0]];
		int rankB = pointToProc[globalPos[1]];
		int clusterA, clusterB;
		// Get the cluster of the first point
		if(rank == rankA) {
			clusterA = clusterIndexes[globalPos[0]-cumNumLocalPointsArr[rank]];
		}
		MPI_Bcast(&clusterA, 1, MPI_INT, rankA, MPI_COMM_CART);
		// Get the cluster of the second point
		if(rank == rankB) {
			clusterB = clusterIndexes[globalPos[1]-cumNumLocalPointsArr[rank]];
		}
		MPI_Bcast(&clusterB, 1, MPI_INT, rankB, MPI_COMM_CART);
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
	MPI_Gatherv(clusterIndexes, numLocalPoints, MPI_INT, allClusterIndexes, numLocalPointsArr, cumNumLocalPointsArr, MPI_INT, 0, MPI_COMM_CART);
	// Gather points
	float allPointsArr[2*totalNumPoints];
	int dNumLocalPointsArr[p], dCumNumLocalPointsArr[p];
	for(int i = 0; i < p; i++) {
		dNumLocalPointsArr[i] = 2*numLocalPointsArr[i];
		dCumNumLocalPointsArr[i] = 2*cumNumLocalPointsArr[i];
	}
	MPI_Gatherv(localPointsArr, 2*numLocalPoints, MPI_FLOAT, allPointsArr, dNumLocalPointsArr, dCumNumLocalPointsArr, MPI_FLOAT, 0, MPI_COMM_CART);
	MPI_Barrier(MPI_COMM_CART);

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
