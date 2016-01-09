#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <map>
#include <string>
#include <fstream>
#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "util.hpp"

using namespace std;
using namespace cv;

RNG rng(12345);

typedef struct distStruct {
	int i;
	int j;
	float dist;
} distStruct;

class DistStructComparator {
	public:
		bool operator() (distStruct a, distStruct b) {
			return a.dist > b.dist;
		}
};

int mpiMain(int argc, char** argv) {
	bool displayImages;
	bool verbose;
	char* imLoc;
	double thresh;
	int optsRet = handleOpts(argc, argv, displayImages, verbose, imLoc, thresh);
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
	
	// Make sure given image exists
	ifstream origF(imLoc);
	if(!origF.good()) {
		origF.close();
		if(rank == 0) {
			fprintf(stderr, "Could not find %s, exiting\n", imLoc);
		}
		MPI_Finalize();
		return 0;
	}
	origF.close();

	// Load contour image
	Mat image = imread(imLoc, CV_LOAD_IMAGE_GRAYSCALE);

	// Get width and height of image on most procs
	int normBlockWidth = image.cols / sqP;
	int normBlockHeight = image.rows / sqP;
	int localBlockWidth = normBlockWidth;
	int localBlockHeight = normBlockHeight;
	// Adjust width or height if this is the last proc
	if(rRow == sqP-1)
		localBlockHeight = image.rows - (sqP-1)*normBlockHeight;
	if(rCol == sqP-1)
		localBlockWidth = image.cols - (sqP-1)*normBlockWidth;

	// Extract local region
	Rect localROI(rCol*normBlockWidth, rRow*normBlockHeight, localBlockWidth, localBlockHeight);
	Mat localImage = image(localROI);

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
	
	double distStartTime;
	if(verbose) {
		MPI_Barrier(MPI_COMM_WORLD);
		distStartTime = MPI_Wtime();
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
	
	// Populate distances PQ
	priority_queue<distStruct, vector<distStruct>, DistStructComparator> dists;
	for(int i = 0; i < numLocalPoints; i++) {
		for(int j = i+1; j < numLocalPoints; j++) {
			dists.push({i, j, norm(localPoints[i]-localPoints[j])});
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
				int whateverJ = j + cumNumLocalPointsArr[otherRank] - cumNumLocalPointsArr[rank];
				dists.push({i, whateverJ, norm(localPoints[i]-otherPoints[j])});
			}
		}
	}

	if(verbose) {
		MPI_Barrier(MPI_COMM_WORLD);
		double distEndTime = MPI_Wtime();
		if(rank == 0) {
			cout << "Populating dists took " << distEndTime-distStartTime << "s" << endl;
		}
	}

	// Print work size information
	if(verbose) {
		int localWork = dists.size();
		
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

		if(verbose && rank == 0 && loopVar % 2000 == 0) {
			double curTime = MPI_Wtime();
			cout << "Iteration " << loopVar << "/" << totalNumPoints*(totalNumPoints-1)/2
					<< " (" << curTime-iterTime << "s)" << endl;
			iterTime = MPI_Wtime();
		}

		// Find the smallest distance
		float minDistF = numeric_limits<float>::max();
		Point minLoc;
		if(!dists.empty()) {
			distStruct minDistStruct = dists.top();
			minDistF = minDistStruct.dist;
			minLoc = Point(minDistStruct.i, minDistStruct.j);
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

		// If the global min was found on this proc, remove it
		if(rank == out.rank) {
			dists.pop();
		}
		
		// Get global position of minimum
		int globalPos[2];
		globalPos[0] = cumNumLocalPointsArr[rank] + minLoc.x;
		globalPos[1] = cumNumLocalPointsArr[rank] + minLoc.y;
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
		Mat final;
		cvtColor(image, final, CV_GRAY2RGB);
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
			imshow("final", image);
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
