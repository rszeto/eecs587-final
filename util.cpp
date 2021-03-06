#include "util.hpp"

using namespace std;
using namespace cv;

Point linToTup(int k, int n) {
	int i = n - 1 - floor(sqrt(-8*k + 4*(n+1)*n - 7)/2.0 - 0.5);
	int j = k + i - (n+1)*n/2 + (n-i+1)*(n-i)/2;
	return Point(i, j);
}

int tupToLin(int i, int j, int n) {
	return n*i + j - i*(i+1)/2;
}

int triMatSize(int rows, int cols) {
	return rows*cols - rows*(rows-1)/2;
}

int handleOpts(int argc, char** argv, bool& displayImages, bool& verbose, char*& imLoc, double& threshold) {
	displayImages = false;
	verbose = false;
	imLoc = NULL;
	threshold = -1;
	
	char optChar;
	while((optChar = getopt(argc, argv, "dvi:t:")) != -1) {
		switch(optChar) {
			case 'd':
				displayImages = true;
				break;
			case 'v':
				verbose = true;
				break;
			case 'i':
				imLoc = optarg;
				break;
			case 't':
				threshold = atof(optarg);
				break;
			case '?':
				if(optopt == 'i')
					cerr << "Image not specified with flag -i" << endl;
				if(optopt == 't')
					cerr << "Threshold not specified with flag -t" << endl;
				else
					cerr << "Unknown option " << optopt << endl;
				return 1;
			default:
				abort();
		}
	}
	if(imLoc == NULL) {
		cerr << "Image argument required" << endl;
		return 1;
	}
	if(threshold == -1) {
		cerr << "Threshold argument required" << endl;
		return 1;
	}
	return 0;
}

void balanceWork(int* numPointsArr, int n, int p) {
	// Calculate total amount of work
	double W = n*(n-1)/2;
	double W_avg = W/p;

	// Init arrays with 0
	for(int i = 0; i < p; i++) {
		numPointsArr[i] = 0;
	}
	int n_rem = n; // Remaining n

	for(int i = 0; i < p-1; i++) {
		// Calculate amount of work on this proc
		for(int n_local = 1; n_local <= n_rem; n_local++) {
			// int W_local_est = n_local*(n_rem-1) - n_local*(n_local-1)/2;
			int W_local_est = n_local*(n_rem-1);
			if(W_local_est > W_avg) {
				numPointsArr[i] = n_local;
				n_rem -= n_local;
				break;
			}
		}
	}
	numPointsArr[p-1] = n_rem;
}

void calculateWork(int* numPointsArr, int* workArr, int p) {
	// Get number of points
	int n = 0;
	int cumul[p];
	for(int i = 0; i < p; i++) {
		cumul[i] = n;
		n += numPointsArr[i];
	}
	for(int i = 0; i < p; i++) {
		int numAhead = n - cumul[i] - 1;
		workArr[i] = numPointsArr[i]*numAhead - numPointsArr[i]*(numPointsArr[i]-1)/2;
	}
}