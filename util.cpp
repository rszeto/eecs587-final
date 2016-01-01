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

int handleOpts(int argc, char** argv, bool& displayImages, bool& verbose, char*& imLoc) {
	char optChar;
	while((optChar = getopt(argc, argv, "dvi:")) != -1) {
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
			case '?':
				if(optopt == 'i')
					cerr << "Image not specified with flag -i" << endl;
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
	return 0;
}