#include "opencvUtil.hpp"

void diff(Mat& result, const Mat& frameA, const Mat& frameB, const range rangeVert, const range rangeHoriz, double threshold) {
	for(int i = rangeVert.a; i <= rangeVert.b; i++) {
		for(int j = rangeHoriz.a; j <= rangeHoriz.b; j++) {
			Vec3b imagePix = frameA.at<Vec3b>(i, j);
			Vec3b imagePix2 = frameB.at<Vec3b>(i, j); 
			Vec3i diff(0, 0, 0);
			for(int c = 0; c < 3; c++) {
				diff[c] = (int)(imagePix2[c] - imagePix[c]);
			}
			if(norm(diff, NORM_L2) > threshold) {
				result.at<uchar>(i, j) = 255;
			}
		}
	}
}

// Given a binary image, locate the connected components in the image
// WARNING: The returned array must be manually destroyed after use.
int** getConnectedComponents(const Mat& components) {
    // Make sure the matrix is of the correct type (uchar)
    assert(components.type() == CV_8UC1);
    int** componentLabels = new int*[components.rows];
    for(int i = 0; i < components.rows; i++) {
        componentLabels[i] = new int[components.cols];
    }

    // The disjoint set structure that keeps track of component classes
    UF compClasses(DEFAULT_NUM_CC);
    // Counter for the components in the image
    int regCounter = 1;
    for(int i = 0; i < components.rows; i++) {
        for(int j = 0; j < components.cols; j++) {
            // Set component class if mask is white at current pixel
            if(components.at<uchar>(i, j) == 255) {
                // Check surrounding pixels
                if(i-1 < 0) {
                    // On top boundary, so just check left
                    if(j-1 < 0) {
                        // This is the TL pixel, so set as new class
                        componentLabels[i][j] = regCounter;
                        regCounter++;
                    }
                    else if(componentLabels[i][j-1] == -1) {
                        // No left neighbor, so set pixel as new class
                        componentLabels[i][j] = regCounter;
                        regCounter++;
                    }
                    else {
                        // Assign pixel class to the same as left neighbor
                        componentLabels[i][j] = componentLabels[i][j-1];
                    }
                }
                else {
                    if(j-1 < 0) {
                        // On left boundary, so just check top
                        if(componentLabels[i-1][j] == -1) {
                            // No top neighbor, so set pixel as new class
                            componentLabels[i][j] = regCounter;
                            regCounter++;
                        }
                        else {
                            // Assign pixel class to same as top neighbor
                            componentLabels[i][j] = componentLabels[i-1][j];
                        }
                    }
                    else {
                        // Normal case (get top and left neighbor and reassign classes if necessary)
                        int topClass = componentLabels[i-1][j];
                        int leftClass = componentLabels[i][j-1];
                        if(topClass == -1 && leftClass == -1) {
                            // No neighbor exists, so set pixel as new class
                            componentLabels[i][j] = regCounter;
                            regCounter++;
                        }
                        else if(topClass == -1 && leftClass != -1) {
                            // Only left neighbor exists, so copy its class
                            componentLabels[i][j] = leftClass;
                        }
                        else if(topClass != -1 && leftClass == -1) {
                            // Only top neighbor exists, so copy its class
                            componentLabels[i][j] = topClass;
                        }
                        else {
                            // Both neighbors exist
                            int minNeighbor = std::min(componentLabels[i-1][j], componentLabels[i][j-1]);
                            int maxNeighbor = std::max(componentLabels[i-1][j], componentLabels[i][j-1]);
                            componentLabels[i][j] = minNeighbor;
                            // If we have differing neighbor values, merge them
                            if(minNeighbor != maxNeighbor) {
                                compClasses.merge(minNeighbor, maxNeighbor);
                            }
                        }
                    }
                }
            }
            else {
                // The pixel is black, so do not give a component label
                componentLabels[i][j] = -1;
            }
        }
    }
    // Unify the labels such that every pixel in a component has the same label
    for(int i=0; i < components.rows; i++) {
        for(int j=0; j < components.cols; j++) {
            componentLabels[i][j] = compClasses.find(componentLabels[i][j]);
        }
    }
    return componentLabels;
}