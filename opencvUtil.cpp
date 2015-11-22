#include "opencvUtil.hpp"

void diff(Mat& result, const Mat& frameA, const Mat& frameB, const Rect_<int> opRange, double threshold) {
	for(int y = opRange.y; y < opRange.y+opRange.height; y++) {
		for(int x = opRange.x; x < opRange.x+opRange.width; x++) {
			Vec3b imagePix = frameA.at<Vec3b>(y, x);
			Vec3b imagePix2 = frameB.at<Vec3b>(y, x); 
			Vec3i diff(0, 0, 0);
			for(int c = 0; c < 3; c++) {
				diff[c] = (int)(imagePix2[c] - imagePix[c]);
			}
			if(norm(diff, NORM_L2) > threshold) {
				result.at<uchar>(y, x) = 255;
			}
		}
	}
}

// Given a binary image, locate the connected components in the image
void getConnectedComponents(Mat& componentLabels, const Mat& components, const Rect_<int> opRange) {
    // Make sure matrices are of the correct type
    // componentLabels should be CV_32SC1, components should be CV_8UC1
    assert(componentLabels.type() == CV_32SC1);
    assert(components.type() == CV_8UC1);

    // The disjoint set structure that keeps track of component classes
    UF compClasses(DEFAULT_NUM_CC);
    // Counter for the components in the image
    int regCounter = 1;
    for(int i = opRange.y; i < opRange.y+opRange.height; i++) {
        for(int j = opRange.x; j < opRange.x+opRange.width; j++) {
            // Set component class if mask is white at current pixel
            if(components.at<uchar>(i, j) == 255) {
                // Check surrounding pixels
                if(i-1 < 0) {
                    // On top boundary, so just check left
                    if(j-1 < 0) {
                        // This is the TL pixel, so set as new class
                        componentLabels.at<int>(i, j) = regCounter;
                        regCounter++;
                    }
                    else if(componentLabels.at<int>(i, j-1) == 0) {
                        // No left neighbor, so set pixel as new class
                        componentLabels.at<int>(i, j) = regCounter;
                        regCounter++;
                    }
                    else {
                        // Assign pixel class to the same as left neighbor
                        componentLabels.at<int>(i, j) = componentLabels.at<int>(i, j-1);
                    }
                }
                else {
                    if(j-1 < 0) {
                        // On left boundary, so just check top
                        if(componentLabels.at<int>(i-1, j) == 0) {
                            // No top neighbor, so set pixel as new class
                            componentLabels.at<int>(i, j) = regCounter;
                            regCounter++;
                        }
                        else {
                            // Assign pixel class to same as top neighbor
                            componentLabels.at<int>(i, j) = componentLabels.at<int>(i-1, j);
                        }
                    }
                    else {
                        // Normal case (get top and left neighbor and reassign classes if necessary)
                        int topClass = componentLabels.at<int>(i-1, j);
                        int leftClass = componentLabels.at<int>(i, j-1);
                        if(topClass == 0 && leftClass == 0) {
                            // No neighbor exists, so set pixel as new class
                            componentLabels.at<int>(i, j) = regCounter;
                            regCounter++;
                        }
                        else if(topClass == 0 && leftClass != 0) {
                            // Only left neighbor exists, so copy its class
                            componentLabels.at<int>(i, j) = leftClass;
                        }
                        else if(topClass != 0 && leftClass == 0) {
                            // Only top neighbor exists, so copy its class
                            componentLabels.at<int>(i, j) = topClass;
                        }
                        else {
                            // Both neighbors exist
                            int minNeighbor = std::min(componentLabels.at<int>(i-1, j), componentLabels.at<int>(i, j-1));
                            int maxNeighbor = std::max(componentLabels.at<int>(i-1, j), componentLabels.at<int>(i, j-1));
                            componentLabels.at<int>(i, j) = minNeighbor;
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
                componentLabels.at<int>(i, j) = 0;
            }
        }
    }
    // Unify the labels such that every pixel in a component has the same label
    for(int i = opRange.y; i < opRange.y+opRange.height; i++) {
        for(int j = opRange.x; j < opRange.x+opRange.width; j++) {
            componentLabels.at<int>(i, j) = compClasses.find(componentLabels.at<int>(i, j));
        }
    }
}

void getContours(vector<vector<Point> >& contours, vector<Vec4i>& hierarchy, const Mat& image) {
    Mat imageClone = image.clone();
    findContours( imageClone, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
}

bool closeContours(const vector<Point>& contA, const vector<Point>& contB, double threshold) {
    for(int i = 0; i < contA.size(); i++) {
        for(int j = 0; j < contB.size(); j++) {
            if(norm(contA[i]-contB[j]) < threshold) {
                return true;
            }
        }
    }
    return false;
}