#pragma once

#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include "Convolution.h"
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

typedef vector<double> keypoint;
typedef vector<double> descriptor;

// Return -1 if fail
// Return 0 if finish

class Detection {
public:
	int detectHarris(const Mat& srcImage, Mat& dstImage, double coefficient, double threshold);

	int detectBlob(const Mat& srcImage, Mat& dstImage, double sigma, double coefficient, double threshold);

	int detectDoG(const Mat& srcImage, Mat& dstImage, vector<keypoint>& keypoints, double sigma, double coefficient, double contrastThreshold, double eRatioThreshold);

	int extractSIFT(const Mat& srcImage, const vector<keypoint>& keypoints, vector<descriptor>& descriptors);

	int matchBySIFT(const Mat& srcImage1, const Mat& srcImage2, int detector, double sigma, double coefficient, double contrastThreshold, double eRatioThreshold, double distanceThreshold, Mat& dstImage);
	
	Detection();
	~Detection();
};
