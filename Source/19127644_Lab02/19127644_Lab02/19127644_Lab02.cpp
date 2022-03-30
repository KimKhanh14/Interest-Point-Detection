#pragma once
#include <iostream>
#include "Detection.h"
#include "CornerDetection.h"

using namespace std;
using namespace cv;

//C:\Users\Jennie\Documents\GitHub\InterestPointDetection\Source\19127644_Lab02\x64\Debug

int main(int argc, char* argv[]) {
	string option, inputPath, inputPath1, inputPath2;

	//.\19127644_Lab02.exe -harris C:\Users\Jennie\Documents\GitHub\InterestPointDetection\Source\19127644_Lab02\x64\Debug 0.04 0.06
	if (argc == 5) {
		//.\19127644_Lab02.exe -harris 01_1.jpg 0.04 0.06
		inputPath = argv[2];
		double coefficient = stof(argv[3]);
		double threshold = stof(argv[4]);

		Mat sourceImg = imread(inputPath, IMREAD_GRAYSCALE);
		Mat desImg;
		Detection temp;

		int h = temp.detectHarris(sourceImg, desImg, coefficient, threshold);

		if(h == -1) {
			cout << "No image or wrong input number!";
			return -1;
		} else
		{
			namedWindow("Results", WINDOW_NORMAL);
			imshow("Results", desImg);
		}
	}
	//.\19127644_Lab02.exe -blob C:\Users\Jennie\Documents\GitHub\InterestPointDetection\Source\19127644_Lab02\x64\Debug 100 0.04 0.06
	//.\19127644_Lab02.exe -blob 01_1.jpg 0.04 0.06
	else if (argc == 6) {
		inputPath = argv[2];
		double sigma = stof(argv[3]);
		double coefficient = stof(argv[4]);
		double threshold = stof(argv[5]);

		Mat sourceImg = imread(inputPath, IMREAD_GRAYSCALE);
		Mat desImg;
		Detection temp;

		int b = temp.detectBlob(sourceImg, desImg, sigma, coefficient, threshold);

		if (b == -1) {
			cout << "No image or wrong input number!";
			return -1;
		}
		else
		{
			namedWindow("Results", WINDOW_NORMAL);
			imshow("Results", desImg);
		}
		
	}
	//.\19127644_Lab02.exe -dog C:\Users\Jennie\Documents\GitHub\InterestPointDetection\Source\19127644_Lab02\x64\Debug 0.04 0.04 0.04 0.06
	//.\19127644_Lab02.exe -dog 01_1.jpg 0.04 0.06
	else if (argc == 7) {
		inputPath = argv[2];
		vector<keypoint> keypoints;
		double sigma = stof(argv[3]);
		double coefficient = stof(argv[4]);
		double contrastThreshold = stof(argv[5]);
		double eRatioThreshold = stof(argv[6]);

		Mat sourceImg = imread(inputPath, IMREAD_GRAYSCALE);
		Mat desImg;
		Detection temp;

		int d = temp.detectDoG(sourceImg, desImg, keypoints, sigma, coefficient, contrastThreshold, eRatioThreshold);

		if (d == -1) {
			cout << "No image or wrong input number!";
			return -1;
		}
		else
		{
			namedWindow("Results", WINDOW_NORMAL);
			imshow("Results", desImg);
		}
	}
	else if (argc == 10) {
		inputPath1 = argv[2];
		inputPath2 = argv[3];
		int detector = stof(argv[4]);
		double sigma = stof(argv[5]);
		double coefficient = stof(argv[6]);
		double contrastThreshold = stof(argv[7]);
		double eRatioThreshold = stof(argv[8]);
		double distanceThreshold = stof(argv[9]);

		Mat sourceImg1 = imread(inputPath1, IMREAD_GRAYSCALE);
		Mat sourceImg2 = imread(inputPath2, IMREAD_GRAYSCALE);
		Mat desImg;
		Detection temp;

		int s = temp.matchBySIFT(sourceImg1, sourceImg2, detector, sigma, coefficient, contrastThreshold, eRatioThreshold, distanceThreshold, desImg);

		if (s == -1) {
			cout << "No image or wrong input number!";
			return -1;
		}
		else
		{
			namedWindow("Results", WINDOW_NORMAL);
			imshow("Results", desImg);
		}

	}
}