#include <iostream>
#include "InterestPointDetection.h"

using namespace std;

int main(int argc, char* argv[])
{
	string option, inputPath, outputPath;
	if (argc == 4) {
		inputPath = argv[1];
		outputPath = argv[2];
		double c = stof(argv[3]);
		double t = stof(argv[4]);

		Mat sourceImg = imread(inputPath);
		Mat desImg;
		InterestPointDetection temp;

		int h = temp.detectHarris(sourceImg, desImg, c, t);

	}
	else if (argc == 5) {
		option = argv[1];
		inputPath = argv[3];
		outputPath = argv[4];
		float value = stof(argv[2]);

		Mat sourceImg = imread(inputPath);
		Mat desImg;
		
	}
}