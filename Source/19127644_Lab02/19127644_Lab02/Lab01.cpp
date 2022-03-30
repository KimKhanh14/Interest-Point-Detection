// Lab01.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once
#include "Detection.h"

int main()
{
	Detection temp;
	Mat sourceImg1,sourceImg2, sourceImg3, desImg1, desImg2;
	string inputPath;
	inputPath = "C:\\Users\\Jennie\\Documents\\GitHub\\InterestPointDetection\\Source\\19127644_Lab02\\19127644_Lab02";
	for (int j = 1; j <= 10; j++) {
		string in1 = "";
		if (j < 10) {
			in1 = "0" + to_string(j) + ".jpg";
		}
		else
		{
			in1 = to_string(j) + ".jpg";
		}
		cout << inputPath + "\\TestImages\\" + in1 << endl;
		sourceImg1 = imread(inputPath + "\\TestImages\\" + in1);
		cvtColor(sourceImg1, sourceImg2, COLOR_RGB2GRAY);
		int x = temp.detectHarris(sourceImg2, desImg1, 0.06, 0.1);
		
		imwrite(inputPath + "\\Harris\\" + in1, desImg1);
	}
}
