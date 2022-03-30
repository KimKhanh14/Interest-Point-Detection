#include "Detection.h"

Detection::Detection() {}

Detection::~Detection() {}

int Detection::detectHarris(const Mat& srcImage, Mat& dstImage, double coefficient, double threshold)
{
	// coefficient from 0.04 to 0.06
	// threshold from 0.00 to 1.00
	// Return 1 if there is no image
	//if (srcImage.empty() || coefficient < 0.04 || coefficient > 0.06 || threshold < 0 || threshold >1) return -1; 
	if (srcImage.empty()) return -1;

	int row = srcImage.rows, col = srcImage.cols;

	// Smoothing image by GaussianBlur
	Mat blur;
	GaussianBlur(srcImage, blur, Size(0, 0), 1);

	// Sobel filter implementation
	double kX[] { -1, 0, 1, -2, 0, 2, -2, 0, 2 };
	double kY[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	// Convolution filter
	Convolution convolution;
	Mat dX, dY;

	convolution.SetKernel(kX, 3, 3);
	convolution.DoConvolution(blur, dX);

	convolution.SetKernel(kY, 3, 3);
	convolution.DoConvolution(blur, dY);

	//Harris response calculation
	Mat harris = Mat(row, col, CV_64FC1, Scalar(0));
	double *harrisdata = (double*)(harris.data);

	double *dXdata = (double*)(dX.data), *dYdata = (double*)(dY.data);

	for (int i = 1; i < row - 1; i++)
		for (int j = 1; j < col - 1; j++) {
			int center = i * col + j;

			double dx = 0, dy = 0, dxdy = 0;

			for (int u = -1; u <= 1; u++)
				for (int v = -1; v <= 1; v++) {
					int cur = center + u * col + v;

					double ix = *(dXdata + cur);
					double iy = *(dYdata + cur);

					dx += ix * ix;
					dy += iy * iy;
					dxdy += ix * iy;
				}

			*(harrisdata + center) = dx * dy - dxdy * dxdy - coefficient * (dx + dy) * (dx + dy);
		}

	// Non-maximum suppression
	double hMax = 0;

	for (int i = 1; i < row - 1; i++)
		for (int j = 1; j < col - 1; j++) {
			int center = i * col + j;

			double value = *(harrisdata + center);
			bool isMaximum = true, isMinimum = true;

			for (int u = -1; u <= 1; u++)
				for (int v = -1; v <= 1; v++)
					if (u != 0 || v != 0) {
						int cur = center + u * col + v;

						double neighbor = *(harrisdata + cur);

						if (value < neighbor)
							isMaximum = false;

						if (value > neighbor)
							isMinimum = false;
					}

			if (isMaximum || isMinimum) {
				hMax = MAX(hMax, value);
			}
			else {
				*(harrisdata + center) = 0;
			}
		}

	// Feature points thresholding and return to color image
	cvtColor(srcImage, dstImage, COLOR_GRAY2RGB);
	double thVal = threshold * hMax;

	for (int i = 1; i < row - 1; i++)
		for (int j = 1; j < col - 1; j++) {
			int center = i * col + j;

			if (*(harrisdata + center) > thVal)
				circle(dstImage, Point(j, i), 3, Scalar(0, 0, 255), -1);
		}

	return 0;
}

int Detection::detectBlob(const Mat& srcImage, Mat& dstImage, double sigma, double coefficient, double threshold) {
	// Return 1 if there is no image
	if (srcImage.empty()) return -1;

	int row = srcImage.rows, col = srcImage.cols;

	// Smoothing image by GaussianBlur
	Mat blur;
	GaussianBlur(srcImage, blur, Size(0, 0), 1);

	// Convert the source images to several binary images [0, 1]
	Mat norm;
	blur.convertTo(norm, CV_64FC1, 1.0 / 255);

	// Create value of sigma at every octave
	double sqrt2 = sqrt(2);
	double sig[10];
	short sqrt2sig[10];

	sig[0] = sigma;
	sqrt2sig[0] = short(ceil(sigma * sqrt2));

	for (int k = 1; k < 10; k++) {
		sig[k] = sig[k - 1] * coefficient;
		sqrt2sig[k] = short(ceil(sig[k] * sqrt2));
	}

	// Find maxima of squared Laplacian response in scale‐space
	Convolution convolution;
	Mat LOG[10];

	for (int k = 0; k < 10; k++) {
		Mat temp;

		convolution.SetScaleNormalizedLOG(sig[k]);
		convolution.DoConvolution(norm, temp);

		pow(temp, 2, LOG[k]);
	}

	// Non-maximum suppression
	cvtColor(srcImage, dstImage, COLOR_GRAY2BGR);

	for (int k = 1; k < 9; k++) {
		double* LOGkdata = (double*)(LOG[k].data);

		for (int i = 1; i < row - 1; i++)
			for (int j = 1; j < col - 1; j++) {
				int center = i * col + j;

				double value = *(LOGkdata + center);
				bool isMaximum = true, isMinimum = true;

				for (int t = -1; t <= 1; t++) {
					double* LOGtdata = (double*)(LOG[k + t].data);

					for (int u = -1; u <= 1; u++)
						for (int v = -1; v <= 1; v++)
							if (t != 0 || u != 0 || v != 0) {
								int cur = center + u * col + v;
								double neighbor = *(LOGtdata + cur);

								if (value < neighbor)
									isMaximum = false;

								if (value > neighbor)
									isMinimum = false;
							}
				}

				// Extreme threshold divergence
				if ((isMaximum || isMinimum) && value > threshold)
					circle(dstImage, Point(j, i), sqrt2sig[k], Scalar(0, 0, 255));
			}
	}

	return 0;
}

int Detection::detectDoG(const Mat& src, Mat& dst, vector<keypoint>& keypoints, double sigma, double coefficient, double contrastThreshold, double eRatioThreshold) {
	// Return 1 if there is no image
	if (src.empty()) return -1;

	int row = src.rows, col = src.cols;

	// Convert the source images to several binary images [0, 1]
	Mat norm;
	src.convertTo(norm, CV_64FC1, 1.0 / 255);

	// Create value of sigma at every octave
	double sig[5][5];
	sig[0][0] = sigma;

	for (int k = 1; k < 5; k++)
		sig[0][k] = sig[0][k - 1] * coefficient;

	double coef2 = coefficient * coefficient;
	for (int oct = 1; oct < 5; oct++)
		for (int k = 0; k < 5; k++)
			sig[oct][k] = sig[oct - 1][k] * coef2;

	// Laplacian of Gaussian
	Mat Gaussian[5][5];
	int scaleRow = row * 2, scaleCol = col * 2;

	for (int oct = 0; oct < 5; oct++) {
		Mat scale;
		resize(norm, scale, Size(scaleCol, scaleRow));

		for (int k = 0; k < 5; k++)
			GaussianBlur(scale, Gaussian[oct][k], Size(0, 0), sig[oct][k]);

		scaleRow = int(round(scaleRow * 0.5));
		scaleCol = int(round(scaleCol * 0.5));
	}

	// Difference of Gaussian
	Mat DOG[5][4];

	for (int oct = 0; oct < 5; oct++)
		for (int k = 0; k < 4; k++) {
			Mat temp = Gaussian[oct][k + 1] - Gaussian[oct][k];

			pow(temp, 2, DOG[oct][k]);
		}

	// Detect key-point
	keypoints.resize(0);
	cvtColor(src, dst, COLOR_GRAY2BGR);

	scaleRow = row * 2;
	scaleCol = col * 2;

	double sqrt2 = sqrt(2);
	double rescale = 0.5;

	for (int oct = 0; oct < 5; oct++) {
		for (int k = 1; k < 3; k++) {
			double* DOGkdata = (double*)(DOG[oct][k].data);

			for (int i = 1; i < scaleRow - 1; i++)
				for (int j = 1; j < scaleCol - 1; j++) {
					int center = i * scaleCol + j;

					double value = *(DOGkdata + center);
					bool isMaximum = true, isMinimum = true;

					for (int t = -1; t <= 1; t++) {
						double* DOGtdata = (double*)(DOG[oct][k + t].data);

						for (int u = -1; u <= 1; u++)
							for (int v = -1; v <= 1; v++)
								if (t != 0 || u != 0 || v != 0) {
									int cur = center + u * scaleCol + v;
									double neighbor = *(DOGtdata + cur);

									if (value < neighbor)
										isMaximum = false;

									if (value > neighbor)
										isMinimum = false;
								}
					}

					if (isMaximum || isMinimum) {
						// Using Taylor's Theorem at maximum value
						double* DOGadata = (double*)(DOG[oct][k + 1].data);
						double* DOGbdata = (double*)(DOG[oct][k - 1].data);

						// First derivative of DoG(x, y, sig)
						Mat DX = Mat(3, 1, CV_64FC1);
						double* DXdata = (double*)(DX.data);

						*(DXdata) = *(DOGkdata + center + 1) - *(DOGkdata + center - 1);
						*(DXdata + 1) = *(DOGkdata + center + scaleCol) - *(DOGkdata + center - scaleCol);
						*(DXdata + 2) = *(DOGadata + center) - *(DOGbdata + center);

						// Second derivative của hàm DoG(x, y, sig)
						Mat DXX = Mat(3, 3, CV_64FC1);
						double* DXXdata = (double*)(DXX.data);

						*(DXXdata) = *(DOGkdata + center + 1) + *(DOGkdata + center - 1) - 2 * *(DOGkdata + center);
						*(DXXdata + 4) = *(DOGkdata + center + scaleCol) + *(DOGkdata + center - scaleCol) - 2 * *(DOGkdata + center);
						*(DXXdata + 8) = *(DOGadata + center) + *(DOGbdata + center) - 2 * *(DOGkdata + center);

						*(DXXdata + 3) = *(DOGkdata + center + scaleCol + 1) + *(DOGkdata + center - scaleCol - 1);
						*(DXXdata + 3) -= *(DOGkdata + center - scaleCol + 1) + *(DOGkdata + center + scaleCol - 1);
						*(DXXdata + 1) = *(DXXdata + 3);

						*(DXXdata + 6) = *(DOGadata + center + 1) + *(DOGadata + center - 1);
						*(DXXdata + 6) -= *(DOGbdata + center + 1) + *(DOGbdata + center - 1);
						*(DXXdata + 2) = *(DXXdata + 6);

						*(DXXdata + 7) = *(DOGadata + center + scaleCol) + *(DOGadata + center - scaleCol);
						*(DXXdata + 7) -= *(DOGbdata + center + scaleCol) + *(DOGbdata + center - scaleCol);
						*(DXXdata + 5) = *(DXXdata + 7);

						// Maximum point of DoG(x, y, sig)
						Mat X = -DXX.inv() * DX; 

						// Maximum value
						Mat expr = 0.5 * DX.t() * X;
						double extrema = value + *(double*)(expr.data);

						// Only high contrast extremes are retained
						if (extrema > contrastThreshold) {
							// Approximate ratio of two eigenvalues
							double eRatio = (*(DXXdata) + *(DXXdata + 4)) * (*(DXXdata) + *(DXXdata + 4));
							eRatio /= *(DXXdata) * *(DXXdata + 4) - *(DXXdata + 1) * *(DXXdata + 1);

							// Keep only the extremes that are not on the edge
							if (eRatio < (eRatioThreshold + 1) * (eRatioThreshold + 1) / eRatioThreshold) {
								int originalX = int(round(j * rescale));
								int originalY = int(round(i * rescale));

								keypoint key{ double(oct), sig[oct][k], double(i), double(j) };
								keypoints.push_back(key);

								circle(dst, Point(originalX, originalY), 1, Scalar(0, 0, 255), -1);
							}
						}
					}

				}
		}

		scaleRow = int(round(scaleRow * 0.5));
		scaleCol = int(round(scaleCol * 0.5));
		rescale *= 2;
	}

	return 0;
}

int Detection::extractSIFT(const Mat& srcImage, const vector<keypoint>& keypoints, vector<descriptor>& descriptors) {
	// Return 1 if there is no image
	if (srcImage.empty())
		return -1;

	int row = srcImage.rows, col = srcImage.cols;

	// Create descriptor for every keypoint
	int num = int(keypoints.size());
	descriptors.resize(0);

	double k[] { -1, 0, 1 };
	Mat kX = Mat(1, 3, CV_64FC1, k);
	Mat kY = Mat(3, 1, CV_64FC1, k);

	for (int key = 0; key < num; key++) {
		int oct = int(round(keypoints[key][0]));
		double sig = keypoints[key][1];
		int ii = int(round(keypoints[key][2]));
		int jj = int(round(keypoints[key][3]));

		double factor = pow(2, -oct + 1);
		int scaleRow = int(round(row * factor)), scaleCol = int(round(col * factor));

		// Gaussian filter at maximum of keypoint
		Mat scale, Gaussian;
		resize(srcImage, scale, Size(scaleCol, scaleRow));
		GaussianBlur(scale, Gaussian, Size(0, 0), sig);

		// Choosing pixels around keypoint
		int minRow = max(0, ii - 8);
		int maxRow = min(scaleRow - 1, ii + 8);
		int minCol = max(0, jj - 8);
		int maxCol = min(scaleCol - 1, jj + 8);
		
		Mat win = Gaussian(Range(minRow, maxRow), Range(minCol, maxCol));
		int winRow = win.rows, winCol = win.cols;

		// Creating weigh matrix
		Mat weights = getGaussianKernel(winRow, 1.5 * sig) * getGaussianKernel(winCol, 1.5 * sig).t();

		// Derivatives in the X, Y directions
		Mat dX, dY;

		filter2D(win, dX, CV_64FC1, kX);
		filter2D(win, dY, CV_64FC1, kY);

		Mat mag, ang;
		cartToPolar(dX, dY, mag, ang, true);

		double* dXdata = (double*)(dX.data);
		double* magdata = (double*)(mag.data);
		double* angdata = (double*)(ang.data);

		// Smoothing the composite derivative
		mag = mag.mul(weights);

		// Divide 36 bin directions, calculate histogram, find main direction
		vector<double> bin36(36, 0);
		double max = -99999;
		double argmax = 0;

		for (int i = 0; i < winRow; i++)
			for (int j = 0; j < winCol; j++) {
				int center = i * winCol + j;

				double angle = *(angdata + center);

				int bin = int(angle / 10);
				bin36[bin] += *(magdata + center);

				if (max < bin36[bin]) {
					max = bin36[bin];
					argmax = angle;
				}
			}

		// Rotate for keypoint towards 0, divide by 8 bin directions, create descriptor
		ang = ang - argmax;
		descriptor des(128, 0);

		for (int i = 0; i < winRow; i++)
			for (int j = 0; j < winCol; j++) {
				int center = i * winCol + j;

				double angle = *(angdata + center);
				if (angle < 0)
					angle += 360;

				int bin = int(angle / 45);

				int pos = (int(i / 4) * 4 + int(j / 4)) * 8 + bin;

				des[pos] += *(magdata + center);
			}

		// Standardized
		double lengthinv = 0;

		for (int i = 0; i < 128; i++)
			lengthinv += des[i] * des[i];

		lengthinv = 1.0 / sqrt(lengthinv);

		for (int i = 0; i < 128; i++)
			des[i] *= lengthinv;

		descriptors.push_back(des);
	}

	return 0;
}

int Detection::matchBySIFT(const Mat& srcImage1, const Mat& srcImage2, int detector, double sigma, double coefficient, double contrastThreshold, double eRatioThreshold, double distanceThreshold, Mat& dstImage) {
	// Return 1 if there is no image
	if (srcImage1.empty() || srcImage2.empty()) return -1;

	Mat dst1, dst2;
	vector<keypoint> key1, key2;
	vector<descriptor> des1, des2;

	// Locating keypoints
	int temp;
	if (detector == 1) {
		temp = detectHarris(srcImage1, dst1, coefficient, contrastThreshold);
		temp = detectHarris(srcImage2, dst2, coefficient, contrastThreshold);
	}
	else if (detector == 2) {
		temp = detectBlob(srcImage1, dst1, sigma, coefficient, contrastThreshold);
		temp = detectBlob(srcImage2, dst2, sigma, coefficient, contrastThreshold);
	}
	else {
		temp = detectDoG(srcImage1, dst1, key1, sigma, coefficient, contrastThreshold, eRatioThreshold);
		temp = detectDoG(srcImage2, dst2, key2, sigma, coefficient, contrastThreshold, eRatioThreshold);
	}

	// Create descriptor for each keypoint
	extractSIFT(srcImage1, key1, des1);
	extractSIFT(srcImage2, key2, des2);

	// Display format
	dstImage = Mat(max(dst1.rows, dst2.rows), dst1.cols + dst2.cols, CV_8UC3, Scalar(0));
	dst1.copyTo(dstImage(Rect(0, 0, dst1.cols, dst1.rows)));
	dst2.copyTo(dstImage(Rect(dst1.cols, 0, dst2.cols, dst2.rows)));

	// Match the descriptor by 1-NN
	for (int i = 0; i < des1.size(); i++) {
		double min = 9999999;
		int argmin = -1;

		for (int j = 0; j < des2.size(); j++) {
			double sdist = 0;
			for (int k = 0; k < 128; k++)
				sdist += pow(des1[i][k] - des2[j][k], 2);
			if (sdist < min) {
				min = sdist;
				argmin = j;
			}
		}

		// Distance thresholding
		if (min <= distanceThreshold) {
			int x1 = int(round(key1[i][3] * pow(2, key1[i][0] - 1)));
			int y1 = int(round(key1[i][2] * pow(2, key1[i][0] - 1)));

			int x2 = int(round(key2[argmin][3] * pow(2, key2[argmin][0] - 1))) + dst1.cols;
			int y2 = int(round(key2[argmin][2] * pow(2, key2[argmin][0] - 1)));

			line(dstImage, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0));
		}
	}

	return 0;
}
