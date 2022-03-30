#pragma once
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

class CornerDetection
{
private:
    vector<KeyPoint> keypoints_container;
public:
    CornerDetection() {};
    ~CornerDetection() {};
	Mat detectCornerHarris(Mat src, int thresh);
    Mat detectBlob(Mat src);
    Mat detectDOG(Mat src, int thresh);
    Mat matchBySIFT(Mat img1, Mat img2, int detector, int thresh);
    vector<KeyPoint> detectKeypoint(Mat img, int detector, int thresh);
};


Mat CornerDetection::detectCornerHarris(Mat src, int thresh)
{
    //Image Matrix
    Mat gray;
    Mat dst, dst_norm;
    //Harris Corner atributes
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    //Drawing Circle atributes
    int radius = 8;

    cvtColor(src, gray, COLOR_BGR2GRAY);
    dst = Mat::zeros(src.size(), CV_32FC1);
    // Detecting corners
    cornerHarris(gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    // Normalizing
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // Drawing a circle around corners
    for (int j = 0; j < dst_norm.rows; j++)
    {
        for (int i = 0; i < dst_norm.cols; i++)
        {

            if ((int)dst_norm.at<float>(j, i) > thresh)
            {
                circle(src, Point(i, j), radius, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
    return dst_norm;
}

Mat CornerDetection::detectBlob(Mat src)
{
    //Image Matrix
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    //Blob Detector Setup param----------------------
    SimpleBlobDetector::Params params;
        // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 255;
        // Filter by Area.
    params.filterByArea = true;
    params.minArea = 25;
        // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;
        // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.5;
        // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    //------------------------------------
    vector<KeyPoint> keypoints;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(gray, keypoints);
    this->keypoints_container = keypoints;
    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
    Mat im_with_keypoints;
    drawKeypoints(src, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return im_with_keypoints;
    
}

Mat CornerDetection::detectDOG(Mat src, int thresh)
{
    Ptr<SiftFeatureDetector> SiftDetect= SiftFeatureDetector::create(0,3,0.15,10,1.6);
    Mat gray;
    vector<KeyPoint> keypoints;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    SiftDetect->detect(gray, keypoints);
    drawKeypoints(src, keypoints, src, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return src;
    ////--------------------------------------
    //Mat low_sigma, high_sigma;
    //Mat  dst,normal_dst;
    //int gammaInt = 5;
   
    ////grayscale
    //
    //cvtColor(src, gray, COLOR_BGR2GRAY);
    //
    ////Low gaussian blur và High gaussian blur
    //GaussianBlur(gray,low_sigma, Size(3,3), 0,0);
    //GaussianBlur(gray, high_sigma,Size(5,5), 0,0);

    ////DoG = Lowgausian - highgaussian
    //subtract(high_sigma, low_sigma, dst);
    ////Nhân gamma vào mỗi phần tử của ma trận để tăng khoảng cách giữa các điểm edge và không phải keypoint
    //dst *= gammaInt;
    //// Normalizing
    //normalize(dst, normal_dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
  
    //// Drawing a circle around corners   
    //for (int j = 0; j < normal_dst.rows; j++)
    //{
    //    for (int i = 0; i < normal_dst.cols; i++)
    //    {
    //        if (normal_dst.at<float>(j, i) > thresh)
    //        {
    //            circle(src, Point(i, j),10, Scalar(0, 0, 255), 2, 8, 0);
    //        }
    //    }
    //}
    //imshow("DoGDetection", src);
    //waitKey(0);
    //return normal_dst;
}
vector<KeyPoint>CornerDetection::detectKeypoint(Mat img,int detector, int thresh)
{
    vector<KeyPoint> keypoints;
    Mat dst;
    if (detector == 1)//Harris
    {
        dst = detectCornerHarris(img, thresh);
        for (int j = 0; j < dst.rows; j++)
        {
            for (int i = 0; i < dst.cols; i++)
            {

                if ((int)dst.at<float>(j, i) > thresh)
                {
                    keypoints.push_back(KeyPoint(Point(i, j), 1));      
                }
            }
        }
    }
    else if (detector == 3)//Blob
    {
        dst = detectBlob(img);
        keypoints = this->keypoints_container;
    }
    else if (detector == 4)//DoG
    {
        dst= detectDOG(img, thresh);
        for (int j = 0; j < dst.rows; j++)
        {
            for (int i = 0; i < dst.cols; i++)
            {

                if ((int)dst.at<float>(j, i) > thresh)
                {
                    keypoints.push_back(KeyPoint(Point(i, j), 1));
                }
            }
        }
    }
    return keypoints;
}
Mat CornerDetection::matchBySIFT(Mat img1, Mat img2, int detector, int thresh)
{
    
    Mat grey1, grey2;
    cvtColor(img1, grey1, COLOR_BGR2GRAY);
    cvtColor(img2, grey2, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints_1, keypoints_2;
   
    //Detect Keypoint
    keypoints_1 = detectKeypoint(img1, detector, thresh);
    keypoints_2 = detectKeypoint(img2, detector, thresh);
  
    //Vẽ Keypoint
    Mat im_with_keypoints_1,  im_with_keypoints_2;
    drawKeypoints(img1, keypoints_1, im_with_keypoints_1, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img2, keypoints_2, im_with_keypoints_2, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);

    //Tính descriptors (feature vectors) bằng Sift  
    Ptr<SiftDescriptorExtractor> SiftDetect = SiftFeatureDetector::create();

    Mat descriptors_1, descriptors_2;
    SiftDetect->compute(grey1, keypoints_1, descriptors_1);
    SiftDetect->compute(grey2, keypoints_2, descriptors_2);

    //KNN matching
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors_1, descriptors_2, knn_matches,2);
    //Lấy các matches có giá trị tốt
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); ++i)
    {
        const float ratio = 0.75; // As in Lowe's paper; can be tuned
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //Vẽ các keypoint matches
    Mat result;
    drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, result);
    resize(result, result, Size(1600, 800));
    return result;
}