#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void subtract_images(cv::Mat img1, cv::Mat img2, cv::Mat& output)
{

    cv::subtract(img1, img2, output);
    imshow("Subtracted Image", output);
    waitKey(0);
}