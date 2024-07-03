#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;


void subtract_images(cv::Mat img1, cv::Mat img2, cv::Mat& output)
{

    cv::subtract(img1, img2, output);

    waitKey(0);
}

int main() {
    // Đọc vào 2 ảnh cần so sánh
    cv::Mat img1 = cv::imread("500kmau.jpg");
    cv::Mat img2 = cv::imread("500kzz1.jpg");
    cv::Mat img3 = cv::imread("500k2.jpg");

    cv::Size size(550, 308);  // Kích cỡ mới
    cv::resize(img1, img1, size);
    cv::resize(img2, img2, size);

    // Khởi tạo bộ trích xuất đặc trưng và bộ khớp từ khóa
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // Tìm kiếm các đặc trưng của 2 ảnh và khớp chúng với nhau
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Lọc các khớp không chính xác
    double max_dist = 0;
    double min_dist = 100;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance <= 2 * min_dist) {
            good_matches.push_back(matches[i]);
        }
    }


    // Tính toán ma trận chuyển đổi giữa 2 ảnh bằng thuật toán RANSAC
    std::vector<cv::Point2f> pts1, pts2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC);

    // Biến đổi ảnh thứ hai để có cùng góc nhìn với ảnh thứ nhất
    cv::Mat img_warped;
    cv::warpPerspective(img2, img_warped, H, img1.size());


    // Hiển thị kết quả
   //cv::cvtColor(img_warped, img_warped, cv::COLOR_BGR2GRAY);
   //cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);

    cv::Mat output = cv::Mat::zeros(img1.size(), img1.type());
    subtract_images(img3, img_warped, output);

    imshow("Subtracted Image", output);
    imshow("anh tien mau", img1);
    imshow("anh tien", img_warped);
    
    
    waitKey(0);

    return 0;
}
