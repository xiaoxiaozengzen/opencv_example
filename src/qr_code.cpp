#include <string> // 字符串
#include <vector> // 向量
#include <iostream> // 输入输出流
#include <set> // 集合

#include <opencv2/core.hpp> // OpenCV核心功能
#include <opencv2/imgcodecs.hpp> // 图像编解码
#include <opencv2/highgui.hpp> // GUI
#include <opencv2/imgproc.hpp> // 图像处理
#include <opencv2/opencv.hpp> // OpenCV主要功能
#include <opencv2/imgproc/types_c.h> // 旧版的图像处理


void PrintLine(cv::Mat &image, std::vector<cv::Vec2f> &lines) {
    for (size_t i = 0; i < lines.size(); i++) {
        // 直线的极坐标参数
        double rho = lines[i][0];  // 距离原点的距离
        double theta = lines[i][1];  // 与 x 轴的夹角

        // 计算直线上的两个点
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        // 绘制直线到图像上
        cv::line(image, pt1, pt2, cv::Scalar(0, 0, 0), 2);
    }
}


cv::RNG rng(12345);
struct index_ {
    int a1;
    int a2;
    int a3;
};
index_ in;
std::vector<index_> vin;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string path = argv[1];
    cv::Mat srcimage = cv::imread(path, cv::IMREAD_COLOR);
    if (srcimage.empty()) {
        std::cout << "image is empty" << std::endl;
        return -1;
    }
    // srcimage.resize(450);

    cv::Mat image;    
    cv::resize(srcimage, image, cv::Size(500, 500));

    cv::Mat contrastImage = cv::Mat::zeros(image.size(), image.type());
    double alpha = 1.8;  // 对比度控制
    int beta = -30;   // 亮度控制
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int c = 0; c < 3; c++) {
                contrastImage.at<cv::Vec3b>(y, x)[c] =
                    cv::saturate_cast<uchar>(alpha * image.at<cv::Vec3b>(y, x)[c] + beta);
            }
        }
    }

    cv::imwrite("../image.png", image);
    cv::imwrite("../contrastImage.png", contrastImage);

    cv::Mat grayImage;
    cv::Mat filterImage;
    cv::Mat binaryImage;
    cv::cvtColor(contrastImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::bilateralFilter(grayImage, filterImage, 13, 26, 6);
    cv::threshold(filterImage, binaryImage, 210, 255, cv::THRESH_BINARY);

    cv::imwrite("../grayImage.png", grayImage);
    cv::imwrite("../filterImage.png", filterImage);
    cv::imwrite("../binaryImage.png", binaryImage);

    // cv::Mat cannyImage;
    // cv::Canny(binaryImage, cannyImage, 10, 100, 3, false);
    // cv::imwrite("../cannyImage.png", cannyImage);

    // std::vector<cv::Vec2f> resLines;
    // cv::HoughLines(cannyImage, resLines, 5, CV_PI / 180, 100);

    // if (resLines.size() < 4) {
    //     std::cout << "resLines.size() < 4" << std::endl;
    //     return -1;
    // } else {
    //     std::cout << "resLines.size() = " << resLines.size() << std::endl;
    // }

    // double A = 50.0;
    // double B = CV_PI / 180 * 20;
    // std::set<size_t> removeIndex;
    // while (1) {
    //     for (size_t i = 0; i < resLines.size(); i++) {
    //         for (size_t j = i + 1; j < resLines.size(); j++) {
    //             // 直线参数
    //             float rho1 = resLines[i][0], theta1 = resLines[i][1];
    //             float rho2 = resLines[j][0], theta2 = resLines[j][1];

    //             // 角度调整，确保比较的一致性
    //             if (theta1 > CV_PI) theta1 -= CV_PI;
    //             if (theta2 > CV_PI) theta2 -= CV_PI;

    //             // 判断是否为相似直线并标记删除
    //             bool thetaFlag = std::abs(theta1 - theta2) <= B ||
    //                             (theta1 > CV_PI / 2 && theta2 < CV_PI / 2 && CV_PI - theta1 + theta2 < B) ||
    //                             (theta2 > CV_PI / 2 && theta1 < CV_PI / 2 && CV_PI - theta2 + theta1 < B);
    //             if (std::abs(rho1 - rho2) <= A && thetaFlag) {
    //                 removeIndex.insert(j);
    //             }
    //         }
    //     }

    //     // 删除标记的直线
    //     std::vector<cv::Vec2f> newLines;
    //     for (size_t i = 0; i < resLines.size(); i++) {
    //         if (removeIndex.find(i) == removeIndex.end()) {
    //             newLines.push_back(resLines[i]);
    //         }
    //     }
    //     resLines = newLines;

    //     // 直线数量达到目标值则终止循环
    //     if (resLines.size() == 4) break;

    //     if (resLines.size() < 4) {
    //         std::cout << "resLines.size() < 4" << std::endl;
    //         return -1;
    //     }
    // }

    // cv::Mat cannyImageLine = cv::Mat(cannyImage.size(), cannyImage.type(), cv::Scalar(255, 255, 255));
    // PrintLine(cannyImageLine, resLines);
    // cv::imwrite("../cannyImageLine.png", cannyImageLine);

    // double threshold = 0.2 * std::min(image.rows, image.cols);
    // std::vector<cv::Point> points;
    // for (int i = 0; i < resLines.size(); i++) {
    //     for (int j = i + 1; j < resLines.size(); j++) {
    //         // 提取直线参数
    //         double rho1 = resLines[i][0], theta1 = resLines[i][1];
    //         double rho2 = resLines[j][0], theta2 = resLines[j][1];

    //         // 调整θ值以处理斜率无穷大的情况
    //         if (theta1 == 0) theta1 = 0.01;
    //         if (theta2 == 0) theta2 = 0.01;

    //         // 计算直线交点
    //         double a1 = cos(theta1), a2 = cos(theta2);
    //         double b1 = sin(theta1), b2 = sin(theta2);
    //         double x = (rho2 * b1 - rho1 * b2) / (a2 * b1 - a1 * b2);
    //         double y = (rho1 - a1 * x) / b1;

    //         // 保证交点在图像范围内
    //         cv::Point pt(cvRound(x), cvRound(y));
    //         if (pt.x <= image.cols + threshold && pt.x >= -threshold &&
    //             pt.y < image.rows + threshold && pt.y >= -threshold) {
    //             points.push_back(pt);
    //         }
    //     }
    // }

    // cv::Point2f srcTri[4];
    // cv::Point2f dstTri[4];
    // // 假设srcTri已经根据points数组正确赋值
    // // dstTri是变换后希望得到的正方形坐标
    // for(int i = 0; i < 4; i++) {
    //     srcTri[i] = points.at(i);
    //     std::cout << points.at(i).x << " " << points.at(i).y << std::endl;
    // }
    // dstTri[0] = cv::Point2f(330,450);
    // dstTri[1] = cv::Point2f(180,450);
    // dstTri[2] = cv::Point2f(330,330);
    // dstTri[3] = cv::Point2f(180,330);

    // cv::Mat perspImage = cannyImage.clone();
    // cv::Mat transmtx = cv::getPerspectiveTransform(srcTri, dstTri);
    // std::cout << transmtx << std::endl;
    // cv::warpPerspective(cannyImage, perspImage, transmtx, cv::Size(600, 480));
    // cv::imwrite("../perspImage.png", perspImage);

    // std::vector<std::vector<cv::Point>> contours;
    // std::vector<cv::Vec4i> hierarchy;
    // cv::findContours(perspImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // cv::Mat contoursImage = perspImage.clone();
    // cv::drawContours(contoursImage, contours, -1, cv::Scalar(255, 255, 255), 2);
    // cv::imwrite("../contoursImage.png", contoursImage);


    cv::QRCodeDetector qrDecoder = cv::QRCodeDetector();
    std::vector<cv::Point> get_points;
    bool ret = qrDecoder.detect(binaryImage, get_points);
    if(ret) {
        std::cout << "qrDecoder.detect() success" << std::endl;
        std::cout << "get_points.size() = " << get_points.size() << std::endl;
        for(int i = 0; i < get_points.size(); i++) {
            std::cout << get_points.at(i).x << " " << get_points.at(i).y << std::endl;
        }
    } else {
        std::cout << "qrDecoder.detect() failed" << std::endl;
        return -1;
    }

    cv::Mat rectifiedImage;
    std::string data = qrDecoder.decode(binaryImage, get_points, rectifiedImage);
    if (data.length() > 0)
    {
        std::cout << "Decoded Data : " << data << std::endl;
        if (!rectifiedImage.empty()) {
            cv::imwrite("../rectifiedImage.png", rectifiedImage);
        }
    }
    else
    {
        std::cout << "QR Code not detected" << std::endl;
    }

    return 0;

}