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

// #include <opencv2/opencv.hpp>
// #include <opencv2/objdetect.hpp>

int main(int argc, char** argv) {

    std::string path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/qr.png";
    cv::Mat srcimage = cv::imread(path, cv::IMREAD_COLOR);
    if (srcimage.empty()) {
        std::cout << "image is empty" << std::endl;
        return -1;
    }

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

    cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_resize_image.png", image);
    cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_contrast_image.png", contrastImage);

    cv::Mat grayImage;
    cv::Mat filterImage;
    cv::Mat binaryImage;
    cv::cvtColor(contrastImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::bilateralFilter(grayImage, filterImage, 13, 26, 6);
    cv::threshold(filterImage, binaryImage, 210, 255, cv::THRESH_BINARY);

    cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_gray_image.png", grayImage);
    cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_filter_image.png", filterImage);
    cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_binary_image.png", binaryImage);

    // cv::QRCodeDetector qrDecoder = cv::QRCodeDetector();
    // std::vector<cv::Point> get_points;
    // bool ret = qrDecoder.detect(binaryImage, get_points);
    // if(ret) {
    //     std::cout << "qrDecoder.detect() success" << std::endl;
    //     std::cout << "get_points.size() = " << get_points.size() << std::endl;
    //     for(int i = 0; i < get_points.size(); i++) {
    //         std::cout << get_points.at(i).x << " " << get_points.at(i).y << std::endl;
    //     }
    // } else {
    //     std::cout << "qrDecoder.detect() failed" << std::endl;
    //     return -1;
    // }

    // cv::Mat rectifiedImage;
    // std::string data = qrDecoder.decode(binaryImage, get_points, rectifiedImage);
    // if (data.length() > 0)
    // {
    //     std::cout << "Decoded Data : " << data << std::endl;
    //     if (!rectifiedImage.empty()) {
    //         cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_rectified_image.png", rectifiedImage);
    //     }
    // } else {
    //     std::cout << "QR Code not detected" << std::endl;
    // }

    return 0;

}