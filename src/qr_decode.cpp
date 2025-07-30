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

#include <opencv2/objdetect.hpp>

void decode_generate_qr() {
  std::string path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qrcode.png";
  cv::Mat srcimage = cv::imread(path, cv::IMREAD_COLOR);
  if (srcimage.empty()) {
      std::cout << "image is empty" << std::endl;
      return;
  }

  cv::Mat resize_image;    
  cv::resize(srcimage, resize_image, cv::Size(500, 500));

  cv::Mat grayImage;
  cv::cvtColor(resize_image, grayImage, cv::COLOR_BGR2GRAY);

  cv::QRCodeDetector qrDecoder = cv::QRCodeDetector();
  std::vector<cv::Point> get_points;
  bool ret = qrDecoder.detect(grayImage, get_points);
  if(ret) {
      std::cout << "qrDecoder.detect() success" << std::endl;
      std::cout << "get_points.size() = " << get_points.size() << std::endl;
      for(int i = 0; i < get_points.size(); i++) {
          std::cout << get_points.at(i).x << " " << get_points.at(i).y << std::endl;
      }
  } else {
      std::cout << "qrDecoder.detect() failed" << std::endl;
      return;
  }

  cv::Mat rectifiedImage;
  std::string data = qrDecoder.decode(grayImage, get_points, rectifiedImage);
  if (data.length() > 0)
  {
      std::cout << "Decoded Data : " << data << std::endl;
      if (!rectifiedImage.empty()) {
          cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_rectified_image.png", rectifiedImage);
      }
  } else {
      std::cout << "QR Code not detected" << std::endl;
  }
}

void decode_random_qr() {
  std::string path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/qr.jpg";
  cv::Mat srcimage = cv::imread(path, cv::IMREAD_COLOR);
  if (srcimage.empty()) {
      std::cout << "image is empty" << std::endl;
      return;
  }
  std::cout << "srcimage.size() = " << srcimage.size() << std::endl;

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
  cv::cvtColor(contrastImage, grayImage, cv::COLOR_BGR2GRAY);

  cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_gray_image.png", grayImage);

  cv::QRCodeDetector qrDecoder = cv::QRCodeDetector();
  std::vector<cv::Point> get_points;
  cv::Mat rectifiedImage;
  std::string data = qrDecoder.detectAndDecode(srcimage, get_points, rectifiedImage);
  if (data.length() > 0)
  {
      std::cout << "Decoded Data : " << data << std::endl;
      if (!rectifiedImage.empty()) {
          cv::imwrite("/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qr_rectified_image_1.png", rectifiedImage);
      }
  } else {
      std::cout << "QR Code not detected" << std::endl;
  }

}

int main(int argc, char** argv) {
  std::cout << "===================== decode_generate_qr =====================" << std::endl;
  decode_generate_qr();
  std::cout << "===================== decode_random_qr =====================" << std::endl;
  decode_random_qr();

  return 0;

}