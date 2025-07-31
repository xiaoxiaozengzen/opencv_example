#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

const std::string image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/";
const std::string output_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/";

int main() {
  std::string image_name = "bicycle.jpg";
  std::string image_file = image_path + image_name;

  cv::Mat img = cv::imread(image_file, cv::IMREAD_GRAYSCALE);
  cv::cuda::GpuMat gpu_img, gpu_result;
  gpu_img.upload(img);

  // 高斯模糊
  cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(gpu_img.type(), gpu_result.type(), cv::Size(5,5), 1.5);
  gauss->apply(gpu_img, gpu_result);

  // Canny 边缘检测
  cv::cuda::GpuMat edges;
  cv::Ptr<cv::cuda::CannyEdgeDetector> cannyFilter = cv::cuda::createCannyEdgeDetector(50, 150);
  cannyFilter->detect(gpu_result, edges);

  cv::Mat result;
  edges.download(result);
  std::string output_image = output_path + "edges.jpg";
  cv::imwrite(output_image, result);
}