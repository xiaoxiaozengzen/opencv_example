#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

int main() {
  cv::Mat img = cv::imread("test.jpg", cv::IMREAD_GRAYSCALE);
  cv::cuda::GpuMat gpu_img, gpu_result;
  gpu_img.upload(img);

  // 高斯模糊
  cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(gpu_img.type(), gpu_result.type(), cv::Size(5,5), 1.5);
  gauss->apply(gpu_img, gpu_result);

  // Canny 边缘检测
  cv::cuda::GpuMat edges;
  cv::cuda::Canny(gpu_img, edges, 50, 150);

  cv::Mat result;
  edges.download(result);
  cv::imwrite("edges.jpg", result);
}