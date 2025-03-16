#include <string> // 字符串
#include <vector> // 向量

#include <opencv2/core.hpp> // OpenCV核心功能
#include <opencv2/imgcodecs.hpp> // 图像编解码
#include <opencv2/highgui.hpp> // GUI
#include <opencv2/imgproc.hpp> // 图像处理
#include <iostream> // 输入输出流
#include <set> // 集合
#include <opencv2/opencv.hpp> // OpenCV主要功能
#include <opencv2/imgproc/types_c.h> // 旧版的图像处理

void ConFun() {
#if 0
  cv::VideoWriter writer;
#endif

  std::string video_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/output.264";

  /**
   * @brief 创建一个VideoWriter对象
   * @param filname 视频文件路径，
   * @param fourcc('M', 'J', 'P', 'G') 视频编码格式
   * @param fps 视频帧率
   * @param frameSize 图像大小
   * @param isColor 是否为彩色图像，默认为true。当为false时，图像为灰度图像
   * 
   * @note fourcc参数为视频编码格式，常用的有：
   * - 'H264'	H.264 video.
   * - 'I420'	YUV video stored in planar 4:2:0 format.
   * - 'IYUV'	YUV video stored in planar 4:2:0 format.
   * - 'M4S2'	MPEG-4 part 2 video.
   * - 'MP4S'	Microsoft MPEG 4 codec version 3. This codec is no longer supported.
   * - 'MP4V'	MPEG-4 part 2 video.
   * - 'MPG1'	MPEG-1 video.
   * - 'MSS1'	Content encoded with the Windows Media Video 7 screen codec.
   * - 'MSS2'	Content encoded with the Windows Media Video 9 screen codec.
   * - 'UYVY'	YUV video stored in packed 4:2:2 format. Similar to YUY2 but with different ordering of data.
   * - 'WMV1'	Content encoded with the Windows Media Video 7 codec.
   * - 'WMV2'	Content encoded with the Windows Media Video 8 codec.
   * - 'WMV3'	Content encoded with the Windows Media Video 9 codec.
   * - 'WMVA'	Content encoded with the older, obsolete version of the Windows Media Video 9 Advanced Profile codec.
   * - 'WMVP'	Content encoded with the Windows Media Video 9.1 Image codec.
   * - 'WVC1'	SMPTE 421M ("VC-1"). Content encoded with Windows Media Video 9 Advanced Profile.
   * - 'WVP2'	Content encoded with the Windows Media Video 9.1 Image v2 codec.
   * - 'YUY2'	YUV video stored in packed 4:2:2 format.
   * - 'YV12'	YUV video stored in planar 4:2:0 or 4:1:1 format. Identical to I420/IYUV except that the U and V planes are switched.
   * - 'YVU9'	YUV video stored in planar 16:1:1 format.
   * - 'YVYU'	YUV video stored in packed 4:2:2 format. Similar to YUY2 but with different ordering of data.
   */
  cv::VideoWriter writer1(
      video_path,
      cv::VideoWriter::fourcc('H', '2', '6', '4'),
      30,
      cv::Size(640, 480),
      true
  );

#if 0
  /**
   * @brief 创建一个VideoWriter对象
   * @param filname 视频文件路径
   * @param apiPreference 底层使用的api，默认为cv::CAP_ANY
   * @param fourcc('M', 'J', 'P', 'G') 视频编码格式
   * @param fps 视频帧率
   * @param frameSize 图像大小
   * @param isColor 是否为彩色图像，默认为true。当为false时，图像为灰度图像
   */
  cv::VideoWriter writer2(
      video_path,
      cv::CAP_FFMPEG,
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
      30,
      cv::Size(640, 480),
      false
  );
#endif

#if 0
  /**
   * @brief 创建一个VideoWriter对象
   * @param filname 视频文件路径
   * @param fourcc('M', 'J', 'P', 'G') 视频编码格式
   * @param fps 视频帧率
   * @param frameSize 图像大小
   * @param params 视频编码参数
   */
  cv::VideoWriter writer3(
      video_path,
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
      30,
      cv::Size(640, 480),
      std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 100}
  );
#endif
}

void MemFun() {
  // 文件拓展名加数字？直接output.avi会报一些异常错误
  std::string video_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/output_1.avi";
  std::string image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/660.jpg";
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if(image.empty()) {
    std::cerr << "Read image failed." << std::endl;
    return;
  }

  int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  double fps = 10.0;

  cv::VideoWriter writer;
  writer.open(
      video_path,
      codec,
      fps,
      image.size(),
      true
  );
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;
  writer << image;


  cv::Mat image1 = image.clone();
  cv::line(image1, cv::Point(0, 0), cv::Point(640, 480), cv::Scalar(0, 0, 255), 2);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);
  writer.write(image1);

  bool is_open = writer.isOpened();
  if(!is_open) {
    std::cerr << "Open video writer failed." << std::endl;
    return;
  } else {
    std::cout << "Open video writer success." << std::endl;
  }

  std::string backen_name = writer.getBackendName();
  std::cout << "VideoWriter backend name: " << backen_name << std::endl;

  double pro = writer.get(cv::VIDEOWRITER_PROP_QUALITY);
  std::cout << "VideoWriter quality: " << pro << std::endl;

  writer.release();
  is_open = writer.isOpened();
  if(!is_open) {
    std::cout << "Release video writer success." << std::endl;
  } else {
    std::cerr << "Release video writer failed." << std::endl;
  }
}

int main() {
  std::cout << "====================== ConFun ======================" << std::endl;
  ConFun();
  std::cout << "====================== MemFun ======================" << std::endl;
  MemFun();

  return 0;
}