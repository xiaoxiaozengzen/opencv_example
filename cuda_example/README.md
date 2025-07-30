# Overview

如果要使用cv的cuda库，会涉及到数据从cpu和gpu之间的交换。
* 一张图片首先会被cpu读取到内存中，
* 然后通过api将cpu中的数据搬运到gpu中，而cpu和gpu之间的数据搬运也是很耗时的，比如gpu_dst.download(dst_cpu)将gpu_dst数据搬运到dst_cpu，数据是8976*4960*3，耗时约37ms，如果你的图像处理比较简单，说不定数据搬运的耗时比直接在cpu上运行更长。

# 相关API
OpenCV 中与 CUDA 相关的接口主要集中在 cv::cuda 命名空间（C++）或 cv2.cuda 模块（Python），这些接口可以直接调用 GPU 加速的图像处理和计算功能。常用的 CUDA 相关接口包括：
* cv::cuda::GpuMat: GPU 上的矩阵（图像/数据）对象，类似于 cv::Mat，但数据存储在显存中。
* 基础运算：
  * cv::cuda::add
  * cv::cuda::subtract
* 图像处理：
  * cv::cuda::resize
  * cv::cuda::cvtColor
* 特征检测与描述
  * cv::cuda::Canny
* 几何变换
  * cv::cuda::warpAffine
* 模板匹配：
  * cv::cuda::matchTemplate
* 背景建模
  * cv::cuda::createBackgroundSubtractorMOG2
* 视频编解码：
  * cv::cudacodec::createVideoReader
  * cv::cudacodec::createVideoWriter

# 相关模块

其中opencv_contrib中包含了许多cuda相关的组件：
* cudaarithm：功能：基础的矩阵运算（加减乘除、逻辑运算、缩放、归一化等）。
* cudafilters：各种滤波操作（如高斯、均值、中值、Sobel、Laplacian 等）
* cudawarping：几何变换（如仿射变换、透视变换、重映射）
* cudacodec：视频编解码（GPU加速的视频解码、编码）
* cudaimgproc：图像处理（颜色空间转换、直方图均衡、阈值、形态学操作等）。
* cudafeatures2d：特征点检测与描述（如 SURF、ORB、FAST 等）
* cudabgsegm：背景建模与前景分割（如 MOG、GMG 等）。
* cudaobjdetect：背景建模与前景分割（如 MOG、GMG 等）
* cudastereo：立体匹配（如 BM、SGBM 算法）。