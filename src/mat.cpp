#include <string> // 字符串
#include <vector> // 向量
#include <fstream>
#include <sstream>

#include <opencv2/core.hpp> // OpenCV核心功能
#include <opencv2/imgcodecs.hpp> // 图像编解码
#include <opencv2/highgui.hpp> // GUI
#include <opencv2/imgproc.hpp> // 图像处理
#include <iostream> // 输入输出流
#include <set> // 集合
#include <opencv2/opencv.hpp> // OpenCV主要功能
#include <opencv2/imgproc/types_c.h> // 旧版的图像处理

void ConFun() {
  cv::Mat mat; // 创建一个空的Mat对象

  cv::Mat mat1(2, 3, CV_8UC3); // 创建一个rows=2, cols=3, type=CV_8UC3类型的Mat对象
  std::cout << "mat1: \n" << mat1 << std::endl;

  cv::Size size(3, 2); // 创建一个Size对象, width=3, height=2
  cv::Mat mat2(size, CV_8UC3); // 创建一个rows=2, cols=3, type=CV_8UC3类型的Mat对象
  std::cout << "mat2: \n" << mat2 << std::endl;

  cv::Scalar scalar(0, 0, 255); // 创建一个Scalar对象, B=0, G=0, R=255
  cv::Mat mat3(2, 3, CV_8UC3, scalar); // 创建一个rows=2, cols=3, type=CV_8UC3类型的Mat对象, 并初始化所有元素为scalar
  std::cout << "mat3: \n" << mat3 << std::endl;

  cv::Mat mat4 = cv::Mat(size, CV_8UC3, scalar); // 创建一个rows=2, cols=3, type=CV_8UC3类型的Mat对象, 并初始化所有元素为scalar
  std::cout << "mat4: \n" << mat4 << std::endl;

  const int sizes[] = {4, 5}; // 创建一个int数组，数组元素大小要跟维度数量一致，例如二维数组，数组元素大小为2；三维数组，数组元素大小为3
  cv::Mat mat5 = cv::Mat(2, sizes, CV_8UC3); // 创建一个2维的Mat对象, 维度大小为4行5列(宽为5，高为4), type=CV_8UC3
  std::cout << "mat5: \n" << mat5 << std::endl;
  std::cout << "mat5 size: " << mat5.size() << std::endl; // 输出mat的size

  cv::Mat mat6 = cv::Mat(2, sizes, CV_8UC4, cv::Scalar{1, 2, 3, 4}); // 创建一个2维的Mat对象, 维度大小为2行3列, type=CV_8UC3, 并初始化所有元素为scalar
  std::cout << "mat6: \n" << mat6 << std::endl;

  cv::Rect rect(1, 2, 3, 2); // 创建一个Rect对象，x=1, y=2, width=3, height=2
  cv::Mat mat7 = cv::Mat(mat6, rect); // 创建一个mat6的子矩阵，即mat6的第2行第3列的元素
  std::cout << "mat7: \n" << mat7
            << ", mat7 size: " << mat7.size()
            << std::endl;

  cv::Mat mat8 = cv::Mat(std::vector<int>{2, 3}, CV_8UC1); // 创建一个2维的Mat对象, 维度大小为2行3列, type=CV_8UC3
  std::cout << "mat8 size: " << mat8.size()
            << ", mat8 channels: " << mat8.channels()
            << std::endl;

  cv::Mat mat9 = cv::Mat(std::vector<int>{3, 4}, CV_8UC1, cv::Scalar(22)); // 创建一个2维的Mat对象, 维度大小为3行4列, type=CV_8UC1, 并初始化所有元素为22
  std::cout << "mat9: \n" << mat9
            << ", mat9 size: " << mat9.size()
            << ", mat9 channels: " << mat9.channels()
            << std::endl;
  
  // 创建一个uchar数组，必须用uchar类型，因为Mat的数据类型是uchar。并且是按照行优先的方式存储的。
  uchar data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; 
  cv::Mat mat10 = cv::Mat(3, 4, CV_8UC1, static_cast<void*>(data)); // 创建一个3行4列的CV_8UC1类型的Mat对象, 并初始化所有元素为data。
  std::cout << "mat10: \n" << mat10
            << ", mat10 size: " << mat10.size()
            << ", mat10 channels: " << mat10.channels()
            << std::endl;
  
  uchar data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                };
  std::size_t step = 8; // 每行的bit数，等价于elementSize * cols = step
  cv::Mat mat11 = cv::Mat(3, 4, CV_8UC2, static_cast<void*>(data1), step); // 创建一个2行3列的CV_8UC2类型的Mat对象, 并初始化所有元素为data1。
  std::cout << "mat11: \n" << mat11
            << ", mat11 size: " << mat11.size()
            << ", mat11 channels: " << mat11.channels()
            << std::endl;

  cv::Mat mat12 = cv::Mat(std::vector<int>{3, 4}, CV_8UC1, static_cast<void*>(data1)); // 创建一个2维的Mat对象, 维度大小为3行4列, type=CV_8UC1, 并初始化所有元素为data1
  std::cout << "mat12: \n" << mat12
            << ", mat12 size: " << mat12.size()
            << ", mat12 channels: " << mat12.channels()
            << std::endl;
}

void PubMem() {
  cv::Mat mat = cv::Mat(3, 4, CV_8UC3, cv::Scalar(0, 0, 255)); // 创建一个rows=3, cols=4, type=CV_8UC3类型的Mat对象

  cv::MatSize image_size = mat.size; // 获取mat的size
  std::cout << "image size: " << image_size << std::endl;
  cv::Size image_size1 = image_size();
  std::cout << "image size1: " << image_size1 << std::endl;
  {
    // cv::Size的一些常用函数
    int width = image_size1.width; // 获取mat的宽度
    std::cout << "cv::Size width: " << width << std::endl;
    int height = image_size1.height; // 获取mat的高度
    std::cout << "cv::Size height: " << height << std::endl;
    double area = image_size1.area(); // 获取mat的面积
    std::cout << "cv::Size area: " << area << std::endl;
    double aspectRatio = image_size1.aspectRatio(); // 获取mat的宽高比
    std::cout << "cv::Size aspectRatio: " << aspectRatio << std::endl;
    bool empty = image_size1.empty(); // 判断mat是否为空
    std::cout << "cv::Size empty: " << empty << std::endl;
  }
  int dims = image_size.dims(); // 获取mat的维度
  std::cout << "image dims: " << dims << std::endl;

  int image_cols = mat.cols; // 获取mat的列数
  std::cout << "image cols: " << image_cols << std::endl;
  int image_rows = mat.rows; // 获取mat的行数
  std::cout << "image rows: " << image_rows << std::endl;
  int image_dims1 = mat.dims; // 获取mat的维度
  std::cout << "image dims1: " << image_dims1 << std::endl;
  // flags表示几个位掩码，用于描述矩阵的属性，如depth, channel, continous, submatrix等
  int flags = mat.flags; // 获取mat的flags
  std::cout << "image flags: " << flags << std::endl;
  std::cout << "CV_8UC3: " << CV_8UC3 << std::endl;
  // 每行的bit数
  cv::MatStep step = mat.step; // 获取mat的step
  std::cout << "image step: " << step << std::endl; // width * channels * sizeof(unit)
  {
    // cv::MatStep的一些常用函数
    std::size_t (&buff)[2] = step.buf; // 获取mat的buf, buf[0]表示每行的bit数，buf[1]表示每个元素的bit数
    std::cout << "buff[0]: " << buff[0] << ", buff[1]: " << buff[1] << std::endl;
  }

  // Mat指向的数据的指针
  uchar * data = mat.data; // 获取mat的data
  const uchar* dataend = mat.dataend; // 获取mat的dataend
  const uchar* datastart = mat.datastart; // 获取mat的datastart
  std::ptrdiff_t diff = dataend - datastart;
  std::cout << "diff: " << diff << std::endl; // width * height * channels * sizeof(unit)
}

void MemFun() {
  cv::Mat mat = cv::Mat(3, 4, CV_16UC3, cv::Scalar(0, 0, 255)); // 创建一个rows=3, cols=4, type=CV_8UC3类型的Mat对象
  std::cout << "mat: \n" << mat << std::endl;

  int channels = mat.channels(); // 获取mat的通道数
  std::cout << "mat channels: " << channels << std::endl;
  double value3 = mat.at<cv::Vec<std::uint16_t, 3>>(0, 2)[2]; // 获取mat的第0行第2列的第2个通道的值，其中Vec的第一个模板参数是数据类型，第二个模板参数是通道数
  double value2 = mat.at<cv::Vec<std::uint16_t, 3>>(0, 2)[1]; // 获取mat的第0行第2列的第1个通道的值
  double value1 = mat.at<cv::Vec<std::uint16_t, 3>>(0, 2)[0]; // 获取mat的第0行第2列的第0个通道的值
  std::cout << "mat at: (0, 2): " << value1 << ", " << value2 << ", " << value3 << std::endl;
  cv::Mat mat2 = mat.clone(); // 复制mat，深拷贝
  cv::Mat mat3(mat); // 复制mat, 浅拷贝
  std::cout << "mat: " << static_cast<void*>(mat.data)
            <<", mat2: " << static_cast<void*>(mat2.data)
            << ", mat3: " << static_cast<void*>(mat3.data)
            << std::endl;
  int type = mat.type(); // 获取mat的类型
  std::cout << "mat type: " << type << ", CV_8UC3: " << CV_8UC3 << ", CV_16UC3: " << CV_16UC3 << std::endl;
  std::size_t elemSize = mat.elemSize(); // 获取mat的每个元素的大小, channels * sizeof(unit)
  std::cout << "mat elemSize: " << elemSize << std::endl;
  std::size_t elemSize1 = mat.elemSize1(); // 获取mat的每个元素的大小, 忽略channel ,sizeof(unit)
  std::cout << "mat elemSize1: " << elemSize1 << std::endl;
  bool isContinuous = mat.isContinuous(); // 判断mat是否是连续的，可能因为ROI等原因导致不连续
  std::cout << "mat isContinuous: " << isContinuous << std::endl;
  std::size_t step1 = mat.step1(); // 获取mat的step1 = step / elemSize1, 快速访问矩阵任意元素
  std::cout << "mat step1: " << step1 << ", step: " << mat.step << std::endl;
  std::size_t total = mat.total(); // 获取mat的总元素数
  std::cout << "mat total: " << total << std::endl;
  cv::Mat mat4 = mat.col(1); // 获取mat的第2列
  std::cout << "mat: " << mat << std::endl;
  std::cout << "mat4: " << mat4 << std::endl;
  int depth = mat.depth(); // 获取mat的深度,单个通道的字节大小
  std::cout << "mat depth: " << depth << std::endl;
  {
    // cv::Range的一些常用函数
    cv::Range range(1, 4); // 创建一个Range对象，start=1, end=4, 即[1, 4)
    std::cout << "range: " << range << std::endl;
    std::size_t size = range.size(); // 获取range的大小
    std::cout << "range size: " << size << std::endl;
    int start = range.start; // 获取range的start
    std::cout << "range start: " << start << std::endl;
    int end = range.end; // 获取range的end
    std::cout << "range end: " << end << std::endl;
    bool empty = range.empty(); // 判断range是否为空
    std::cout << "range empty: " << empty << std::endl;
    cv::Range all = cv::Range::all(); // 获取一个全范围的Range对象,即[-2147483648, 2147483647)
    std::cout << "range all: " << all << std::endl;
  }
  cv::Mat mat5 = cv::Mat::eye(3, 3, CV_8UC1); // 创建一个3*3的单位矩阵
  std::cout << "mat5: \n" << mat5 << std::endl;
  cv::Mat mat6 = cv::Mat::ones(3, 3, CV_8UC1); // 创建一个3*3的全1矩阵
  std::cout << "mat6: \n" << mat6 << std::endl;
  cv::Mat mat7 = cv::Mat::zeros(3, 3, CV_8UC1); // 创建一个3*3的全0矩阵
  std::cout << "mat7: \n" << mat7 << std::endl;
  cv::Mat mat8 = cv::Mat::ones(cv::Size(3, 3), CV_8UC1); // 创建一个3*3的全1矩阵
  std::cout << "mat8: \n" << mat8 << std::endl;

  // cv::Mat的对图片的操作
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  std::cout << "image size: " << image.size()
            << ", image channels: " << image.channels()
            << ", image type: " << image.type()
            << ", image depth: " << image.depth()
            << ", image width: " << image.cols
            << ", image height: " << image.rows
            << std::endl;

  cv::Mat image_col = image.col(20); // 获取image的第20列
  cv::imwrite(root_path + "/output/image_col.png", image_col);

  cv::Mat image_colrange = image.colRange(0, 100); // 获取image的第0列到第100列,即[0, 100)
  cv::imwrite(root_path + "/output/image_colrange.png", image_colrange);
  std::cout << "image_colrange size: " << image_colrange.size() << std::endl;
  int depth1 = image_colrange.depth(); // 获取image_colrange的深度
  std::cout << "image_colrange depth: " << depth1 << std::endl;
  cv::Mat image_popback = image.clone();
  for (int i = 0; i < 600 ; i++) {
    image_popback.pop_back(); // 删除image_popback的最后一行
  }
  image_popback.pop_back(50); // 删除image_popback的最后50行，如果大于总行数，则抛出异常
  std::cout << "image_popback size: " << image_popback.size() << std::endl;
  cv::imwrite(root_path + "/output/image_popback.png", image_popback);
  cv::Mat image_resize = image.clone();
  image_resize.resize(2000); // 重新设置image_resize的行数为2000，如果小于原行数，则删除多余的行，如果大于原行数，则添加空行
  std::cout << "image_resize size: " << image_resize.size()
            << ", image_resize isContinuous: " << image_resize.isContinuous()
            << std::endl;
  cv::imwrite(root_path + "/output/image_resize.png", image_resize);
  cv::Mat image_reshape = image.reshape(1); // reshape只是重新解释当前存储数据的形状。将image_reshape转换为通道数为1的图像，然后导致原来列数*3
  std::cout << "image_reshape elemSize: " << image_reshape.elemSize() << std::endl;
  std::cout << "image_reshape channels: " << image_reshape.channels() << std::endl;
  std::cout << "image_reshape size: " << image_reshape.size()
            << ", image size: " << image.size()
            << std::endl;
  cv::imwrite(root_path + "/output/image_reshape.png", image_reshape);
  cv::Mat image_reshape_row = image.reshape(1, 3 * 1080); // 将image_reshape_row转换为通道数为1的图像，行数为3 * 1080
  std::cout << "image_reshape_row size: " << image_reshape_row.size()
            << ", image size: " << image.size()
            << ", image_reshape_row elemSize: " << image_reshape_row.elemSize()
            << ", image_reshape_row channels: " << image_reshape_row.channels()
            << ", image_reshape_row isContinuous: " << image_reshape_row.isContinuous()
            << std::endl;
  cv::imwrite(root_path + "/output/image_reshape_row.png", image_reshape_row);
  // create会重新分配内存，确保当前Mat有指定的尺寸和类型。如果当前Mat已经有指定的尺寸和类型，则不会重新分配内存。
  cv::Mat image_create = image.clone();
  image_create.create(image.rows - 500, image.cols - 500, image.type());
  std::cout << "image_create size: " << image_create.size()
            << ", isContinuous: " << image_create.isContinuous()
            << std::endl;
  cv::imwrite(root_path + "/output/image_create.png", image_create);
  cv::Mat image_convert = cv::Mat(image.size(), CV_8UC1); // 创建一个和image大小一样的CV_8UC1类型的Mat对象
  /**
   * @brief 更改图像的类型和数值
   * @param m 需要更改的图像
   * @param rtype 输出图像的深度和通道数
   * @param alpha 乘法因子
   * @param beta 加法因子
   * 
   * @note 不改变通道数，会改变深度
   * 
   * @note m(x, y) = saturate_cast<rType>(m(x, y) * alpha + beta)
   */
  image.convertTo(image_convert, CV_8UC1, 0.5, 100); // 将image转换为CV_8UC1类型
  std::cout << "image_convert size: " << image_convert.size()
            << ", image_convert channels: " << image_convert.channels()
            << std::endl;
  cv::imwrite(root_path + "/output/image_convert.png", image_convert);
  cv::Mat image_convert_depth = cv::Mat(image.size(), CV_16UC3); // 创建一个和image大小一样的CV_16UC3类型的Mat对象
  image.convertTo(image_convert_depth, CV_16UC3, 256); // 将image转换为CV_16UC3类型，通道值也要做相应的映射，可以保证图片不变：65535/255=256
  cv::imwrite(root_path + "/output/image_convert_depth.png", image_convert_depth);
}

void GlobalFun() {
  // cv::Mat的对图片的操作
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  /**********************************cv::remap*********************************************/
  std::cout << "/*************cv::remap***************/" << std::endl;
  // remap函数，对图像进行重映射. 重新映射图像的像素值，例如图像的旋转、缩放、平移等
  double angle = 90.0; // 旋转角度
  cv::Point2f center(image.cols / 2, image.rows / 2); // 旋转中心
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0); // 获取旋转矩阵
  cv::Mat map1(image.size(), CV_32FC1); // 创建一个和image大小一样的CV_32FC1类型的Mat对象
  cv::Mat map2(image.size(), CV_32FC1); // 创建一个和image大小一样的CV_32FC1类型的Mat对象
  for(int y = 0; y < image.rows; y++) {
    for(int x = 0; x < image.cols; x++) {
      cv::Point2f p(x, y); // 创建一个点
      cv::Point2f new_p = cv::Point2f(rot.at<double>(0, 0) * p.x + rot.at<double>(0, 1) * p.y + rot.at<double>(0, 2),
                                      rot.at<double>(1, 0) * p.x + rot.at<double>(1, 1) * p.y + rot.at<double>(1, 2)); // 旋转后的点
      map1.at<float>(y, x) = new_p.x; // x坐标的映射
      map2.at<float>(y, x) = new_p.y; // y坐标的映射
    }

  }
  cv::Mat remap_rotated; // 创建一个空的Mat对象
  /**
   * @param src 输入图像
   * @param dst 输出图像, size和map1一样，type和src一样
   * @param map_1 第一个映射矩阵，即x坐标的映射
   * @param map_2 第二个映射矩阵，即y坐标的映射
   * @param interpolation 插值方法
   * @param borderMode 边界模式
   * @param borderValue 边界值
   */
  cv::remap(image, remap_rotated, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 0, 0));
  cv::imwrite(root_path + "/output/remap_rotated.png", remap_rotated);

  /**********************************cv::Rect*********************************************/
  std::cout << "/*************cv::Rect***************/" << std::endl;
  // Rect矩形类，用于表示矩形区域
  /**
   * @param x 矩形左上角的x坐标，x轴为矩形水平方向
   * @param y 矩形左上角的y坐标，y轴为矩形垂直方向
   * @param width 矩形的宽度
   * @param height 矩形的高度
   */
  cv::Rect rect(1, 2, 4, 3); // 创建一个Rect对象，x=1, y=2, width=3, height=3
  std::cout << "rect: \n" << rect << std::endl;
  double area = rect.area(); // 获取rect的面积
  std::cout << "rect area: " << area << std::endl;
  cv::Point br = rect.br(); // 获取rect的右下角坐标
  std::cout << "rect br: " << br << std::endl;
  cv::Point p(1, 2);
  bool contains = rect.contains(p); // 判断rect是否包含点p
  std::cout << "rect contains: " << contains << std::endl;
  bool empty = rect.empty(); // 判断rect是否为空
  std::cout << "rect empty: " << empty << std::endl;
  cv::Size size = rect.size(); // 获取rect的大小
  std::cout << "rect size: " << size << std::endl;
  cv::Point tl = rect.tl(); // 获取rect的左上角坐标
  std::cout << "rect tl: " << tl << std::endl;
  int height = rect.height; // 获取rect的高度
  std::cout << "rect height: " << height << std::endl;
  int width = rect.width; // 获取rect的宽度
  std::cout << "rect width: " << width << std::endl;
  int x = rect.x; // 获取rect的左上角x坐标
  std::cout << "rect x: " << x << std::endl;
  int y = rect.y; // 获取rect的左上角y坐标
  std::cout << "rect y: " << y << std::endl;
  cv::Rect rect2(2, 3, 5, 5); // 创建一个Rect对象，x=2, y=3, width=5, height=5
  cv::Rect rect3 = rect & rect2; // 获取rect和rect2的交集
  std::cout << "rect3: width: " << rect3.width
            << ", height: " << rect3.height
            << ", rect3: \n" << rect3
            << std::endl;
  cv::Rect rect4 = rect | rect2; // 获取rect和rect2的并集
  std::cout << "rect4: width: " << rect4.width
            << ", height: " << rect4.height
            << ", rect4: \n" << rect4
            << std::endl;
  cv::Rect rect1(100, 200, 500, 500); // 创建一个Rect对象，x=2, y=3, width=4, height=4
  cv::Mat image_rect = cv::Mat(image, rect1); // 获取image的rect1区域
  std::cout << "image_rect size: " << image_rect.size()
            << ", image size: " << image.size()
            << std::endl;
  cv::imwrite(root_path + "/output/image_rect.png", image_rect);

  /**********************************cv::split*********************************************/
  std::cout << "/*************cv::split***************/" << std::endl;
  // split函数，将多通道图像分割成多个单通道图像
  std::vector<cv::Mat> channels; // 创建一个Mat向量
  cv::split(image, channels); // 将image的通道分割到channels中
  std::cout << "channels size: " << channels.size()
            << ", image channels: " << image.channels()
            << std::endl;
  cv::Mat image_split_0 = channels[0]; // 获取channels的第0个元素
  cv::Mat image_split_1 = channels[1]; // 获取channels的第1个元素
  cv::Mat image_split_2 = channels[2]; // 获取channels的第2个元素
  std::cout << "image_split_0 size: " << image_split_0.size()
            << ", image_split_0 channels: " << image_split_0.channels()
            << ", image size: " << image.size()
            << std::endl;
  cv::imwrite(root_path + "/output/image_split_0.png", image_split_0);
  cv::imwrite(root_path + "/output/image_split_1.png", image_split_1);
  cv::imwrite(root_path + "/output/image_split_2.png", image_split_2);

  /**********************************cv::merge*********************************************/
  std::cout << "/*************cv::merge***************/" << std::endl;
  // merge函数，将多个单通道图像合并成多通道图像
  cv::Mat image_merge;
  std::vector<cv::Mat> channels_r; // 创建一个Mat向量
  channels_r.push_back(channels[0]); // 添加channels的第0个元素
  channels_r.push_back(channels[1]); // 添加channels的第1个元素
  cv::Mat blank = cv::Mat::zeros(channels[0].size(), CV_8UC1); // 创建一个和channels[0]大小一样的全0矩阵
  channels_r.push_back(blank); // 添加blank
  cv::merge(channels_r, image_merge); // 将channels_r合并到image中
  std::cout << "image_merge size: " << image_merge.size()
            << ", image size: " << image.size()
            << std::endl;
  cv::imwrite(root_path + "/output/image_merge.png", image_merge);

  /**********************************cv::flip*********************************************/
  std::cout << "/*************cv::flip***************/" << std::endl;
  // flip函数，对图像进行翻转
  cv::Mat image_flip;
  cv::flip(image, image_flip, 1); // 对image进行水平翻转：1表示水平翻转(沿y轴)，0表示垂直翻转(沿x轴)，-1表示水平垂直翻转
  cv::imwrite(root_path + "/output/image_flip.png", image_flip);

  /**********************************cv::vconcat*********************************************/
  std::cout << "/*************cv::vconcat***************/" << std::endl;
  // vconcat函数，对图像进行垂直拼接
  cv::Mat image_vconcat;
  cv::vconcat(image, image_flip, image_vconcat); // 对image进行垂直拼接
  std::cout << "image_vconcat size: " << image_vconcat.size()
            << ", image size: " << image.size()
            << ", image_flip size: " << image_flip.channels()
            << std::endl;
  cv::imwrite(root_path + "/output/image_vconcat.png", image_vconcat);
}

void ImRead() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";

  /**
   * @param filename 图片路径
   * @param flags 读取图片的方式
   * 
   * @return cv::Mat 返回读取的图片。
   *                 如果读取失败，返回一个空的Mat对象
   *                 如果图片有颜色，返回的Mat对象的通道按照BGR的顺序排列
   * 
   * @note 支持的文件格式有：
   * - Windows bitmaps - *.bmp, *.dib (always supported)
   * - JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section)
   * - JPEG 2000 files - *.jp2 (see the Notes section)
   * - Portable Network Graphics - *.png (see the Notes section)
   * - WebP - *.webp (see the Notes section)
   * - Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
   * 
   * flags 可以参考cv::ImreadModes：
   * cv::IMREAD_UNCHANGED = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
   * cv::IMREAD_GRAYSCALE = 0, //!< If set, always convert image to the single channel grayscale image.
   * cv::IMREAD_COLOR = 1, //!< If set, always convert image to the 3 channel BGR color image.
   * cv::IMREAD_ANYDEPTH = 2, //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
   * cv::IMREAD_ANYCOLOR = 4, //!< If set, the image is read in any possible color format.
   * 。。。
   */
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  cv::Mat image_unchanged = cv::imread(path, cv::IMREAD_UNCHANGED);
  std::cout << "image_unchnaged size: " << image_unchanged.size()
            << ", image_unchnaged channels: " << image_unchanged.channels()
            << ", image_unchnaged depth: " << image_unchanged.depth()
            << ", image_unchanged elemSize: " << image_unchanged.elemSize()
            << std::endl;
}

void ImWrite() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  cv::Mat image_alpha = cv::Mat(image.size(), CV_8UC1); // 创建一个和image大小一样的CV_8UC4类型的Mat对象
  for(int y = 0; y < image.rows; y++) {
    for(int x = 0; x < image.cols; x++) {
      image_alpha.at<uchar>(y, x) = 100; // 设置image_alpha的所有元素为255
    }
  }
  cv::Mat image_png_alpha = cv::Mat(image.size(), CV_8UC4); // 创建一个和image大小一样的CV_8UC4类型的Mat对象
  cv::merge(std::vector<cv::Mat>{image, image_alpha}, image_png_alpha); // 将image和image_alpha合并到image_png_alpha中

  std::vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION); // 设置保存参数
  params.push_back(9); // 设置保存参数

  /**
   * @brief 将Mat对象保存为图片,图片格式由filename的后缀决定
   * @param filename 图片保存路径
   * @param img 要保存的图片
   * @param params 保存参数，默认为空
   * 
   * @return bool 保存成功返回true，否则返回false
   * 
   * @note 通常情况下，只有CV_8UC1, CV_8UC3(BGR)类型的Mat对象才能保存为图片，除了以下几种类型：
   * - CV_16U可以在PNG，JPEG2000，TIFF中保存
   * - PNG图像带有透明通道的CV_8UC4、CV_16UC4类型的Mat对象可以保存为PNG
   */
  cv::imwrite(root_path + "/output/image_png_alpha.png", image_png_alpha, params); // 将image_gray保存为png格式
}

void Color() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  cv::Mat image_gray;

  /**
   * @brief 将图像从一种颜色空间转换为另一种颜色空间
   * @param src 输入图像
   * @param dst 输出图像
   * @param code 颜色空间转换方式
   * @param dstCn 输出图像的通道数，如果为0，则根据src和code自动设置
   * 
   * @note code可以参考cv::ColorConversionCodes
   * cv::COLOR_BGR2BGRA = 0, //!< add alpha channel to RGB or BGR image
   * cv::COLOR_RGB2RGBA = COLOR_BGR2BGRA, //!< add alpha channel to RGB or BGR image
   * cv::COLOR_BGRA2BGR = 1, //!< remove alpha channel from RGB or BGR image
   * cv::COLOR_RGBA2RGB = COLOR_BGRA2BGR, //!< remove alpha channel from RGB or BGR image
   * cv::COLOR_BGR2RGBA = 2, //!< convert between RGB and BGR color spaces (with or without alpha channel)
   * cv::COLOR_RGB2BGRA = COLOR_BGR2RGBA, //!< convert between RGB and BGR color spaces (with or without alpha channel)
   */
  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY); // 将image转换为灰度图
  std::cout << "image_gray size: " << image_gray.size()
            << ", image_gray channels: " << image_gray.channels()
            << std::endl;
  cv::imwrite(root_path + "/output/image_gray.png", image_gray);

  // BGR顺序，这是一张蓝色的图片
  cv::Mat image_blue = cv::Mat(image.size(), CV_8UC3, cv::Scalar{255, 0, 0}); // 创建一个和image大小一样的CV_8UC3类型的Mat对象
  cv::Mat image_green = cv::Mat(image.size(), CV_8UC3, cv::Scalar{0, 255, 0}); // 创建一个和image大小一样的CV_8UC3类型的Mat对象
  cv::Mat image_red = cv::Mat(image.size(), CV_8UC3, cv::Scalar{0, 0, 255}); // 创建一个和image大小一样的CV_8UC3类型的Mat对象
  cv::imwrite(root_path + "/output/image_blue.png", image_blue);
  cv::imwrite(root_path + "/output/image_red.png", image_red);
  cv::imwrite(root_path + "/output/image_green.png", image_green);
}

void line() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  cv::Mat image_line = image.clone();

  /**
   * @brief 画线
   * @param img 要画线的图像
   * @param pt1 线的起点
   * @param pt2 线的终点
   * @param color 线的颜色
   * @param thickness 线的粗细
   * @param lineType 线的类型
   * @param shift 小数点位数偏移
   * 
   * @note lineType可以参考cv::LineTypes
   * cv::FILLED = -1, //!< filled
   * cv::LINE_4 = 4, //!< 4-connected line
   * cv::LINE_8 = 8, //!< 8-connected line
   * cv::LINE_AA = 16 //!< antialiased line
   */
  cv::line(image_line, cv::Point(0, 0), cv::Point(100, 100), cv::Scalar(0, 0, 255), 2, cv::LINE_4); // 在image_line上画一条直线
  cv::imwrite(root_path + "/output/image_line.png", image_line);
}

void putText() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  
  cv::Mat image_text = image.clone();

  /**
   * @brief 在图像上写文字
   * @param img 要写文字的图像
   * @param text 要写的文字
   * @param org 文字的左下角坐标
   * @param fontFace 字体
   * @param fontScale 字体大小
   * @param color 字体颜色
   * @param thickness 字体粗细
   * @param lineType 线的类型
   * @param bottomLeftOrigin 文字的左下角坐标是否为左下角, 默认为false。true表示左上角坐标
   * 
   * @note fontFace可以参考cv::HersheyFonts
   * cv::FONT_HERSHEY_SIMPLEX = 0, //!< normal size sans-serif font
   * cv::FONT_HERSHEY_PLAIN = 1, //!< small size sans-serif font
   */
  cv::putText(image_text, "Hello, OpenCV", cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
  cv::imwrite(root_path + "/output/image_text.png", image_text);
}

void rectangle() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  cv::Mat image_rectangle = image.clone();

  /**
   * @brief 画矩形
   * @param img 要画矩形的图像
   * @param pt1 矩形的左上角
   * @param pt2 矩形的右下角
   * @param color 矩形的颜色
   * @param thickness 矩形的粗细
   * @param lineType 线的类型
   * @param shift 小数点位数偏移
   * 
   * @note lineType可以参考cv::LineTypes
   * cv::FILLED = -1, //!< filled
   * cv::LINE_4 = 4, //!< 4-connected line
   * cv::LINE_8 = 8, //!< 8-connected line
   * cv::LINE_AA = 16 //!< antialiased line
   */
  cv::rectangle(image_rectangle, cv::Point(10, 10), cv::Point(100, 100), cv::Scalar(0, 0, 255), 2, cv::LINE_4); // 在image_rectangle上画一个矩形
  cv::imwrite(root_path + "/output/image_rectangle.png", image_rectangle);
}

void circle() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  cv::Mat image_circle = image.clone();

  /**
   * @brief 画圆
   * @param img 要画圆的图像
   * @param center 圆心
   * @param radius 半径
   * @param color 圆的颜色
   * @param thickness 圆的粗细
   * @param lineType 线的类型
   * @param shift 小数点位数偏移
   */
  cv::circle(image_circle, cv::Point(100, 100), 50, cv::Scalar(0, 0, 255), 2, cv::LINE_4); // 在image_circle上画一个圆
  cv::imwrite(root_path + "/output/image_circle.png", image_circle);
}

void imdecode() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";


  // 这种方式不行，因为Mat.data存储图像格式跟原始文件不一致。data指向的是图像的数据，而不是图像的完整二进制数据
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  std::ptrdiff_t size = image.dataend - image.datastart;
  std::cout << "image size: " << image.size()
            << ", image channels: " << image.channels()
            << ", image depth: " << image.depth()
            << ", image length: " << image.cols
            << ", image width: " << image.rows
            << ", length * width = " << image.cols * image.rows
            << ", diff: " << size
            << ", total: " << image.total() << " * elemSize: " << image.elemSize()
            << " = " << image.total() * image.elemSize()
            << std::endl;
  std::vector<uchar> data(const_cast<uchar*>(image.datastart), const_cast<uchar*>(image.dataend));


  std::ifstream ifs = std::ifstream(path, std::ios::binary);
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string str = ss.str();
  std::vector<uchar> data1(
    reinterpret_cast<uchar*>(const_cast<char*>(str.data())),
    str.size() + reinterpret_cast<uchar*>(const_cast<char*>(str.data()))
  );
  std::cout << "data1 size: " << data1.size() << " B, "
            << data1.size() / 1024.0 << " KB, "
            << data1.size() / 1024.0 / 1024.0 << " MB" << std::endl;

  /**
   * @brief 从内存中的buffer解析图片
   * @param buf 图片数据
   * @param flags 读取图片的方式，参考cv::ImreadModes
   */
  cv::Mat image_decode = cv::imdecode(data1, cv::IMREAD_UNCHANGED);
  if(image_decode.empty()) {
    std::cout << "imdecode error" << std::endl;
    return;
  }
  std::cout << "image_decode size: " << image_decode.size()
            << ", image_decode channels: " << image_decode.channels()
            << std::endl;
  cv::imwrite(root_path + "/output/image_decode.png", image_decode);

  ifs.close();
}

void imencode() {
  std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example";
  std::string path = root_path + "/image/bicycle.jpg";
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  std::vector<uchar> buf;
  std::vector<int> params;
  params.push_back(cv::IMWRITE_JPEG_QUALITY); // 设置保存参数
  params.push_back(100); // 设置保存参数
  cv::imencode(".jpg", image, buf); // 将image保存为jpg格式

  std::ofstream ofs = std::ofstream(root_path + "/output/image_encode.jpg", std::ios::binary);
  ofs.write(reinterpret_cast<char*>(buf.data()), buf.size());
  ofs.close();
}

int main(int argc, char** argv) {
    std::cout << "===================ConFun===================" << std::endl;
    ConFun();
    std::cout << "===================PubMem===================" << std::endl;
    PubMem();
    std::cout << "===================MemFun===================" << std::endl;
    MemFun();
    std::cout << "===================GlobalFun===================" << std::endl;
    GlobalFun();
    std::cout << "===================ImRead===================" << std::endl;
    ImRead();
    std::cout << "===================ImWrite===================" << std::endl;
    ImWrite();
    std::cout << "===================Color===================" << std::endl;
    Color();
    std::cout << "===================line===================" << std::endl;
    line();
    std::cout << "===================putText===================" << std::endl;
    putText();
    std::cout << "===================rectangle===================" << std::endl;
    rectangle();
    std::cout << "===================circle===================" << std::endl;
    circle();
    std::cout << "===================imdecode===================" << std::endl;
    imdecode();
    std::cout << "===================imencode===================" << std::endl;
    imencode();
    std::cout << "===================CommandLineParser Function===================" << std::endl;
    
    /* commandline */
    const cv::String keys =
            "{help h usage ? |      | print this message   }"
            "{@image1        |      | image1 for compare   }"
            "{@image2        |<none>| image2 for compare   }"
            "{@repeat        |1     | number               }"
            "{path           |.     | path to file         }"
            "{fps            | -1.0 | fps for output video }"
            "{N count        |100   | count of objects     }"
            "{ts timestamp   |      | use time stamp       }"
            ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("\nThis program demonstrates connected components and use of the trackbar\n");
    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }
    // parser.printMessage();
    auto parserpath = parser.get<cv::String>("path");
    std::cout << "path: " << parserpath << std::endl;

    return 0;
}