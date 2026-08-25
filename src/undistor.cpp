#include <string> // 字符串
#include <vector> // 向量
#include <iostream> // 输入输出流
#include <fstream> // 文件流
#include <sstream> // 字符串流
#include <algorithm> // 算法
#include <set> // 集合

#include <opencv2/core.hpp> // OpenCV核心功能
#include <opencv2/imgcodecs.hpp> // 图像编解码
#include <opencv2/highgui.hpp> // GUI
#include <opencv2/imgproc.hpp> // 图像处理
#include <opencv2/opencv.hpp> // OpenCV主要功能
#include <opencv2/imgproc/types_c.h> // 旧版的图像处理


/**
 * 坐标：
 * 0.世界坐标：
 *   - P_w = (X_w, Y_w, Z_w)
 * 1.相机坐标系(3D,单位m)：
 *   - P_c = (X_c, Y_c, Z_c)
 * 2.归一化平面坐标系(Z=1那个平面):
 *   - (x, y) = (X_c / Z_c, Y_c / Z_c)
 * 3.像素坐标系(2D,单位px):
 *   - (u, v)
 *
 * 两个都是半径：
 *   - r: 归一化平面上到原点的距离,也就是sqrt(x^2 + y^2)
 *   - phi：像素平面上到中心的距离，单位是像素
 *   - 畸变模型就是建立r(或者等价的thea)和phi的关系
 */

/**
 * 所有相机模型本质都是：
 *   - 一条入射角是thea(光线跟光轴的夹角)的光线，会落在像面居中心phi多远的地方
 *   - 不同模型就是不同的 phi = f(thea) 函数。
 *   - 方位角gama：从光轴方向俯视看，绕着光轴转了多少度。
 *   - 畸变：实际的phi和理想的phi不一样，畸变模型就是描述这个差异的函数。畸变系数就是修正这个差异的参数。
 * 成像链条：
 *  世界坐标 -> 相机坐标 -> 归一化平面坐标(x_n, y_n) -> 畸变 -> (x_d, y_d) -> K矩阵 -> 像素坐标(u, v)
 * 下列参数含义：
 *   - thea: 入射角，光线与光轴的夹角
 *   - phi：单位是像素，而像素数量依赖焦距f，因此除以焦距f后，phi/f = f(thea) / f统一到跟焦距无关平面
 * 针孔相机模型(pinhole camera model)：
 *   - r = sqrt(x^2 + y^2)，这里是归一化平面的坐标(x,y)
 *   - 径向：x_r = x(1 + k1*r^2 + k2*r^4 + k3*r^6), y_r = y(1 + k1*r^2 + k2*r^4 + k3*r^6)
 *   - 切向：x_d = x_r + [2*p1*x*y + p2*(r^2 + 2*x^2)], y_d = y_r + [p1*(r^2 + 2*y^2) + 2*p2*x*y]
 *   - 像素：u = fx * x_d + cx, v = fy * y_d + cy
 *   - 系数：顺序按照OpenCV的顺序，k1, k2, p1, p2, k3
 *   - phi = tan(thea) * (1 + k1*tan(thea)^2 + k2*tan(thea)^4 + k3*tan(thea)^6) + 切向，其中r=tan(thea)
 * KB(Kannala-Brandt)模型:
 *   - r = sqrt(x^2 + y^2)，这里是归一化平面的坐标(x,y)
 *   - thea = atan(r)
 *   - phi = thea + k1*thea^3 + k2*thea^5 + k3*thea^7 + k4*thea^9
 *   - u = fx * phi * cos(gama) + cx
 *   - v = fy * phi * sin(gama) + cy
 * Omnidirectional Camera Model(OCAM)模型:
 *   - Omnidr = Scaramuzza/OCamCalib多项式
 *   - r = sqrt(x^2 + y^2)，这里是归一化平面的坐标(x,y)
 *   - thea = atan2(1, r)
 *   - phi = a0 + a1*thea + a2*thea^2 + a3*thea^3 + a4*thea^4 + a5*thea^5 + a6*thea^6 + a7*thea^7 + a8*thea^8
 *   - (u, v) = affine2x2 * (phi * cos(gama), phi * sin(gama)) + (cx, cy)
 *   - 上述的系数都没有物理意义
 */

/**
 * 0.去畸变：
 *      相机是3D射线到2d像素的映射。理想针孔相机这个映射是摄影线性的(一个矩阵K)
 *      但是实际相机是有畸变的。去畸变就是造一张新图，让这张新图严格服从理想针孔映射
 * 1.针孔相机模型的畸变矫正：
 *      当前的getOptimalNewCameraMatrix/initUndistortRectifyMap/remap
 *      适用于常规透视相机模型(针孔模型)，不适用鱼眼
 * 2.鱼眼相机模型的畸变矫正：
 *      OpenCV提供了专门针对鱼眼相机模型的函数，
 *      如fisheye::estimateNewCameraMatrixForUndistortRectify和fisheye::initUndistortRectifyMap
 * 3.OCAM(omnidirectional camera model, 全向相机模型)：
 *      比普通的鱼眼模型更泛化，适合超大视场甚至接近全向镜头
 *      不再是简单的针孔或者鱼眼畸变模型能解决的，而是直接使用多项式去描述畸变关系
 *
 * resize会改变图像的尺寸：
 *  1.相机内参中的焦距和主点坐标也需要相应地进行缩放。fx和fy需要乘以水平和垂直的缩放因子，cx和cy也需要乘以相应的缩放因子。
 *  2.畸变系数通常不需要调整，为它们是归一化坐标下的参数，和分辨率无关。
 */

 /**
  * 1. FrontWide:
  *      - fx:1906.6, fy:1906.18, cx:1923.26, cy:1022.45
  *      - k1:-0.0299548, k2:-0.00364585, p1:-0.00155829, p2:0.00104736
  *      - 3840x2048
  *      - fisheye
  * 2. FrontLong:
  *      - fx:7328.27, fy:7329.37, cx:1899.02, cy:997.359
  *      - k1:0.205114, k2:-4.08226, p1:40.0991, p2:-165.095
  *      - 3840x2048
  *      - fisheye
  * 3.Rear:
  *      - fx:1100.54, fy:1100.49, cx:963.113, cy:508.711
  *      - k1:-0.00767063, k2:0.00487208, p1:-0.00427585, p2:0.0095581
  *      - 1920x1024
  *      - fisheye
  * 4.SideLeftFront:
  *      - fx=526.563, fy=526.652, cx=478.093, cy=255.934
  *      - -0.0128233, -0.0098962, 0.0130589, -0.00157833
  *      - 960x512
  *      - fisheye
  * 5.SideLeftRear:
  *      - fx=526.854, fy=526.877, cx=479.752, cy=255.504
  *      - -0.0148726, -0.00231126, 0.00333634, 0.00283011
  *      - 960x512
  *      - fisheye
  * 6.SideRightFront:
  *      - fx=526.774, fy=526.745, cx=480.695, cy=254.451
  *      - -0.0151338, -0.00391945, 0.00588716, 0.0013771
  *      - 960x512
  *      - fisheye
  * 7.SideRightRear:
  *      - fx=526.298, fy=526.323, cx=481.244, cy=254.291
  *      - -0.0138223, -0.00853396, 0.0121397, -0.00161409
  *      - 960x512
  *      - fisheye
  * 8.Infrared:
  *      - fx=742.425, fy=742.425, cx=316.808, cy=253.326
  *      - -0.303234, 0.053968, -0.001405, -0.000301, 0.0
  *      - 640x512
  *      - pinhole
  */
struct CameraParams {
    cv::Mat K; // 内参矩阵
    std::vector<double> dist_coeffs; // 畸变系数
    cv::Size ori_image_size; // 原始图像尺寸
    cv::Size undistorted_image_size; // 去畸变后图像尺寸
    double balance = 0.0; // balance参数
    std::string input_image_path = ""; // 输入图像路径
    std::string output_input_image_path = ""; // 输入图像保存路径
    std::string output_undistorted_image_path = ""; // 去畸变后图像保存路径
};

void fisheye_undistor_sequnce(const CameraParams& params) {
    // 相机内参矩阵
    cv::Mat k = params.K;
    // 相机的畸变系数
    std::vector<double> dist_coeffs = params.dist_coeffs;
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // 相机的分辨率
    int width = params.ori_image_size.width;
    int height = params.ori_image_size.height;
    cv::Size image_size(width, height);
    // 参数设置
    double balance = params.balance;
    cv::Size new_image_size = params.undistorted_image_size;
    cv::Mat new_camera_matrix = cv::Mat();

    std::cout << "Camera matrix K: \n" << k << std::endl;
    std::cout << "Distortion coefficients: \n" << dist_coeffs_mat.t() << std::endl;
    std::cout << "Original image size: " << image_size << std::endl;
    std::cout << "Undistorted image size: " << new_image_size << std::endl;
    std::cout << "Balance: " << balance << std::endl;

    // 得到新的相机内参矩阵
    /**
     * @brief 得到新的相机内参矩阵，用于去畸变和校正
     * @param K 输入的相机内参矩阵
     * @param D 输入的畸变系数
     * @param image_size 输入图像的大小
     * @param R 可选的矫正旋转矩阵(3x3 1-channel, or vector: 3x1/1x3)，一般为单位矩阵
     * @param P 输出的新的内参矩阵
     * @param balance 黑边保留程度，默认0.0，0表示最大限度去黑边，1表示最大限度保留原始图片
     * @param new_size 去畸变后图像尺寸，通常与原图一致，默认cv::Size()
     * @param fov_scale 视场缩放因子，默认1.0，小于1会缩小视场，等于1保持原始视场
     */
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(k, dist_coeffs_mat, image_size, cv::Mat(), new_camera_matrix, balance, new_image_size, 1.0);
    std::cout << "new_camera_matrix: \n" << new_camera_matrix << std::endl;
    
    // 计算畸变矫正映射
    cv::Mat map1, map2;

    /**
     * @brief 计算去畸变和校正的映射
     * @param K 输入的相机内参矩阵
     * @param D 输入的畸变系数
     * @param R 可选的矫正旋转矩阵，一般设置为单位矩阵
     * @param P 输入的新的相机内参矩阵
     * @param size 去畸变后图像的大小，通常与原图一致
     * @param m1type 输出map1的类型，通常为CV_32FC1或CV_32FC2
     * @param map1 输出的第一个映射矩阵，x坐标的映射
     * @param map2 输出的第二个映射矩阵，y坐标的映射
     */
    cv::fisheye::initUndistortRectifyMap(k, dist_coeffs_mat, cv::Mat(), new_camera_matrix, new_image_size, CV_32FC1, map1, map2);
    std::cout << "map1 size: " << map1.size() << ", map1 type: " << map1.type() << std::endl;
    std::cout << "map2 size: " << map2.size() << ", map2 type: " << map2.type() << std::endl;

    // 获取输入图像
    std::string input_image_path = params.input_image_path;
    std::ifstream input_image_stream(input_image_path, std::ios::binary | std::ios::ate);
    if(!input_image_stream.is_open()) {
        std::cerr << "Could not open input image file: " << input_image_path << std::endl;
        return;
    }
    std::streamsize input_image_size = input_image_stream.tellg();
    input_image_stream.seekg(0, std::ios::beg);
    if(input_image_size != image_size.width * image_size.height * 3 / 2) {
        std::cerr << "Input image size does not match expected size: " << input_image_size << " bytes." << std::endl;
        return;
    }
    std::cout << "Input image size: " << input_image_size << " bytes." << std::endl;
    std::vector<unsigned char> input_image_data(input_image_size);
    if(!input_image_stream.read(reinterpret_cast<char*>(input_image_data.data()), input_image_size)) {
        std::cerr << "Error reading input image file: " << input_image_path << std::endl;
        return;
    }
    cv::Mat input_image(image_size.height + image_size.height / 2, image_size.width, CV_8UC1, input_image_data.data());
    cv::Mat input_image_bgr;
    cv::cvtColor(input_image, input_image_bgr, cv::COLOR_YUV2BGR_NV12); // NV12格式转换为BGR格式
    std::string output_input_image_path = params.output_input_image_path;
    cv::imwrite(output_input_image_path, input_image_bgr);
    std::cout << "Saved input image in BGR format to: " << output_input_image_path << std::endl;
    
    // 使用remap函数进行去畸变
    cv::Mat undistorted_image_bgr;

    /**
     * @brief 对图像进行几何变换
     * @param src 输入图像，InputArray类型
     * @param dst 输出图像，跟map的size一样，跟src的type一样
     * @param map1 x坐标的映射矩阵
     * @param map2 y坐标的映射矩阵
     * @param interpolation 插值方法
     * @param borderMode 边界模式，默认为BORDER_CONSTANT
     * @param borderValue 边界值，默认为Scalar()
     */
    cv::remap(input_image_bgr, undistorted_image_bgr, map1, map2, cv::INTER_LINEAR);
    std::string output_undistorted_image_path = params.output_undistorted_image_path;
    cv::imwrite(output_undistorted_image_path, undistorted_image_bgr);   
    std::cout << "Saved undistorted image to: " << output_undistorted_image_path << std::endl; 
}

void infrared_pinhole_undistor_sequnce() {
    // 红外相机内参矩阵
    cv::Mat k = (cv::Mat_<double>(3, 3) << 742.425, 0, 316.808
                                            , 0, 742.425, 253.326
                                            , 0, 0, 1);
    // 红外相机的畸变系数
    std::vector<double> dist_coeffs = {-0.303234, 0.053968, -0.001405, -0.000301, 0.0};
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // 红外相机的分辨率
    int width = 640;
    int height = 512;
    cv::Size image_size(width, height);

    // 参数设置
    double alpha = 0.0;
    cv::Size new_image_size = image_size;
    cv::Rect valid_pix_roi(0, 0, width, height);

    /**
     * @brief 返回新的内参矩阵
     * @param cameraMatrix 输入的相机内参矩阵
     * @param distCoeffs 输入的畸变系数
     * @param imageSize 输入图像的大小
     * @param alpha 自由缩放参数，范围[0, 1]。
     *              alpha=0表示返回一个裁剪后的图像，去掉所有无效像素；
     *              alpha=1表示返回一个未裁剪的图像，包含所有像素；
     *              0<alpha<1表示返回一个介于两者之间的图像，裁剪掉部分无效像素。
     * @param newImgSize 畸变矫正后图像的大小，默认与输入图像大小相同。
     * @param validPixROI 输出参数，有效像素区域，即裁剪后的图像中包含的有效像素的矩形区域(非填充或者无效区域)。
     * @param centerPrincipalPoint 是否将新的内参矩阵的主点设置为图像中心，默认为false。
     */
    cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(k, dist_coeffs_mat, image_size, alpha, new_image_size, &valid_pix_roi);
    std::cout << "new_camera_matrix: \n" << new_camera_matrix << std::endl;
    std::cout << "valid_pix_roi: \n" << valid_pix_roi << std::endl;
    std::cout << "new_image_size: " << new_image_size << std::endl;

    // 计算畸变矫正映射
    cv::Mat map1, map2;
    /**
     * @brief 计算去畸变和校正的映射
     * @param cameraMatrix 输入的相机内参矩阵
     * @param distCoeffs 输入的畸变系数
     * @param R 可选的矫正旋转矩阵(3x3)，默认为单位矩阵
     * @param newCameraMatrix 输入的新的相机内参矩阵
     * @param size 去畸变后图像的大小，默认为与输入图像相同
     * @param m1type 输出map1的类型，可选CV_32FC1，CV_32FC2或CV_16SC2，默认为CV_32FC1
     * @param map1 输出的第一个映射矩阵，x坐标方向的映射。
     * @param map2 输出的第二个映射矩阵，y坐标方向的映射。
     *
     * @note 根据去畸变前的相机内参，畸变系数和去畸变后的相机内参，以及校正旋转(可选参数)
     *       将输入图像的每个像素点映射到去畸变后的图像中的对应位置。
     *       并把这种映射关系保存在map1和map2中，map1存储x坐标的映射，map2存储y坐标的映射。
     */
    cv::initUndistortRectifyMap(k, dist_coeffs_mat, cv::Mat(), new_camera_matrix, new_image_size, CV_32FC1, map1, map2);
    std::cout << "map1 size: " << map1.size() << ", map1 type: " << map1.type() << std::endl;
    std::cout << "map2 size: " << map2.size() << ", map2 type: " << map2.type() << std::endl;

    // 使用remap函数进行去畸变
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/infrared_640_512_nv12.yuv";
    int input_image_width = 640;
    int input_image_height = 512;
    std::ifstream input_image_stream(input_image_path, std::ios::binary | std::ios::ate);
    if(!input_image_stream.is_open()) {
        std::cerr << "Could not open input image file: " << input_image_path << std::endl;
        return;
    }
    std::streamsize input_image_size = input_image_stream.tellg();
    input_image_stream.seekg(0, std::ios::beg);
    if(input_image_size != input_image_width * input_image_height * 3 / 2) {
        std::cerr << "Input image size does not match expected size: " << input_image_size << " bytes." << std::endl;
        return;
    }
    std::cout << "Input image size: " << input_image_size << " bytes." << std::endl;
    std::vector<unsigned char> input_image_data(input_image_size);
    if(!input_image_stream.read(reinterpret_cast<char*>(input_image_data.data()), input_image_size)) {
        std::cerr << "Error reading input image file: " << input_image_path << std::endl;
        return;
    }
    cv::Mat input_image(input_image_height + input_image_height / 2, input_image_width, CV_8UC1, input_image_data.data());
    cv::Mat input_image_bgr;
    cv::cvtColor(input_image, input_image_bgr, cv::COLOR_YUV2BGR_NV12); // NV12格式转换为BGR格式
    std::string output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/infrared_640_512_nv12_bgr.jpg";
    cv::imwrite(output_input_image_path, input_image_bgr);
    std::cout << "Saved input image in BGR format to: " << output_input_image_path << std::endl;
    cv::Mat undistorted_image_bgr;

    /**
     * @brief 对图像进行几何变换
     * @param src 输入图像，InputArray类型
     * @param dst 输出图像，跟map的size一样，跟src的type一样
     * @param map1 x坐标的映射矩阵
     * @param map2 y坐标的映射矩阵
     * @param interpolation 插值方法
     * @param borderMode 边界模式，默认为BORDER_CONSTANT
     * @param borderValue 边界值，默认为Scalar()
     */
    cv::remap(input_image_bgr, undistorted_image_bgr, map1, map2, cv::INTER_LINEAR);
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/infrared_640_512_nv12_pinhole_undistorted.jpg";
    cv::imwrite(output_image_path, undistorted_image_bgr);
    std::cout << "Saved undistorted image to: " << output_image_path << std::endl;
}

void fw_fisheye_undistor_sequnce() {
    // FW相机内参矩阵
    cv::Mat k = (cv::Mat_<double>(3, 3) << 1906.6, 0, 1923.26
                                            , 0, 1906.18, 1022.45
                                            , 0, 0, 1);
    // FW相机的畸变系数
    std::vector<double> dist_coeffs = {-0.0299548, -0.00364585, -0.00155829, 0.00104736};
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // FW相机的分辨率
    int width = 3840;
    int height = 2048;
    cv::Size image_size(width, height);

    // 得到新的相机内参矩阵
    double balance = 1.0;
    cv::Size new_image_size = image_size;
    cv::Mat new_camera_matrix = cv::Mat();

    CameraParams params;
    params.K = k;
    params.dist_coeffs = dist_coeffs;
    params.ori_image_size = image_size;
    params.undistorted_image_size = new_image_size;
    params.balance = balance;
    params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/frontwide_3840_2048_nv12.yuv";
    params.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_bgr.jpg";
    params.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_undistorted.jpg";
    
    fisheye_undistor_sequnce(params);
}

void fl_fisheye_undistor_sequnce() {
    // fl相机内参矩阵
    cv::Mat k = (cv::Mat_<double>(3, 3) << 7328.27, 0, 1899.02
                                            , 0, 7329.37, 997.359
                                            , 0, 0, 1);
    // fl相机的畸变系数
    std::vector<double> dist_coeffs = {0.205114, -4.08226, 40.0991, -165.095};
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // fl相机的分辨率
    int width = 3840;
    int height = 2048;
    cv::Size image_size(width, height);

    // 参数设置
    double balance = 1.0;
    cv::Size new_image_size = image_size;
    cv::Mat new_camera_matrix = cv::Mat();

    CameraParams params;
    params.K = k;
    params.dist_coeffs = dist_coeffs;
    params.ori_image_size = image_size;
    params.undistorted_image_size = new_image_size;
    params.balance = balance;
    params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/frontlong_3840_2048_nv12.yuv";
    params.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontlong_3840_2048_nv12_bgr.jpg";
    params.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontlong_3840_2048_nv12_undistorted.jpg";

    fisheye_undistor_sequnce(params);
}

void sfl_fisheye_undistor_sequnce() {
    // fl相机内参矩阵
    cv::Mat k = (cv::Mat_<double>(3, 3) << 526.563, 0, 478.093
                                            , 0, 526.652, 255.934
                                            , 0, 0, 1);
    // fl相机的畸变系数
    std::vector<double> dist_coeffs = {-0.0128233, -0.0098962, 0.0130589, -0.00157833};
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // fl相机的分辨率
    int width = 960;
    int height = 512;
    cv::Size image_size(width, height);

    // 参数设置
    double balance = 1.0;
    cv::Size new_image_size = image_size;
    cv::Mat new_camera_matrix = cv::Mat();

    CameraParams params;
    params.K = k;
    params.dist_coeffs = dist_coeffs;
    params.ori_image_size = image_size;
    params.undistorted_image_size = new_image_size;
    params.balance = balance;
    params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/sideleftfront_960_512_nv12.yuv";
    params.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/sideleftfront_960_512_nv12_bgr.jpg";
    params.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/sideleftfront_960_512_nv12_undistorted.jpg";

    fisheye_undistor_sequnce(params); 
}

/**
 * 有一点反直觉，distortPoints函数是负责：虚拟相机的归一化坐标 -> 真实鱼眼图的像素坐标，用来得到去畸变的图像的
 */
void fisheye_distort_point() {
    // 1.畸变图像
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_bgr.jpg";
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if(input_image.empty()) {
        std::cerr << "Could not read input image: " << input_image_path << std::endl;
        return;
    }
    std::cout << "Input image path: " << input_image_path << std::endl;
    std::cout << "Input image size: " << input_image.size() << ", channels: " << input_image.channels() << std::endl;

    if(input_image.size() != cv::Size(3840, 2048)) {
        std::cerr << "Input image size does not match expected size: " << input_image.size() << std::endl;
        return;
    }

    // FW相机内参矩阵
    cv::Mat k = (cv::Mat_<double>(3, 3) << 1906.6, 0, 1923.26
                                            , 0, 1906.18, 1022.45
                                            , 0, 0, 1);
    // FW相机的畸变系数
    std::vector<double> dist_coeffs = {-0.0299548, -0.00364585, -0.00155829, 0.00104736};
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // FW相机的分辨率
    int width = 3840;
    int height = 2048;
    // 针孔虚拟相机的内参矩阵
    cv::Mat new_camera_matrix = (cv::Mat_<double>(3, 3) << 1108.406286393554, 0, 1925.365853013935
                                            , 0, 1108.162118429489, 1022.594463401068
                                            , 0, 0, 1);
    double fx_new = new_camera_matrix.at<double>(0, 0);
    double fy_new = new_camera_matrix.at<double>(1, 1);
    double cx_new = new_camera_matrix.at<double>(0, 2);
    double cy_new = new_camera_matrix.at<double>(1, 2);

    // 2.去即便图像归一化的点
    std::vector<cv::Point2f> undistorted_norm_points;
    for(int h = 0; h < input_image.rows; h++) {
        for(int w = 0; w < input_image.cols; w++) {
            undistorted_norm_points.emplace_back((w - cx_new) / fx_new, (h - cy_new) / fy_new);
        }
    }

    // 3. 得到畸变图像上点跟去畸变图像上点的映射关系
    std::vector<cv::Point2f> distorted_points(undistorted_norm_points.size());
    /**
     * @brief 
     * @param undistorted 输入无畸变图像的坐标位置(需要先归一化，即减去主点坐标并除以焦距)，可以是一个点或者一组点
     * @param distorted 输出对应的畸变图像的坐标位置，和输入点一一对应
     * @param K 输入的相机内参矩阵
     * @param D 输入的畸变系数
     */
    cv::fisheye::distortPoints(undistorted_norm_points, distorted_points, k, dist_coeffs_mat);
    
    // 4. 写映射矩阵
    cv::Mat mapx;
    cv::Mat mapy;
    mapx.create(cv::Size(3840, 2048), CV_32F);
    mapy.create(cv::Size(3840, 2048), CV_32F);
    for(int v = 0; v < 2048; v++) {
        for(int u = 0; u < 3840; u++) {
            const cv::Point2f& distort_point = distorted_points[v * 3840 + u];
            mapx.at<float>(v, u) = distort_point.x;
            mapy.at<float>(v, u) = distort_point.y;
        }
    }

    // 5. remap
    cv::Mat distorted_image = cv::Mat::zeros(input_image.size(), input_image.type());
    cv::remap(input_image, distorted_image, mapx, mapy, cv::INTER_LINEAR);
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_fisheye_distort_point.jpg";
    cv::imwrite(output_image_path, distorted_image);
    std::cout << "Saved distorted image to: " << output_image_path << std::endl;

    // 6. not remap
    cv::Mat distorted_image_2 = cv::Mat::zeros(input_image.size(), input_image.type());
    for(int v = 0; v < 2048; v++) {
        for(int u = 0; u < 3840; u++) {
            cv::Point2f distort_point = distorted_points[v * 3840 + u];
            int distort_point_x = static_cast<int>(distort_point.x);
            int distort_point_y = static_cast<int>(distort_point.y);
            if(distort_point_x >= 0 && distort_point_x < 3840 && distort_point_y >= 0 && distort_point_y < 2048) {
                distorted_image_2.at<cv::Vec3b>(v, u) = input_image.at<cv::Vec3b>(distort_point_y, distort_point_x);
            }
        }
    }
    std::string output_image_path_2 = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_fisheye_distort_point_2.jpg";
    cv::imwrite(output_image_path_2, distorted_image_2);
    std::cout << "Saved distorted image to: " << output_image_path_2 << std::endl;
}

void fisheye_undistort() {
    // FW相机内参矩阵
    cv::Mat k = (cv::Mat_<double>(3, 3) << 1906.6, 0, 1923.26
                                            , 0, 1906.18, 1022.45
                                            , 0, 0, 1);
    // FW相机的畸变系数
    std::vector<double> dist_coeffs = {-0.0299548, -0.00364585, -0.00155829, 0.00104736};
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // FW相机的分辨率
    int width = 3840;
    int height = 2048;
    cv::Size image_size(width, height);

    // FW图像路径
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/frontwide_3840_2048_nv12.yuv";
    std::ifstream input_image_stream(input_image_path, std::ios::binary | std::ios::ate);
    if(!input_image_stream.is_open()) {
        std::cerr << "Could not open input image file: " << input_image_path << std::endl;
        return;
    }
    std::streamsize input_image_size = input_image_stream.tellg();
    input_image_stream.seekg(0, std::ios::beg);
    if(input_image_size != image_size.width * image_size.height * 3 / 2) {
        std::cerr << "Input image size does not match expected size: " << input_image_size << " bytes." << std::endl;
        return;
    }
    std::vector<unsigned char> input_image_data(input_image_size);
    if(!input_image_stream.read(reinterpret_cast<char*>(input_image_data.data()), input_image_size)) {
        std::cerr << "Error reading input image file: " << input_image_path << std::endl;
        return;
    }
    cv::Mat input_image(image_size.height + image_size.height / 2, image_size.width, CV_8UC1, input_image_data.data());
    cv::Mat input_image_bgr;
    cv::cvtColor(input_image, input_image_bgr, cv::COLOR_YUV2BGR_NV12); // NV12格式转换为BGR格式

    cv::Mat new_camera_matrix = cv::Mat();
    double balance = 1.0;

    // 1.得到新内参
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(k, dist_coeffs_mat, image_size, cv::Mat(), new_camera_matrix, balance, image_size, 1.0);
    cv::Mat dst;

    // 2.去畸变
    /**
     * @brief 去畸变
     * @param distorted 带畸变的图像数据
     * @param undistorted 去即便的图像数据
     * @param K
     * @param D
     * @param KNew
     * @param new_size
     */
    cv::fisheye::undistortImage(input_image_bgr, dst, k, dist_coeffs_mat, new_camera_matrix, image_size);
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_undistorted_2.jpg";
    cv::imwrite(output_image_path, dst);
    std::cout << "Saved undistorted image to: " << output_image_path << std::endl;
}

/**
 * @brief undistortPoints函数是用于给图片加畸变的
 */
void fisheye_undistort_point() {
    // 1. 去畸变的图
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_undistorted.jpg";
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if(input_image.empty()) {
        std::cerr << "Could not read input image: " << input_image_path << std::endl;
        return;
    }
    std::cout << "Input image path: " << input_image_path << std::endl;
    std::cout << "Input image size: " << input_image.size() << ", channels: " << input_image.channels() << std::endl;

    if(input_image.size() != cv::Size(3840, 2048)) {
        std::cerr << "Input image size does not match expected size: " << input_image.size() << std::endl;
        return;
    }

    // FW相机内参矩阵
    cv::Mat k = (cv::Mat_<double>(3, 3) << 1906.6, 0, 1923.26
                                            , 0, 1906.18, 1022.45
                                            , 0, 0, 1);
    // FW相机的畸变系数
    std::vector<double> dist_coeffs = {-0.0299548, -0.00364585, -0.00155829, 0.00104736};
    cv::Mat dist_coeffs_mat = cv::Mat(dist_coeffs); // 将dist_coeffs转换为Mat对象
    // FW相机的分辨率
    int width = 3840;
    int height = 2048;
    // FW 去畸变图像对应的内参矩阵
    cv::Mat new_camera_matrix = (cv::Mat_<double>(3, 3) << 1108.406286393554, 0, 1925.365853013935
                                            , 0, 1108.162118429489, 1022.594463401068
                                            , 0, 0, 1);
    double fx_new = new_camera_matrix.at<double>(0, 0);
    double fy_new = new_camera_matrix.at<double>(1, 1);
    double cx_new = new_camera_matrix.at<double>(0, 2);
    double cy_new = new_camera_matrix.at<double>(1, 2);

    // 2.鱼眼图的像素坐标
    std::vector<cv::Point2f> norm_points;
    for(int h = 0; h < input_image.rows; h++) {
        for(int w = 0; w < input_image.cols; w++) {
            norm_points.emplace_back(w, h);
        }
    }

    // 3.得到图像映射关系
    std::vector<cv::Point2f> undistorted_points(norm_points.size());
    /**
     * @brief 基于鱼眼模型进行去畸变
     * @param distorted 输入点，鱼眼图的像素坐标
     * @param undistorted 输出点
     * @param K 
     * @param D
     * @param R 默认noArray()
     * @param P 新相机矩阵，默认noArray()。虚拟针孔相机的内参。作用于输出端
     */
    cv::fisheye::undistortPoints(norm_points, undistorted_points, k, dist_coeffs_mat, cv::Matx33d::eye(), new_camera_matrix);
    
    // 4. 写映射矩阵
    cv::Mat mapx;
    cv::Mat mapy;
    mapx.create(cv::Size(3840, 2048), CV_32F);
    mapy.create(cv::Size(3840, 2048), CV_32F);
    for(int v = 0; v < 2048; v++) {
        for(int u = 0; u < 3840; u++) {
            const cv::Point2f& p = undistorted_points[v * 3840 + u];
            mapx.at<float>(v, u) = p.x;
            mapy.at<float>(v, u) = p.y;
        }
    }

    // 5. remap
    cv::Mat distorted_image = cv::Mat::zeros(input_image.size(), input_image.type());
    cv::remap(input_image, distorted_image, mapx, mapy, cv::INTER_LINEAR);
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_fisheye_undistort_point.jpg";
    cv::imwrite(output_image_path, distorted_image);
    std::cout << "Saved distorted image to: " << output_image_path << std::endl;
}

int main() {
    std::cout << "======================== infrared_pinhole_undistor_sequnce ========================" << std::endl;
    infrared_pinhole_undistor_sequnce();
    std::cout << "======================== fw_fisheye_undistor_sequnce ========================" << std::endl;
    fw_fisheye_undistor_sequnce();
    std::cout << "======================== fl_fisheye_undistor_sequnce ========================" << std::endl;
    fl_fisheye_undistor_sequnce();
    std::cout << "======================== sfl_fisheye_undistor_sequnce ========================" << std::endl;
    sfl_fisheye_undistor_sequnce();
    std::cout << "======================== fisheye_distort_point ========================" << std::endl;
    fisheye_distort_point();
    std::cout << "======================== fisheye_undistort ========================" << std::endl;
    fisheye_undistort();
    std::cout << "======================== fisheye_undistort_point ========================" << std::endl;
    fisheye_undistort_point();

    return 0;
}