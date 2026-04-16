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
 * 1.针孔相机模型的畸变矫正：
 *      当前的getOptimalNewCameraMatrix/initUndistortRectifyMap/remap
 *      适用于常规透视相机模型(针孔模型)，不适用鱼眼
 * 2.鱼眼相机模型的畸变矫正：
 *      OpenCV提供了专门针对鱼眼相机模型的函数，
 *      如fisheye::estimateNewCameraMatrixForUndistortRectify和fisheye::initUndist
 *
 * resize会改变图像的尺寸：
 *  1.相机内参中的焦距和主点坐标也需要相应地进行缩放。fx和fy需要乘以水平和垂直的缩放因子，cx和cy也需要乘以相应的缩放因子。
 *  2.畸变系数通常不需要调整，为它们是归一化坐标下的参数，和分辨率无关。
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

void fw_pinhole_undistor_sequnce() {
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
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/frontwide_3840_2048_nv12.yuv";
    int input_image_width = 3840;
    int input_image_height = 2048;
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
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_3840_2048_nv12_pinhole_undistorted.jpg";
    cv::imwrite(output_image_path, undistorted_image_bgr);
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
    params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/frontwide_3840_2048_nv12.yuv";
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
    params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/frontlong_3840_2048_nv12.yuv";
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
    params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/sidefrontleft_960_512_nv12.yuv";
    params.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/sidefrontleft_960_512_nv12_bgr.jpg";
    params.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/sidefrontleft_960_512_nv12_undistorted.jpg";

    fisheye_undistor_sequnce(params); 
}

int main() {
    std::cout << "======================== fw_pinhole_undistor_sequnce ========================" << std::endl;
    fw_pinhole_undistor_sequnce();
    std::cout << "======================== fw_fisheye_undistor_sequnce ========================" << std::endl;
    fw_fisheye_undistor_sequnce();
    std::cout << "======================== fl_fisheye_undistor_sequnce ========================" << std::endl;
    fl_fisheye_undistor_sequnce();
    std::cout << "======================== sfl_fisheye_undistor_sequnce ========================" << std::endl;
    sfl_fisheye_undistor_sequnce();

    return 0;
}