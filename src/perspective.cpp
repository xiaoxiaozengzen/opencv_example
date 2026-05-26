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

/**
 * 图像的透视变换是指按照物体的成像投影规律进行变换。通过透视变换可以将图像投影到新的成像平面上。
 * 图像的透视变换通常用来解决相机的视线与物体所在平面不垂直的问题。
 * 常用的两个函数：warpPerspective()和getPerspectiveTransform()。
 */

int main() {
    std::string path_dir = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/";

    // 1. 自己生成一个图像
    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::rectangle(img, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(0, 255, 0), -1); // 绿色矩形
    cv::line(img, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 0, 0), 5); // 蓝色对角线
    cv::line(img, cv::Point(100, 300), cv::Point(300, 100), cv::Scalar(255, 0, 0), 5); // 蓝色对角线
    std::string original_path = path_dir + "perspective_original.jpg";
    cv::imwrite(original_path, img); // 保存原始图像
    std::cout << "Original image saved to: " << original_path << std::endl;

    // 2. 对其进行透视变换
    std::vector<cv::Point2f> src_points = {cv::Point2f(100, 100), cv::Point2f(300, 100), cv::Point2f(300, 300), cv::Point2f(100, 300)};
    std::vector<cv::Point2f> dst_points = {cv::Point2f(80, 120), cv::Point2f(320, 80), cv::Point2f(280, 320), cv::Point2f(120, 280)};
    /**
     * @brief 计算透视变换矩阵
     * @param src 输入图像中的四个点
     * @param dst 输出图像中的四个点
     * @param solveMethod 求解方法，传递给 cv2.solve求解变换矩阵。默认为0，表示使用最小二乘法求解
     *                    DECOMP_LU：0,选择最优轴的高斯消去法，默认方法
     *                    DECOMP_SVD：1,奇异值分解法
     *                    DECOMP_EIG：2,特征值分解法
     *                    DECOMP_QR：3,QR分解法
     * @return 3x3的透视变换矩阵
     */
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points); // 计算透视变换矩阵
    cv::Mat warped_img;

    /**
     * @brief 对图像进行透视变换
     * @param src 输入图像
     * @param dst 输出图像，与输入图像大小，类型相同
     * @param M 3x3的透视变换矩阵
     * @param dsize 输出图像的大小
     * @param flags 插值方法，默认为INTER_LINEAR
     * @param borderMode 边界模式，默认为BORDER_CONSTANT
     * @param borderValue 边界值，默认为0
     */
    cv::warpPerspective(img, warped_img, perspective_matrix, img.size()); // 应用透视变换
    std::string warped_path = path_dir + "perspective_warped.jpg";
    cv::imwrite(warped_path, warped_img); // 保存变换后的图像
    std::cout << "Warped image saved to: " << warped_path << std::endl;

    // 3. 对变换后的图像进行逆透视变换
    cv::Mat inverse_perspective_matrix = cv::getPerspectiveTransform(dst_points, src_points); // 计算逆透视变换矩阵
    cv::Mat unwarped_img;
    cv::warpPerspective(warped_img, unwarped_img, inverse_perspective_matrix, img.size()); // 应用逆透视变换
    std::string unwarped_path = path_dir + "perspective_unwarped.jpg";
    cv::imwrite(unwarped_path, unwarped_img); // 保存逆变换后的图像
    std::cout << "Unwarped image saved to: " << unwarped_path << std::endl;

    return 0;
}