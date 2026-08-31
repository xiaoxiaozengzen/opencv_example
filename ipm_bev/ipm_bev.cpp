#include <cstddef>
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

struct SurroundCameraParams {
    cv::Size ori_image_size; // 原始图像尺寸
    cv::Size undistorted_image_size; // 去畸变后图像尺寸
    double fov = 0.0; // fov视场角度
    std::vector<double> affine;
    std::vector<double> ray2pixel;
    std::vector<double> distort_center;
    std::string input_image_path = ""; // 输入图像路径
    std::string output_input_image_path = ""; // 输入图像保存路径
    std::string output_undistorted_image_path = ""; // 去畸变后图像保存路径
};

cv::Point2f GetOCamSrcPoint(const double& u_norm, const double& v_norm, const SurroundCameraParams& param) {
    double r = std::sqrt(u_norm * u_norm + v_norm * v_norm);
    if(r < 1e-12) {
        // 落在光轴
        return cv::Point2f(static_cast<float>(param.distort_center[0]), static_cast<float>(param.distort_center[1]));
    }

    // 求光线跟光轴的夹角
    double theta = std::atan(r);

    /**
     * @brief OCam多项式的角度从xy平面量起来，且光轴方向是负数
     * phi(光线跟光轴夹角)  -  OCAM对应的角度
     *  0°                     -90°
     *  30°                    -60°
     *  90°                    0°
     *  95°                    +5°
     * @note 因为OCam最初是给折返相机设计的，即允许入射角度超过90度。在角度从小于90度穿过大于90°时，tan的值会剧烈变化
     *       因此调整到xy平面，就可以从-90°到+90°，tan线性变化
     */
    double thetad = theta - CV_PI/2.0;

    double rho = 0.0;
    for(int i = param.ray2pixel.size() - 1; i >= 0 ; i--) {
        rho = rho * thetad + param.ray2pixel.at(i);
    }

    double u_distort = u_norm / r * rho;
    double v_distort = v_norm / r * rho;
    return cv::Point2f(
        static_cast<float>(u_distort * param.affine[0] + v_distort * param.affine[1] + param.distort_center[0]),
        static_cast<float>(u_distort * param.affine[2] + v_distort * param.affine[3] + param.distort_center[1])
    );
}

struct BevParams {
    double round_bev_x_dis;
    double round_bev_y_dis;
    double single_pixel_dis;
    double shift_center_x;
    double shift_center_y;
    std::vector<double> world2cam_matrix_arr;
    std::string input_image_path = "";
    std::string output_image_path = "";
};

void ipm_bev_surround(const BevParams& bev_param, const SurroundCameraParams& surround_param) {
    // 1.定义BEV平面
    double round_bev_x_dis = bev_param.round_bev_x_dis;
    double round_bev_y_dis = bev_param.round_bev_y_dis;
    double single_pixel_dis = bev_param.single_pixel_dis;
    int bev_width = static_cast<int>(round_bev_x_dis/single_pixel_dis);
    int bev_height = static_cast<int>(round_bev_y_dis/single_pixel_dis);
    cv::Size bev_size(bev_width, bev_height);

    // 偏移，车辆后轴中心在屏幕的正中央
    double shift_center_x = bev_param.shift_center_x;
    double shift_center_y = bev_param.shift_center_y;
    int mid_x_dis = round_bev_x_dis * 0.5 + shift_center_x;
    int mid_y_dis = round_bev_y_dis * 0.5 + shift_center_y;

    // 2.world2cam
    double world2cam_matrix_arr[16];
    std::memcpy(world2cam_matrix_arr, reinterpret_cast<void*>(const_cast<double*>(bev_param.world2cam_matrix_arr.data())), bev_param.world2cam_matrix_arr.size() * sizeof(double));
    cv::Mat world2cam_matrix(4, 4, CV_64F, world2cam_matrix_arr);
    std::vector<std::array<double, 3>> world_points(bev_size.width * bev_size.height);
    std::vector<std::array<double, 3>> cam_points(bev_size.width * bev_size.height);
    std::vector<std::array<double, 3>> norm_points(bev_size.width * bev_size.height);
    std::vector<bool> can_get_src(bev_size.width * bev_size.height, false);
    for(int v = 0; v < bev_size.height; v++) {
        for(int u = 0; u < bev_size.width; u++) {
            world_points[v*bev_size.height + u] = {
                -v * single_pixel_dis + mid_y_dis * 1.0,
                -u * single_pixel_dis + mid_x_dis * 1.0,
                0.0
            };

            cv::Mat world_points_mat(4, 1, CV_64F);
            world_points_mat.at<double>(0) = world_points[v*bev_size.height + u].at(0);
            world_points_mat.at<double>(1) = world_points[v*bev_size.height + u].at(1);
            world_points_mat.at<double>(2) = world_points[v*bev_size.height + u].at(2);
            world_points_mat.at<double>(3) = 1.0;

            cv::Mat cam_points_mat = world2cam_matrix * world_points_mat;
            cam_points[v*bev_size.height + u] = {cam_points_mat.at<double>(0), cam_points_mat.at<double>(1), cam_points_mat.at<double>(2)};

            if(cam_points[v*bev_size.height + u].at(2) < 0) {
                norm_points[v*bev_size.height + u] = {0, 0, -1.0};
            } else {
                norm_points[v*bev_size.height + u] = {
                    cam_points[v*bev_size.height + u].at(0) / cam_points[v*bev_size.height + u].at(2),
                    cam_points[v*bev_size.height + u].at(1) / cam_points[v*bev_size.height + u].at(2),
                    1.0
                };
                can_get_src[v*bev_size.height + u] = true;
            }
        }
    }

    // 3.基于norm图像，获取对应位置畸变图像像素坐标
    cv::Mat mapx(bev_size, CV_32F);
    cv::Mat mapy(bev_size, CV_32F);
    for(int v = 0; v < bev_size.height; v++) {
        for(int u = 0; u < bev_size.width; u++) {
            if(can_get_src[v*bev_size.height + u] == true) {
                cv::Point2f src_point = GetOCamSrcPoint(norm_points[v*bev_size.height + u].at(0), norm_points[v*bev_size.height + u].at(1), surround_param);
                mapx.at<float>(v, u) = src_point.x;
                mapy.at<float>(v, u) = src_point.y;
            } else {
                mapx.at<float>(v, u) = 100000;
                mapx.at<float>(v, u) = 100000;
            }
        }
    }

    // 4.开始进行映射
    std::string input_image_path = bev_param.input_image_path;
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if(input_image.empty()) {
        std::cerr << "Could not read input image: " << input_image_path << std::endl;
        return;
    }
    std::cout << "Input image path: " << input_image_path << std::endl;
    std::cout << "Input image size: " << input_image.size() << ", channels: " << input_image.channels() << std::endl;

    if(input_image.size() != surround_param.ori_image_size) {
        std::cerr << "Input image size does not match expected size: " << input_image.size() << std::endl;
        return;
    }
    cv::Mat undistorted_image;
    cv::remap(input_image, undistorted_image, mapx, mapy, cv::INTER_LINEAR);
    std::string output_image_path = bev_param.output_image_path;
    cv::imwrite(output_image_path, undistorted_image);
    std::cout << "Saved undistorted image to: " << output_image_path << std::endl;
}

void ipm_bev_front() {
    struct SurroundCameraParams param;
    param.ori_image_size = cv::Size(1920, 1536);
    param.undistorted_image_size = cv::Size(1920, 1536);
    param.fov = 140.0;
    param.affine = std::vector<double>{1.00032, 0.000233026, 0.000178776, 1};
    param.ray2pixel = std::vector<double>{815.031, 603.915, 87.5408, 85.5428, 64.9007, 18.26, 16.3885, 11.3528, 1.8612, 8.67193, 7.59118, -3.17747, -6.09492, -2.58876, -0.371392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    param.distort_center = std::vector<double>{954.717, 770.999};
    param.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/surroundfront_1920x1536_nv12.yuv";
    param.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundfront_1920x1536_nv12_bgr.jpg";
    param.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundfront_1920x1536_nv12_undistort.jpg";

    struct BevParams bev_params;
    bev_params.round_bev_x_dis = 16.0;
    bev_params.round_bev_y_dis = 16.0;
    bev_params.single_pixel_dis = 0.02;
    bev_params.shift_center_x = 0.0;
    bev_params.shift_center_y = 0.8;
    bev_params.world2cam_matrix_arr = {
        -0.0330839, -0.999415, -0.00874261, 0.130036, 
        -0.424998, 0.0219843, -0.904928, 2.15572, 
        0.90459, -0.0262226, -0.425476, -3.11181, 
        0, 0, 0, 1
    };
    bev_params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundfront_1920x1536_nv12_bgr.jpg";
    bev_params.output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundfront_1920x1536_nv12_bev_bgr.jpg";

    ipm_bev_surround(bev_params, param);
}

void ipm_bev_left() {
    struct SurroundCameraParams param;
    param.ori_image_size = cv::Size(1920, 1536);
    param.undistorted_image_size = cv::Size(1920, 1536);
    param.fov = 140.0;
    param.affine = std::vector<double>{1.00019, -0.000156714, -0.0000130094, 1};
    param.ray2pixel = std::vector<double>{811.112, 602.894, 92.681, 89.5724, 65.9754, 19.982, 17.6316, 11.3648, 2.51892, 9.23328, 7.35395, -3.15194, -5.66501, -2.32672, -0.323997};
    param.distort_center = std::vector<double>{957.83, 767.21};
    param.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/surroundleft_1920x1536_nv12.yuv";
    param.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundleft_1920x1536_nv12_bgr.jpg";
    param.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundleft_1920x1536_nv12_undistort.jpg";

    struct BevParams bev_params;
    bev_params.round_bev_x_dis = 16.0;
    bev_params.round_bev_y_dis = 16.0;
    bev_params.single_pixel_dis = 0.02;
    bev_params.shift_center_x = 0.0;
    bev_params.shift_center_y = 0.8;
    bev_params.world2cam_matrix_arr = {
        0.999966, 0.00661366, -0.00492463, -2.04999, 
        -0.000335581, -0.564096, -0.825709, 1.30535, 
        -0.00823893, 0.825682, -0.564074, -0.278161, 
        0, 0, 0, 1
    };
    bev_params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundleft_1920x1536_nv12_bgr.jpg";
    bev_params.output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundleft_1920x1536_nv12_bev_bgr.jpg";

    ipm_bev_surround(bev_params, param);
}

void ipm_bev_rear() {
    struct SurroundCameraParams param;
    param.ori_image_size = cv::Size(1920, 1536);
    param.undistorted_image_size = cv::Size(1920, 1536);
    param.fov = 140.0;
    param.affine = std::vector<double>{1.00088, 0.000204419, 0.000193557, 1};
    param.ray2pixel = std::vector<double>{813.391, 603.85, 92.4195, 89.931, 66.0683, 19.9283, 17.6774, 11.3558, 2.79006, 9.61182, 7.16409, -3.77732, -6.12483, -2.47658, -0.342968};
    param.distort_center = std::vector<double>{957.593, 767.195};
    param.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/surroundrear_1920x1536_nv12.yuv";
    param.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundrear_1920x1536_nv12_bgr.jpg";
    param.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundrear_1920x1536_nv12_undistort.jpg";

    struct BevParams bev_params;
    bev_params.round_bev_x_dis = 16.0;
    bev_params.round_bev_y_dis = 16.0;
    bev_params.single_pixel_dis = 0.02;
    bev_params.shift_center_x = 0.0;
    bev_params.shift_center_y = 0.8;
    bev_params.world2cam_matrix_arr = {
        0.00172154, 0.999962, 0.00852483, 0.0121369, 
        0.550779, 0.00616723, -0.834628, 1.28575, 
        -0.834649, 0.00613209, -0.550748, -0.359723, 
        0, 0, 0, 1
    };
    bev_params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundrear_1920x1536_nv12_bgr.jpg";
    bev_params.output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundrear_1920x1536_nv12_bev_bgr.jpg";

    ipm_bev_surround(bev_params, param);
}

void ipm_bev_right() {
    struct SurroundCameraParams param;
    param.ori_image_size = cv::Size(1920, 1536);
    param.undistorted_image_size = cv::Size(1920, 1536);
    param.fov = 140.0;
    param.affine = std::vector<double>{0.998986, -0.000496911, -0.000412609, 1};
    param.ray2pixel = std::vector<double>{815.401, 606.012, 93.6462, 90.9017, 66.6304, 20.288, 17.96, 11.4538, 2.92051, 9.79934, 7.19289, -3.8655, -6.18249, -2.48868, -0.343486};
    param.distort_center = std::vector<double>{962.225, 768.768};
    param.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/car/surroundright_1920x1536_nv12.yuv";
    param.output_input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundright_1920x1536_nv12_bgr.jpg";
    param.output_undistorted_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundright_1920x1536_nv12_undistort.jpg";

    struct BevParams bev_params;
    bev_params.round_bev_x_dis = 16.0;
    bev_params.round_bev_y_dis = 16.0;
    bev_params.single_pixel_dis = 0.02;
    bev_params.shift_center_x = 0.0;
    bev_params.shift_center_y = 0.8;
    bev_params.world2cam_matrix_arr = {
        -0.99979, -0.00641023, -0.0194407, 2.05136, 
        0.0121527, 0.578361, -0.81569, 1.28018, 
        0.0164725, -0.815755, -0.578162, -0.298589, 
        0, 0, 0, 1
    };
    bev_params.input_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundright_1920x1536_nv12_bgr.jpg";
    bev_params.output_image_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/surroundright_1920x1536_nv12_bev_bgr.jpg";

    ipm_bev_surround(bev_params, param);
}

int main() {
    std::cout << "======================== ipm_bev_front ========================" << std::endl;
    ipm_bev_front();
    std::cout << "======================== ipm_bev_left ========================" << std::endl;
    ipm_bev_left();
    std::cout << "======================== ipm_bev_rear ========================" << std::endl;
    ipm_bev_rear();
    std::cout << "======================== ipm_bev_right ========================" << std::endl;
    ipm_bev_right();

    return 0;
}