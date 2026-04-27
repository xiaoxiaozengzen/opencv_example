#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

void nchw2yuv() {
    // 设置输入文件路径和图像参数
    std::string bin_file = "/mnt/workspace/cgz_workspace/Exercise/images.bin";
    int batch_size = 4;
    int channel_num = 3;
    int width = 1920;
    int height = 1536;
    int image_size = channel_num * width * height; // 每张图像的大小
    int total_size = batch_size * image_size; // 总的大小
    int total_size_bytes = total_size * sizeof(float); // 总的字节大小

    // 读取文件
    std::ifstream ifs(bin_file, std::ios::binary | std::ios::ate);
    if(!ifs.is_open()) {
        std::cout << "Failed to open file: " << bin_file << std::endl;
        return;
    }
    std::streamsize size = ifs.tellg();
    if(size != total_size_bytes) {
        std::cout << "File size does not match expected size. File size: " << size << " bytes, expected size: " << total_size_bytes << " bytes" << std::endl;
        return;
    }
    ifs.seekg(0, std::ios::beg);
    
    std::vector<std::string> camera_names = {"SurroundFront", "SurroundLeft", "SurroundRear", "SurroundRight"};

    for(int i = 0; i < batch_size; i++) {
        std::cout << "Processing camera: " << camera_names[i] << std::endl;
        // 读取单个图像数据
        std::vector<float> surround_data(image_size);
        ifs.seekg(i * image_size * sizeof(float), std::ios::beg);
        ifs.read(reinterpret_cast<char*>(surround_data.data()), image_size * sizeof(float));

        // 将NCHW格式的数据转换为HWC格式，并且将数据类型从float转换为uint8_t
        cv::Mat surround_data_mat(height, width, CV_32FC3, surround_data.data());

        // 拆分通道
        // 上步骤得到的图像RG B通道的数据是连续存储的，即先存储R通道的所有像素值，再存储G通道的所有像素值，最后存储B通道的所有像素值。
        // 我们需要将这些数据拆分成三个单独的通道，然后再合并成HWC格式。因为opencv是RGBRGB...的存储方式，所以我们需要按照R、G、B的顺序来拆分通道。
        std::vector<cv::Mat> channels;
        for(int c = 0; c < 3; c++) {
            cv::Mat channel(height, width, CV_32FC1, surround_data.data() + c * width * height);
            channels.push_back(channel);
        }

        // 合并为HWC
        cv::Mat surround_data_hwc;
        cv::merge(channels, surround_data_hwc);

        // 转为uint8_t
        cv::Mat surround_data_uint8;
        surround_data_hwc.convertTo(surround_data_uint8, CV_8UC3, 255.0); // 将float数据乘以255.0并转换为uint8_t

        // RGB转BRG
        cv::Mat surround_data_bgr;
        cv::cvtColor(surround_data_uint8, surround_data_bgr, cv::COLOR_RGB2BGR);

        // 保存为PNG格式
        std::string output_path = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/" + camera_names[i] + ".png";
        cv::imwrite(output_path, surround_data_bgr);
    }
}

int main() {
    std::cout << "===================nchw2yuv===================" << std::endl;
    nchw2yuv();

    return 0;
}