#include <fstream>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>

int bayerCode(const std::string &bayer) {
    if (bayer == "RGGB") return cv::COLOR_BayerRG2BGR;
    if (bayer == "BGGR") return cv::COLOR_BayerBG2BGR;
    if (bayer == "GRBG") return cv::COLOR_BayerGR2BGR;
    if (bayer == "GBRG") return cv::COLOR_BayerGB2BGR;
    throw std::runtime_error("unknown bayer");
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: raw_to_jpeg <bayer>" << std::endl;
        return -1;
    }
    std::string bayer = argv[1];

    std::string inFile = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/image/PCB_1280x720_8u.raw";
    int W = 1280;
    int H = 720;
    std::string outFile = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/PCB_1280x720_8u.jpg";

    std::ifstream ifs(inFile, std::ios::binary);
    if (!ifs) { 
        std::cerr << "Cannot open input\n"; return -1; 
    }
    std::vector<uint8_t> buf(W * H);
    ifs.read(reinterpret_cast<char*>(buf.data()), buf.size()*sizeof(uint8_t));
    if (!ifs) { 
        std::cerr << "Read error or size mismatch\n"; return -1; 
    }

    cv::Mat raw(H, W, CV_8U, buf.data());
    
    // 1. demosaic
    cv::Mat bgr8;
    int code = bayerCode(bayer);
    cv::cvtColor(raw, bgr8, code);
    
    cv::imwrite(outFile, bgr8);
    std::cout << "Saved " << outFile << std::endl;

    return 0;
}