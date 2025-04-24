#pragma once
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "armor.hpp"
#include "NvInfer.h" // TensorRT API
#include "buffers.h"
#include "logger.h"

namespace rm_armor_detector {

struct DL_INIT_PARAM {
    std::string modelPath;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    bool cudaEnable;
    int logSeverityLevel;
    int detectColor; // -1 表示不指定颜色

    DL_INIT_PARAM()
        : imgSize({640, 640}),
          rectConfidenceThreshold(0.6f),
          iouThreshold(0.5f),
          cudaEnable(true),
          logSeverityLevel(3),
          detectColor(-1) {}
};

class TensorRTDetector {
public:
    TensorRTDetector(const DL_INIT_PARAM &params);
    virtual ~TensorRTDetector();

    bool build();
    bool infer(const cv::Mat &input_image, int detect_color = -1);
    std::vector<Armor> armors_;

private:
    DL_INIT_PARAM mParams;
    std::shared_ptr<nvinfer1::IRuntime> Runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> Engine;
    nvinfer1::Dims mInputDims, mOutputDims;

    // 预处理与后处理
    bool processInput(const samplesCommon::BufferManager &buffers, const cv::Mat &input_image);
    std::vector<Detection> postprocess(float *output, int output_size, int original_w, int original_h);
    void convertToArmor(const std::vector<Detection> &detections, int detect_color);

    // 辅助函数
    cv::Mat letterboxImage(const cv::Mat &src, const cv::Size &target_size);
};

} // namespace rm_armor_detector