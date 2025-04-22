#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "armor.hpp"
#include "NvInfer.h"//api
#include "buffers.h"
#include "logger.h"

typedef struct _DL_INIT_PARAM {
    std::string modelPath;
    std::vector<int> imgSize = {640, 640};
    float rectConfidenceThreshold = 0.6f;
    float iouThreshold = 0.5f;
    bool cudaEnable = true;
    int logSeverityLevel = 3;  // 新增日志级别参数
} DL_INIT_PARAM;

class TensorRTDetector {
public:
    TensorRTDetector(const DL_INIT_PARAM& params);
    bool build();
    bool infer(const cv::Mat& input_image, int detect_color); // 新增 detect_color 参数
    std::vector<Armor> armors_;

private:
    DL_INIT_PARAM mParams;
    std::shared_ptr<IRuntime> mRuntime;
    std::shared_ptr<ICudaEngine> mEngine;
    nvinfer1::Dims mInputDims, mOutputDims;

    // 预处理与后处理
    bool processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& input_image);
    std::vector<Detection> postprocess(float* output, int output_size, int original_w, int original_h);
    void convertToArmor(const std::vector<Detection>& detections, int detect_color); // 新增颜色参数

    cv::Mat letterboxImage(const cv::Mat& src, const cv::Size& target_size);
};