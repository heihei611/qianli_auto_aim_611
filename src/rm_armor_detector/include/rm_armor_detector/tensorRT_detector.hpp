#pragma once

#define USE_TENSORRT
#define RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/imgproc/types_c.h>

// TensorRT 相关头文件
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "armor.hpp"
#pragma once

#define USE_TENSORRT
#define RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "armor.hpp"

// TensorRT 核心头文件
#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <cuda_runtime_api.h>

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

// 初始化参数结构体（统一声明）
typedef struct _DL_INIT_PARAM {
    std::string modelPath;
    std::vector<int> imgSize = {640, 640};
    float rectConfidenceThreshold = 0.6f;
    float iouThreshold = 0.5f;
    bool cudaEnable = true;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;

// 检测结果结构体（统一声明）
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class TensorRTDetector {
public:
    TensorRTDetector();
    ~TensorRTDetector();
    
    // 创建 TensorRT 会话
    char* CreateSession(DL_INIT_PARAM& params);
    // 执行推理
    void infer(const cv::Mat& input, int detect_color);
    
    // 检测结果
    std::vector<Armor> armors_;

private:
    DL_INIT_PARAM m_Params;
    std::shared_ptr<nvinfer1::IRuntime> m_Runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_Engine;
    nvinfer1::Dims m_InputDims;
    nvinfer1::Dims m_OutputDims;
    
    // TensorRT 网络构建
    bool buildEngine(DL_INIT_PARAM& params);
    bool constructNetwork(
        samplesCommon::SampleUniquePtr<nvinfer1::IBuilder>& builder,
        samplesCommon::SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
        samplesCommon::SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        samplesCommon::SampleUniquePtr<nvonnxparser::IParser>& parser
    );
    
    // 输入预处理
    bool processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& input_image);
    // 输出后处理
    std::vector<Detection> postprocess(float* output, int output_size, int original_w, int original_h);
    // 转换为 Armor 结构体
    void convertToArmor(const std::vector<Detection>& detections, int detect_color);
};
#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif


typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    bool cudaEnable = true;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;


struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class TensorRTDetector
{
public:
    TensorRTDetector();
    ~TensorRTDetector();
    char* CreateSession(DL_INIT_PARAM& iParams);
    void infer(const cv::Mat& input, int detect_color);
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    // 输入输出维度
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;

    const float IMAGE_WIDTH_ = 640;
    const float IMAGE_HEIGHT_ = 640;
    const float CONFIDENCE_THRESHOLD_ = 0.8;
    const float SCORE_THRESHOLD_ = 0.9;
    const float COLOR_THRESHOLD_ = 0.9;
    const float NMS_THRESHOLD_ = 0.5;
    const std::vector<std::string> class_names_ = { "sentry", "1", "2", "3", "4", "5", "outpost", "base", "base_big" };
    // 检测结果
    std::vector<Armor> armors_;

private:
    DL_INIT_PARAM mParams;

 
    bool constructNetwork(samplesCommon::SampleUniquePtr<nvinfer1::IBuilder>& builder,
        samplesCommon::SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
        samplesCommon::SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        samplesCommon::SampleUniquePtr<nvonnxparser::IParser>& parser,
        samplesCommon::SampleUniquePtr<nvinfer1::ITimingCache>& timingCache);

    // 处理输入图像
    bool processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& input_image);

    // 后处理输出结果
    std::vector<Detection> postprocess(float* output, int output_size, int original_w, int original_h);

    // 从 Detection 转换为 Armor
    void convertDetectionsToArmors(const std::vector<Detection>& detections, int detect_color);
};
    