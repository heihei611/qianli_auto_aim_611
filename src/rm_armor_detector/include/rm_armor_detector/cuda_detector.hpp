#pragma once

#define    USE_TENSORRT
#define    RET_OK nullptr

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

// 定义 TensorRT 相关的头文件
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

// 检测结果
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class TensorRTDetector
{
public:
    TensorRTDetector();

    char* CreateSession(DL_INIT_PARAM& iParams);

    void infer(const cv::Mat &input, int detect_color);

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales; // letterbox scale

    const float IMAGE_WIDTH_ = 640;
    const float IMAGE_HEIGHT_ = 640;
    const float CONFIDENCE_THRESHOLD_ = 0.8;
    const float SCORE_THRESHOLD_ = 0.9;
    const float COLOR_THRESHOLD_ = 0.9;
    const float NMS_THRESHOLD_ = 0.5;
    const std::vector<std::string> class_names_ = {"sentry", "1", "2", "3", "4", "5", "outpost", "base", "base_big"};
    std::vector<Armor> armors_;

private:
    DL_INIT_PARAM mParams;

    //!
    //! \brief Parses an ONNX model for YOLOv8 and creates a TensorRT network
    //!
    bool constructNetwork(samplesCommon::SampleUniquePtr<nvinfer1::IBuilder>& builder,
        samplesCommon::SampleUniquePtr<nvinfer1::INetworkDefinition>& network, samplesCommon::SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        samplesCommon::SampleUniquePtr<nvonnxparser::IParser>& parser, samplesCommon::SampleUniquePtr<nvinfer1::ITimingCache>& timingCache);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& input_image);

    //!
    //! \brief Classifies digits and verify result
    //!
    std::vector<Detection> postprocess(float* output, int output_size, int original_w, int original_h);
};
    