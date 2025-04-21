/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

// 检测结果
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

struct DL_INIT_PARAM {
    float rectConfidenceThreshold;
    float iouThreshold;
    std::string modelPath;
    std::vector<int> imgSize;
    bool cudaEnable;
};

class CudaDetector {
public:
    char* CreateSession(const DL_INIT_PARAM& params);
};

//! \brief  The SampleOnnxYOLOv8 class implements the ONNX YOLOv8 sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxYOLOv8
{
public:
    SampleOnnxYOLOv8(const DL_INIT_PARAM& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(const cv::Mat& input_image);

private:
    DL_INIT_PARAM mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for YOLOv8 and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& input_image);

    //!
    //! \brief Classifies digits and verify result
    //!
    std::vector<Detection> postprocess(float* output, int output_size, int original_w, int original_h);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx YOLOv8 network by parsing the Onnx model and builds
//!          the engine that will be used to run YOLOv8 (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleOnnxYOLOv8::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto timingCache = SampleUniquePtr<nvinfer1::ITimingCache>();

    auto constructed = constructNetwork(builder, network, config, parser, timingCache);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx YOLOv8 Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx YOLOv8 network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxYOLOv8::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache)
{
    auto parsed = parser->parseFromFile(mParams.modelPath.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.cudaEnable) {
        // 可以在这里添加更多 CUDA 相关的配置
    }

    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxYOLOv8::infer(const cv::Mat& input_image)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    if (!processInput(buffers, input_image))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.modelPath));
    const int output_size = mOutputDims.d[1] * mOutputDims.d[2];
    std::vector<Detection> detections = postprocess(output, output_size, input_image.cols, input_image.rows);

    for (const auto& det : detections) {
        std::cout << "Class: " << det.class_id << ", Confidence: " << det.confidence
                  << ", Bbox: (" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << ")" << std::endl;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxYOLOv8::processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& input_image)
{
    const int inputH = mParams.imgSize[1];
    const int inputW = mParams.imgSize[0];

    // 1. 图像填充（Letterbox）
    cv::Mat letterbox_img;
    cv::Size target_size(inputW, inputH);
    float ratio = std::min((float)target_size.width / input_image.cols, (float)target_size.height / input_image.rows);
    int new_w = static_cast<int>(input_image.cols * ratio);
    int new_h = static_cast<int>(input_image.rows * ratio);
    cv::resize(input_image, letterbox_img, cv::Size(new_w, new_h));

    // 2. 填充至目标尺寸
    cv::Mat padded_img = cv::Mat::zeros(target_size.height, target_size.width, CV_8UC3);
    letterbox_img.copyTo(padded_img(cv::Rect(0, 0, new_w, new_h)));

    // 3. BGR转RGB并归一化,这是mnist单通道灰度化没有的，要变通
    cv::Mat rgb_img;
    cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32F, 1.0 / 255.0);  // 归一化至 [0, 1]

    // 4. 转换为 NCHW 格式（TensorRT 输入格式）
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.modelPath));
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < inputH; ++h) {
            for (int w = 0; w < inputW; ++w) {
                hostDataBuffer[c * inputH * inputW + h * inputW + w] =
                    rgb_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return true;
}

//!
//! \brief 后处理函数：解析输出、NMS 筛选
//!
std::vector<Detection> SampleOnnxYOLOv8::postprocess(float* output, int output_size, int original_w, int original_h)
{
    const int num_classes = 80;  // YOLOv8 默认类别数
    const int stride = 4 + 1 + num_classes;  // 4坐标 + 1置信度 + num_classes类别概率

    std::vector<Detection> detections;

    for (int i = 0; i < output_size; i += stride) {
        float x1 = output[i];
        float y1 = output[i + 1];
        float x2 = output[i + 2];
        float y2 = output[i + 3];
        float obj_conf = output[i + 4];

        // 提取类别概率并计算置信度
        float max_class_conf = 0;
        int class_id = 0;
        for (int c = 0; c < num_classes; ++c) {
            if (output[i + 5 + c] > max_class_conf) {
                max_class_conf = output[i + 5 + c];
                class_id = c;
            }
        }
        float confidence = obj_conf * max_class_conf;

        // 过滤低置信度检测
        if (confidence < mParams.rectConfidenceThreshold) continue;

        // 反归一化坐标（假设输入图像经过 Letterbox 填充）
        float ratio = std::min((float)mParams.imgSize[0] / original_w, (float)mParams.imgSize[1] / original_h);
        int pad_w = mParams.imgSize[0] - static_cast<int>(original_w * ratio);
        int pad_h = mParams.imgSize[1] - static_cast<int>(original_h * ratio);
        x1 = (x1 - pad_w / 2) / ratio;
        y1 = (y1 - pad_h / 2) / ratio;
        x2 = (x2 - pad_w / 2) / ratio;
        y2 = (y2 - pad_h / 2) / ratio;

        // 限制坐标范围
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        x2 = std::min((float)original_w, x2);
        y2 = std::min((float)original_h, y2);

        detections.push_back({x1, y1, x2, y2, confidence, class_id});
    }

    // 非极大值抑制（NMS）
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto& det : detections) {
        boxes.emplace_back(cv::Rect(cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2)));
        scores.push_back(det.confidence);
    }
    cv::dnn::NMSBoxes(boxes, scores, mParams.rectConfidenceThreshold, mParams.iouThreshold, indices);

    std::vector<Detection> results;
    for (size_t i = 0; i < indices.size(); ++i) {
        results.push_back(detections[indices[i]]);
    }

    return results;
}


class ArmorDetectorNode : public rclcpp::Node
{
public:
    ArmorDetectorNode() : Node("armor_detector_node")
    {
        auto pkg_path = ament_index_cpp::get_package_share_directory("rm_armor_detector");
        DL_INIT_PARAM params;
        params.rectConfidenceThreshold = 0.1;
        params.iouThreshold = 0.5;
        params.modelPath = pkg_path + "/model/four_points_armor/armor.onnx";
        params.imgSize = { 640, 640 };
        params.cudaEnable = true;

        RCLCPP_INFO(this->get_logger(), "Cuda detect mode!");
        detector_ = std::make_shared<SampleOnnxYOLOv8>(params);
        if (!detector_->build())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to build TensorRT engine!");
        }
    }

    void detect(const cv::Mat& input_image)
    {
        if (!detector_->infer(input_image))
        {
            RCLCPP_ERROR(this->get_logger(), "Inference failed!");
        }
    }

private:
    std::shared_ptr<SampleOnnxYOLOv8> detector_;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArmorDetectorNode>();

    cv::Mat input_image = cv::imread("test.jpg");
    if (input_image.empty()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to read input image!");
        return -1;
    }

    node->detect(input_image);

    rclcpp::shutdown();
    return 0;
}    

