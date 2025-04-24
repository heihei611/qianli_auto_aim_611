/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

 #include "tensorRT_detector.hpp" 
 #include "argsParser.h"
 #include "buffers.h"
 #include "common.h"
 #include "logger.h"
 #include "parserOnnxConfig.h"
 
 #include <cuda_runtime_api.h>//api
 #include <opencv2/opencv.hpp>
 #include <rclcpp/rclcpp.hpp>
 
 using namespace nvinfer1;
 using samplesCommon::SampleUniquePtr;//待定
 //使用线程锁来保护日志输出
 std::mutex log_mutex;
 void log(Severity severity, const char* msg) noexcept {
     if (severity <= Severity::kWARNING) {
         std::lock_guard<std::mutex> lock(log_mutex);
         std::cout << msg << std::endl;
     }
 }
 //  class Logger : public nvinfer1::ILogger {
 // 	void log(Severity severity, const char* msg) noexcept override {
 // 		// suppress info-level messages
// 		if (severity <= Severity::kWARNING)
// 			std::cout << msg << std::endl;
// 	}
// } logger;
 // The TensorRTDetector class implementation
//  bool TensorRTDetector::build() {
      auto builder = SampleUniquePtr<IBuilder>(createInferBuilder(sample::gLogger.getTRTLogger()));
//      auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
// // network definition
//      if (!builder) return false;
 
//      auto network = SampleUniquePtr<INetworkDefinition>(builder->createNetworkV2(0));
//      auto config = SampleUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
//      auto parser = SampleUniquePtr<nvonnxparser::IParser>(
//          nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger())
//      );
 
//      if (!parser->parseFromFile(mParams.modelPath.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()))) {
//          return false;
//      }
 
//      if (mParams.cudaEnable) {
//          config->setFlag(BuilderFlag::kGPU_FALLBACK);
//      }
 
//      auto profileStream = samplesCommon::makeCudaStream();
//      if (!profileStream) return false;
//      config->setProfileStream(*profileStream);
 
//      SampleUniquePtr<IHostMemory> plan = builder->buildSerializedNetwork(*network, *config);
//      if (!plan) return false;
 
//      mRuntime = std::shared_ptr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
//      mEngine = std::shared_ptr<ICudaEngine>(
//          mRuntime->deserializeCudaEngine(plan->data(), plan->size()), 
//          samplesCommon::InferDeleter()
//      );
 
//      ASSERT(network->getNbInputs() == 1);
//      mInputDims = network->getInput(0)->getDimensions();
//      ASSERT(mInputDims.nbDims == 4);
 
//      ASSERT(network->getNbOutputs() == 1);
//      mOutputDims = network->getOutput(0)->getDimensions();
//      ASSERT(mOutputDims.nbDims == 3);
 
//      return true;
 //}
 //张量的确认是否是必须
 //构建引擎
 bool TensorRTDetector::build()
{
  std::ifstream f(this->engine_file_path.c_str());
  bool fileflag = f.good();
  if (fileflag)
  {
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    std::cout << "Loading TensorRT engine from plan file..." << std::endl;
    std::ifstream file(this->engine_file_path.c_str(), std::ios::in | std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    auto buffer = std::unique_ptr<char[]>(new char[size]);
    file.read(buffer.get(), size);
    file.close();

    auto runtime = SampleUniquePtr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    auto engine = SampleUniquePtr<ICudaEngine>(runtime->deserializeCudaEngine(buffer.get(), size));
    // IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    // auto engine = runtime->deserializeCudaEngine(buffer.get(), size);
    this->Engine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, [buffer = std::move(buffer)](nvinfer1::ICudaEngine* ptr) {
      ptr->destroy();
    });
  }
  else
  {
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    builder->setMaxBatchSize(this->mbatchSize);
    if (!builder)
    {
      return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
      return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 30);//256M
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

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
      return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
      std::cout << "Failed to create CUDA stream for profiling." << std::endl;
      return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
    if (!plan)
    {
      std::cout << "Failed to build serialized network." << std::endl;
      return false;
    }

    SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
    // if (!runtime)
    // {
    //   return false;
    // }
    try {
      std::ifstream file(this->engine_file_path, std::ios::binary);
      if (!file) {
          throw std::runtime_error("Failed to open engine file.");
      }
    
  } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return false;
  }

    this->Engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!this->Engine)
    {
      return false;
    }
    //save serialize mode to file
    std::ofstream f(this->engine_file_path, std::ios::out | std::ios::binary);
    f.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    f.close();
  }
  this->prepare();
  return true;
}

//model转换为engine
bool TensorRTDetector::prepare()
{
  this->mcontext = SampleUniquePtr<nvinfer1::IExecutionContext>(this->Engine->createExecutionContext());
  /*this->mcontext->setOptimizationProfileAsync(0, this->mstream);
  cudaStreamCreate(&this->mstream);*/
  if (!this->mcontext)
  {
    return false;
  }
  return true;
}

 //对输入输出buffer进行分配
samplesCommon::BufferManager buffers(this->Engine);

 bool TensorRTDetector::infer(const cv::Mat& input_image) {
    samplesCommon::BufferManager buffers(this->Engine);
    auto context = SampleUniquePtr<IExecutionContext>(this->Engine->createExecutionContext());
    if (!context) return false;

    for (int32_t i = 0; i < this->Engine->getNbIOTensors(); i++) {
        context->setTensorAddress(this->Engine->getIOTensorName(i), buffers.getDeviceBuffer(this->Engine->getIOTensorName(i)));
        int num_outputs = this->Engine->getNbIOTensors();
        std::cout << "Number of output tensors: " << num_outputs << std::endl;
    }

    if (!processInput(buffers, input_image)) return false;
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        std::cout << "Failed to execute inference." << std::endl;
        return false;
    }

    buffers.copyOutputToHost();

    // 根据输出名称获取输出数据
    const char* output_name = this->Engine->getIOTensorName(0);  // 假设只有一个输出，索引为0
    float* output = static_cast<float*>(buffers.getHostBuffer(output_name)); 
    int num_detections = mOutputDims.d[1];
    int num_attributes = mOutputDims.d[2];
    int output_size = num_detections * num_attributes;
    std::vector<Detection> detections = postprocess(output, output_size, input_image.cols, input_image.rows);

    return true;
}
 
void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
  bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
 {
  // if (false) {
  //  int maxLen = MAX(image.rows, image.cols);
  //  outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
  //  image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
  //  params[0] = 1;
  //  params[1] = 1;
  //  params[3] = 0;
  //  params[2] = 0;
  // }
 
  cv::Size shape = image.size();
  float r = std::min((float)newShape.height / (float)shape.height,
   (float)newShape.width / (float)shape.width);
  if (!scaleUp)
   r = std::min(r, 1.0f);
 
  float ratio[2]{ r, r };
  int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };
 
  auto dw = (float)(newShape.width - new_un_pad[0]);
  auto dh = (float)(newShape.height - new_un_pad[1]);
 
  if (autoShape)
  {
   dw = (float)((int)dw % stride);
   dh = (float)((int)dh % stride);
  }
  else if (scaleFill)
  {
   dw = 0.0f;
   dh = 0.0f;
   new_un_pad[0] = newShape.width;
   new_un_pad[1] = newShape.height;
   ratio[0] = (float)newShape.width / (float)shape.width;
   ratio[1] = (float)newShape.height / (float)shape.height;
  }
 
  dw /= 2.0f;
  dh /= 2.0f;
 
  if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
  {
   cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
  }
  else {
   outImage = image.clone();
  }
 
  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  params[0] = ratio[0];
  params[1] = ratio[1];
  params[2] = left;
  params[3] = top;
  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
 }
 


 bool TensorRTDetector::processInput(const samplesCommon::BufferManager& buffers, const cv::Mat& input_image) {
     const int inputH = mParams.imgSize[1], inputW = mParams.imgSize[0];
     cv::Mat letterboxImage = letterbox(input_image, cv::Size(inputW, inputH));
     cv::Mat rgb_img;
     cv::cvtColor(letterboxImage, rgb_img, cv::COLOR_BGR2RGB);
     rgb_img.convertTo(rgb_img, CV_32F, 1/255.0f);
 //填充到输入缓冲区（NCHW格式）
    //  float* hostData = static_cast<float*>(buffers.getHostBuffer(this->Engine->getIOTensorName(0)));
    //  for (int c = 0; c < 3; ++c) {
    //      for (int h = 0; h < inputH; ++h) {
    //          for (int w = 0; w < inputW; ++w) {
    //              hostData[c * inputH * inputW + h * inputW + w] = rgb_img.at<cv::Vec3f>(h, w)[c];
    //          }
    //      }
    //使用cv::Mat指针操作数据内存
    float* hostData = static_cast<float*>(buffers.getHostBuffer(this->Engine->getIOTensorName(0)));
std::memcpy(hostData, rgb_img.data, inputH * inputW * 3 * sizeof(float));
     //}
     return true;
 }
 
//  std::vector<Detection> TensorRTDetector::postprocess(float* output, int output_size, int original_w, int original_h) {
//      const int num_classes = 80;//这个地方需要修改，onnx模型输出不是80
//      const int stride = 4 + 1 + num_classes;
//      std::vector<Detection> detections;
 
//      for (int i = 0; i < output_size; i += stride) {
//          float x1 = output[i], y1 = output[i+1], x2 = output[i+2], y2 = output[i+3];
//          float obj_conf = output[i+4];
//          int class_id = std::max_element(output+i+5, output+i+5+num_classes) - (output+i+5);
//          float confidence = obj_conf * output[i+5+class_id];
 
//          if (confidence < mParams.rectConfidenceThreshold) continue;
 
//          float ratio = std::min((float)mParams.imgSize[0]/original_w, (float)mParams.imgSize[1]/original_h);
//          int pad_w = mParams.imgSize[0] - int(original_w * ratio);
//          int pad_h = mParams.imgSize[1] - int(original_h * ratio);
//          x1 = (x1 - pad_w/2) / ratio;
//          y1 = (y1 - pad_h/2) / ratio;
//          x2 = (x2 - pad_w/2) / ratio;
//          y2 = (y2 - pad_h/2) / ratio;
 
//          detections.push_back({x1, y1, x2, y2, confidence, class_id});
//      }
 
//      std::vector<int> indices;
//      std::vector<cv::Rect> boxes;
//      std::vector<float> scores;
//      for (const auto& det : detections) {
//          boxes.emplace_back(cv::Rect(cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2)));
//          scores.push_back(det.confidence);
//      }
//      cv::dnn::NMSBoxes(boxes, scores, mParams.rectConfidenceThreshold, mParams.iouThreshold, indices);
 
//      std::vector<Detection> results;
//      for (size_t i : indices) {
//          results.push_back(detections[i]);
//      }
//      return results;
//  }
std::vector<Detection> TensorRTDetector::postprocess(float (&rst)[1][84][8400], cv::Mat &img, cv::Vec4d params)
{ 
 std::vector<cv::Rect> boxes;
 std::vector<float> scores;
 std::vector<int> det_rst;
 static const float score_threshold = 0.6;
    static const float nms_threshold = 0.45;
    std::vector<int> indices;
    int num_anchors = mOutputDims.d[2];
    int num_classes = mOutputDims.d[1] - 4; // 假设前4个是坐标
 for(int Anchors=0 ;Anchors < 8400; Anchors++)
 {
  float max_score = 0.0;
  int max_score_det = 99;
  float pdata[4];
  for(int prob = 4; prob < 84; prob++)
  {
   if(rst[0][prob][Anchors] > max_score){
    max_score = rst[0][prob][Anchors];
    max_score_det = prob - 4;
    pdata[0] = rst[0][0][Anchors];
    pdata[1] = rst[0][1][Anchors];
    pdata[2] = rst[0][2][Anchors];
    pdata[3] = rst[0][3][Anchors];
   }
  }
  if(max_score >= score_threshold)
  {
   float x = (pdata[0] - params[2]) / params[0];  
   float y = (pdata[1] - params[3]) / params[1];  
   float w = pdata[2] / params[0];  
   float h = pdata[3] / params[1];  
   int left = MAX(int(x - 0.5 * w + 0.5), 0);
   int top = MAX(int(y - 0.5 * h + 0.5), 0);
   boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
   scores.emplace_back(max_score);
   det_rst.emplace_back(max_score_det);
  }
 }
 auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),outputNodeNames.size());
 float* pdata = outputTensor.front().GetTensorMutableData<float>();
cv::Mat output_buffer(outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(1)
                                     , outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(2), CV_32FC1, pdata);

 std::vector<int> class_number_ids;
 std::vector<float> class_number_scores;
 std::vector<int> class_color_ids;
 std::vector<float> class_color_scores;
 std::vector<cv::Rect> boxes;
 std::vector<std::vector<cv::Point2f>> four_points_vec;
 for(int i = 1;i<output_buffer.rows;i++)
	{
		if(output_buffer.at<float>(i, 8) < CONFIDENCE_THRESHOLD_)
            {
                continue;
            }
            cv::Mat xyxyxyxy_boxes_mat = output_buffer.row(i).colRange(0,8);
            cv::Mat color_scores_mat = output_buffer.row(i).colRange(9,13);
            cv::Mat number_scores_mat = output_buffer.row(i).colRange(13,22);
            cv::Point number_id_point;
            cv::Point color_id_point;
            double maxNumberScore;
            double maxColorScore;
            cv::minMaxLoc(number_scores_mat, 0, &maxNumberScore, 0, &number_id_point);
            cv::minMaxLoc(color_scores_mat, 0, &maxColorScore, 0, &color_id_point);
            maxNumberScore = sigmoid(maxNumberScore);
            maxColorScore = sigmoid(maxColorScore);
            if(maxNumberScore > SCORE_THRESHOLD_ && maxColorScore > COLOR_THRESHOLD_)
            {
                int class_number = number_id_point.x;
                float class_number_score = maxNumberScore;
                int class_color = color_id_point.x;
                if(class_color == 0 && detect_color == 0) // blue
                {
                }
                else if(class_color == 1 && detect_color == 1) // red
                {
                }
                else
                {
                    continue;
                }
            }
        }
  }
// ——————————————————————————————————————————NMS分界————————————————————————————————————————————————————————
 cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);

 for (int i = 0; i < indices.size(); i++) {
        std::cout << boxes[indices[i]] << std::endl;
  cv::rectangle(img, boxes[indices[i]], Scalar(255, 0, 0), 2, LINE_8,0);
    }

 cv::imshow("rst",img);
 cv::waitKey(0);
}
 

 void ArmorDetectorNode::detect(const cv::Mat& input_image) {
     if (!detector_->infer(input_image)) {
         RCLCPP_ERROR(this->get_logger(), "TensorRT inference failed!");
     }
 }
 
 int main(int argc, char** argv) {
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