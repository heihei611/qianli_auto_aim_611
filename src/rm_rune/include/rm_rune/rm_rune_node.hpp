// rm_rune_node.hpp

#ifndef RM_RUNE_NODE_HPP
#define RM_RUNE_NODE_HPP

#include <iostream>
#include <memory> // 新增
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "rm_msgs/msg/armor.hpp"
#include "rm_msgs/msg/status.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include "power_rune.hpp"
#include "blade.hpp"
#include "prediction.hpp"
#include "detect.hpp"
#include "openvino_detect.hpp"
#include <image_transport/image_transport.hpp> // 新增

namespace qianli_rm_rune
{

class RuneNode : public rclcpp::Node
{
public:
    RuneNode(const rclcpp::NodeOptions & options);

    void rune_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void status_callback(const rm_msgs::msg::Status::SharedPtr msg);


    // 发布者
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr rune_pose_pub_;
    std::unique_ptr<image_transport::ImageTransport> it_; // 修改为 unique_ptr
    image_transport::Publisher result_image_pub_; // 修改类型为 image_transport::Publisher

    // 订阅者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rune_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
    rclcpp::Subscription<rm_msgs::msg::Status>::SharedPtr status_sub_;
    

    // 相机矩阵
    cv::Mat camera_matrix_;
    size_t frame_count_;
    rclcpp::Time last_time_;

    // 配置和处理类
    Configuration cfg_;
    PowerRune power_rune_;
    yolo::Inference inference;
    ContourInfo contour_info_;
    Prediction predictor;
    std::vector<ContourInfo> contours_info_;

    // TF2 缓存和监听器
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;


    // // 定时器用于延迟初始化 image_transport
    rclcpp::TimerBase::SharedPtr init_timer_;

    bool is_rune_ = true; // 默认为打符模式，才能刷新接受者接受图片消息, 即开始默认自己不能接受图片消息
};


} // namespace qianli_rm_rune

#endif // RM_RUNE_NODE_HPP