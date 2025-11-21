#ifndef OBJECT_DETECTION_ROS_NODE_HPP
#define OBJECT_DETECTION_ROS_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include "DetectorSetup.hpp"
#include "Detection.hpp"

namespace object_detection_ros {

class ObjectDetectionNode : public rclcpp::Node {
public:
    ObjectDetectionNode();
    ~ObjectDetectionNode() = default;

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    vision_msgs::msg::Detection2DArray createDetectionMessage(
        const std::vector<Detection>& detections,
        const std_msgs::msg::Header& header);
    
    // ROS2 publishers/subscribers
    image_transport::Subscriber image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
    image_transport::Publisher debug_image_pub_;
    
    // Object detection
    std::unique_ptr<Detector> detector_;
    std::vector<std::string> class_labels_;
    
    // Parameters
    std::string model_type_;
    std::string weights_path_;
    std::string labels_path_;
    float confidence_threshold_;
    bool use_gpu_;
    bool publish_debug_image_;
    std::vector<int> input_sizes_;
};

} // namespace object_detection_ros

#endif // OBJECT_DETECTION_ROS_NODE_HPP
