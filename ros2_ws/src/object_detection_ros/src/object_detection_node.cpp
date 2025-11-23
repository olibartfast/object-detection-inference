#include "object_detection_ros/object_detection_node.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

namespace object_detection_ros {

ObjectDetectionNode::ObjectDetectionNode() 
    : Node("object_detection_node") {
    
    // Declare parameters
    this->declare_parameter<std::string>("model_type", "yolov8");
    this->declare_parameter<std::string>("weights_path", "");
    this->declare_parameter<std::string>("labels_path", "");
    this->declare_parameter<float>("confidence_threshold", 0.25);
    this->declare_parameter<bool>("use_gpu", false);
    this->declare_parameter<bool>("publish_debug_image", true);
    this->declare_parameter<std::vector<int64_t>>("input_sizes", {3, 640, 640});
    this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    this->declare_parameter<std::string>("detection_topic", "/detections");
    this->declare_parameter<std::string>("debug_image_topic", "/detection_image");
    
    // Get parameters
    this->get_parameter("model_type", model_type_);
    this->get_parameter("weights_path", weights_path_);
    this->get_parameter("labels_path", labels_path_);
    this->get_parameter("confidence_threshold", confidence_threshold_);
    this->get_parameter("use_gpu", use_gpu_);
    this->get_parameter("publish_debug_image", publish_debug_image_);
    
    std::vector<int64_t> input_sizes_int64;
    this->get_parameter("input_sizes", input_sizes_int64);
    input_sizes_ = std::vector<int>(input_sizes_int64.begin(), input_sizes_int64.end());
    
    std::string image_topic, detection_topic, debug_image_topic;
    this->get_parameter("image_topic", image_topic);
    this->get_parameter("detection_topic", detection_topic);
    this->get_parameter("debug_image_topic", debug_image_topic);
    
    // Validate required parameters
    if (weights_path_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "weights_path parameter is required!");
        rclcpp::shutdown();
        return;
    }
    
    if (labels_path_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "labels_path parameter is required!");
        rclcpp::shutdown();
        return;
    }
    
    // Load class labels
    std::ifstream labels_file(labels_path_);
    if (!labels_file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open labels file: %s", labels_path_.c_str());
        rclcpp::shutdown();
        return;
    }
    
    std::string label;
    while (std::getline(labels_file, label)) {
        if (!label.empty()) {
            class_labels_.push_back(label);
        }
    }
    labels_file.close();
    
    RCLCPP_INFO(this->get_logger(), "Loaded %zu class labels", class_labels_.size());
    
    // Setup inference engine and detector
    try {
        // Convert input sizes to format expected by setup_inference_engine
        std::vector<std::vector<int64_t>> input_sizes_2d;
        if (!input_sizes_.empty()) {
            std::vector<int64_t> input_sizes_int64(input_sizes_.begin(), input_sizes_.end());
            input_sizes_2d.push_back(input_sizes_int64);
        }

        // Setup inference engine
        engine_ = setup_inference_engine(weights_path_, use_gpu_, 1, input_sizes_2d);
        if (!engine_) {
            throw std::runtime_error("Failed to setup inference engine for " + weights_path_);
        }

        // Get model info from engine
        const auto model_info = engine_->get_model_info();

        // Create detector
        detector_ = DetectorSetup::createDetector(model_type_, model_info);
        if (!detector_) {
            throw std::runtime_error("Failed to setup detector " + model_type_);
        }

        RCLCPP_INFO(this->get_logger(), "Detector initialized successfully");
        RCLCPP_INFO(this->get_logger(), "  Model type: %s", model_type_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Weights: %s", weights_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Confidence threshold: %.2f", confidence_threshold_);
        RCLCPP_INFO(this->get_logger(), "  GPU enabled: %s", use_gpu_ ? "true" : "false");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize detector: %s", e.what());
        rclcpp::shutdown();
        return;
    }
    
    // Setup publishers and subscribers
    detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
        detection_topic, 10);
    
    if (publish_debug_image_) {
        debug_image_pub_ = image_transport::create_publisher(this, debug_image_topic);
    }
    
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    image_sub_ = image_transport::create_subscription(
        this,
        image_topic,
        std::bind(&ObjectDetectionNode::imageCallback, this, std::placeholders::_1),
        "raw",
        qos.get_rmw_qos_profile());
    
    RCLCPP_INFO(this->get_logger(), "Object detection node started");
    RCLCPP_INFO(this->get_logger(), "  Subscribing to: %s", image_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Publishing detections to: %s", detection_topic.c_str());
    if (publish_debug_image_) {
        RCLCPP_INFO(this->get_logger(), "  Publishing debug images to: %s", debug_image_topic.c_str());
    }
}

void ObjectDetectionNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image = cv_ptr->image;
        
        // Run detection
        auto start = std::chrono::high_resolution_clock::now();

        // Preprocess
        const auto input_blob = detector_->preprocess_image(image);

        // Inference
        const auto [outputs, shapes] = engine_->get_infer_results(input_blob);

        // Postprocess
        std::vector<Detection> detections = detector_->postprocess(outputs, shapes, image.size());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        RCLCPP_DEBUG(this->get_logger(), "Detected %zu objects in %ld ms", 
                     detections.size(), duration.count());
        
        // Publish detections
        auto detection_msg = createDetectionMessage(detections, msg->header);
        detection_pub_->publish(detection_msg);
        
        // Publish debug image if enabled
        if (publish_debug_image_ && debug_image_pub_.getNumSubscribers() > 0) {
            cv::Mat debug_image = image.clone();
            
            // Draw detections
            for (const auto& det : detections) {
                cv::rectangle(debug_image, det.bbox, cv::Scalar(0, 255, 0), 2);
                
                std::string label = det.label < static_cast<int>(class_labels_.size()) 
                    ? class_labels_[det.label] 
                    : "Unknown";
                
                std::string text = label + " " + std::to_string(static_cast<int>(det.score * 100)) + "%";
                
                int baseline;
                cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                
                cv::rectangle(debug_image, 
                             cv::Point(det.bbox.x, det.bbox.y - text_size.height - 5),
                             cv::Point(det.bbox.x + text_size.width, det.bbox.y),
                             cv::Scalar(0, 255, 0), -1);
                
                cv::putText(debug_image, text,
                           cv::Point(det.bbox.x, det.bbox.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
            
            // Convert back to ROS message
            sensor_msgs::msg::Image::SharedPtr debug_msg = 
                cv_bridge::CvImage(msg->header, "bgr8", debug_image).toImageMsg();
            debug_image_pub_.publish(debug_msg);
        }
        
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
    }
}

vision_msgs::msg::Detection2DArray ObjectDetectionNode::createDetectionMessage(
    const std::vector<Detection>& detections,
    const std_msgs::msg::Header& header) {
    
    vision_msgs::msg::Detection2DArray msg;
    msg.header = header;
    
    for (const auto& det : detections) {
        vision_msgs::msg::Detection2D detection;
        detection.header = header;
        
        // Set bounding box
        detection.bbox.center.position.x = det.bbox.x + det.bbox.width / 2.0;
        detection.bbox.center.position.y = det.bbox.y + det.bbox.height / 2.0;
        detection.bbox.size_x = det.bbox.width;
        detection.bbox.size_y = det.bbox.height;
        
        // Set class and score
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = std::to_string(det.label);
        hypothesis.hypothesis.score = det.score;
        detection.results.push_back(hypothesis);
        
        msg.detections.push_back(detection);
    }
    
    return msg;
}

} // namespace object_detection_ros

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<object_detection_ros::ObjectDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
