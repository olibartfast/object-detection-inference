# ROS2 Object Detection Package

## Quick Reference

### Build
```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select object_detection_ros
source install/setup.bash
```

### Run
```bash
# Basic usage
ros2 launch object_detection_ros detection.launch.py \
    model_type:=yolov8 \
    weights_path:=/path/to/model.onnx \
    labels_path:=/path/to/labels.txt

# With USB camera
ros2 launch object_detection_ros detection_usb_cam.launch.py \
    model_type:=yolov8 \
    weights_path:=/path/to/model.onnx \
    labels_path:=/path/to/labels.txt \
    camera_device:=/dev/video0
```

### Topics
- **Subscribe**: `/camera/image_raw` (sensor_msgs/Image)
- **Publish**: `/detections` (vision_msgs/Detection2DArray)
- **Publish**: `/detection_image` (sensor_msgs/Image) - annotated debug images

See [ROS2-Deployment.md](../../docs/ROS2-Deployment.md) for complete documentation.
