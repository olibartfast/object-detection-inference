# ROS2 Deployment Guide

This guide explains how to deploy the object detection inference framework as a ROS2 node for robotics applications.

## Overview

The ROS2 wrapper provides a bridge between the object-detection-inference library and the ROS2 ecosystem, allowing you to:
- Subscribe to camera image topics
- Run real-time object detection
- Publish detection results as ROS2 messages
- Visualize results with annotated debug images

## Prerequisites

### System Requirements
- Ubuntu 22.04 (for ROS2 Humble) or Ubuntu 24.04 (for ROS2 Jazzy)
- ROS2 installation (Humble or Jazzy recommended)
- Built object-detection-inference library

### ROS2 Installation
If you haven't installed ROS2 yet:

```bash
# For Ubuntu 22.04 - ROS2 Humble
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop

# For Ubuntu 24.04 - ROS2 Jazzy
# Similar process with ros-jazzy-desktop
```

### Required ROS2 Packages
```bash
# Source ROS2
source /opt/ros/humble/setup.bash  # or jazzy

# Install required packages
sudo apt install -y \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-usb-cam
```

## Building the Object Detection Library

First, build the core object detection library:

```bash
cd /workspaces/object-detection-inference

# Setup dependencies (choose your backend)
./scripts/setup_dependencies.sh --backend onnx_runtime

# Build the library
mkdir -p build && cd build
cmake -DDEFAULT_BACKEND=ONNX_RUNTIME -DBUILD_ONLY_LIB=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Building the ROS2 Package

```bash
# Navigate to ROS2 workspace
cd /workspaces/object-detection-inference/ros2_ws

# Source ROS2
source /opt/ros/humble/setup.bash

# Build the package
colcon build --packages-select object_detection_ros

# Source the workspace
source install/setup.bash
```

## Usage

### Basic Launch

Launch the detection node with a custom image topic:

```bash
ros2 launch object_detection_ros detection.launch.py \
    model_type:=yolov8 \
    weights_path:=/path/to/yolov8s.onnx \
    labels_path:=/path/to/coco.names \
    image_topic:=/camera/image_raw
```

### Launch with USB Camera

To run detection with a USB camera:

```bash
ros2 launch object_detection_ros detection_usb_cam.launch.py \
    model_type:=yolov8 \
    weights_path:=/path/to/yolov8s.onnx \
    labels_path:=/path/to/coco.names \
    camera_device:=/dev/video0
```

### Using Configuration Files

Create a configuration file (see examples in `config/`) and run:

```bash
ros2 run object_detection_ros object_detection_node \
    --ros-args --params-file config/yolov8_example.yaml
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_type` | `yolov8` | Detection model type (yolov4, yolov5, yolov8, rtdetr, etc.) |
| `weights_path` | (required) | Path to model weights file |
| `labels_path` | (required) | Path to class labels file |
| `confidence_threshold` | `0.25` | Minimum confidence for detections |
| `use_gpu` | `false` | Enable GPU acceleration |
| `image_topic` | `/camera/image_raw` | Input image topic to subscribe to |
| `detection_topic` | `/detections` | Output detection topic |
| `debug_image_topic` | `/detection_image` | Annotated debug image topic |
| `publish_debug_image` | `true` | Publish annotated images |
| `input_sizes` | `[3, 640, 640]` | Model input dimensions [C, H, W] |

## ROS2 Topics

### Subscribed Topics
- `image_topic` (default: `/camera/image_raw`) - `sensor_msgs/Image`
  - Input camera images for detection

### Published Topics
- `detection_topic` (default: `/detections`) - `vision_msgs/Detection2DArray`
  - Detected objects with bounding boxes and classifications
  
- `debug_image_topic` (default: `/detection_image`) - `sensor_msgs/Image`
  - Annotated images with detection overlays (if `publish_debug_image=true`)

## Supported Models

All models supported by the object-detection-inference framework can be used:

- **YOLO Family**: YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLOv12
- **RT-DETR Family**: RT-DETR, RT-DETRv2, RT-DETR Ultralytics
- **Specialized**: YOLO-NAS, D-FINE, DEIM, DEIMv2, RF-DETR

Refer to the main [README](../README.md) for model export instructions.

## Examples

### YOLOv8 with ONNX Runtime
```bash
ros2 launch object_detection_ros detection.launch.py \
    model_type:=yolov8 \
    weights_path:=/models/yolov8s.onnx \
    labels_path:=/labels/coco.names \
    confidence_threshold:=0.3 \
    use_gpu:=false
```

### RT-DETR with GPU Acceleration
```bash
ros2 launch object_detection_ros detection.launch.py \
    model_type:=rtdetrul \
    weights_path:=/models/rtdetr-l.onnx \
    labels_path:=/labels/coco.names \
    confidence_threshold:=0.4 \
    use_gpu:=true
```

### RF-DETR Real-time Detection
```bash
ros2 launch object_detection_ros detection.launch.py \
    model_type:=rfdetr \
    weights_path:=/models/rfdetr.onnx \
    labels_path:=/labels/coco.names \
    confidence_threshold:=0.35
```

## Visualization

### Using RViz2

```bash
# Launch RViz2
rviz2

# Add displays:
# 1. Image display for /detection_image
# 2. MarkerArray for visualizing bounding boxes (if implemented)
```

### Using rqt_image_view

```bash
# View detection images
rqt_image_view /detection_image
```

## Performance Tips

1. **GPU Acceleration**: Enable GPU support for faster inference
   ```bash
   use_gpu:=true
   ```

2. **Backend Selection**: Build with TensorRT for best GPU performance
   ```bash
   ./scripts/setup_dependencies.sh --backend tensorrt
   ```

3. **Model Selection**: Choose model size based on hardware
   - Edge devices: YOLOv8n, YOLOv10n, RF-DETR nano
   - Standard GPUs: YOLOv8s/m, RT-DETR
   - High-end GPUs: YOLOv8l/x, RT-DETR-x

4. **Image Resolution**: Match camera resolution to model input size when possible

## Integration with Existing ROS2 Systems

### Camera Integration
The node subscribes to standard `sensor_msgs/Image` topics, making it compatible with:
- `usb_cam` - USB cameras
- `realsense2_camera` - Intel RealSense cameras
- `cv_camera` - OpenCV-based cameras
- Any camera publishing `sensor_msgs/Image`

### Detection Output Format
Detection results are published as `vision_msgs/Detection2DArray`, which is compatible with:
- Standard ROS2 vision pipelines
- Robot navigation systems
- Object tracking nodes

## Troubleshooting

### Common Issues

**Issue**: Node fails to start with "Failed to initialize detector"
- **Solution**: Verify weights path and model type match. Check backend is properly built.

**Issue**: No detections published
- **Solution**: Check confidence threshold. Verify camera topic is publishing. Use `ros2 topic echo` to debug.

**Issue**: Low frame rate
- **Solution**: Enable GPU, use smaller model, or reduce input resolution.

**Issue**: "cv_bridge exception"
- **Solution**: Ensure camera image encoding is supported (BGR8 or RGB8).

### Debug Commands

```bash
# List active topics
ros2 topic list

# Check camera images
ros2 topic echo /camera/image_raw --no-arr

# Monitor detection output
ros2 topic echo /detections

# Check node parameters
ros2 param list /object_detection_node

# View node logs
ros2 run object_detection_ros object_detection_node --ros-args --log-level debug
```

## Advanced Configuration

### Custom Message Filters
For synchronized multi-camera systems, you can modify the node to use `message_filters` for time synchronization.

### Dynamic Reconfiguration
Parameters can be updated at runtime:
```bash
ros2 param set /object_detection_node confidence_threshold 0.5
```

## Docker Deployment with ROS2

Coming soon: Dockerized ROS2 deployment with GPU support.

## License

This ROS2 wrapper follows the same MIT license as the main object-detection-inference project.

## Support

For issues specific to ROS2 deployment:
1. Check this documentation
2. Review main project [README](../README.md)
3. Open an issue on the [GitHub repository](https://github.com/olibartfast/object-detection-inference/issues)

For ROS2-specific questions, refer to the [ROS2 documentation](https://docs.ros.org/).
