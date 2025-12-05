#!/bin/bash
# Wrapper script to setup environment and run the object detection node

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Source the environment setup script
source "$SCRIPT_DIR/setup_env.sh"

# Run the ROS 2 node
echo "Starting object_detection_node..."
ros2 run object_detection_ros object_detection_node --ros-args \
    --log-level debug \
    -p weights_path:="$SCRIPT_DIR/data/models/dummy.onnx" \
    -p labels_path:="$SCRIPT_DIR/data/labels/coco.names" \
    -p publish_debug_image:=false
