#!/bin/bash
# Helper script to setup the environment for object-detection-inference ROS node

# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define build paths
DETECTORS_LIB_PATH="$SCRIPT_DIR/build/detectors"
NEURIPLO_LIB_PATH="$SCRIPT_DIR/build/_deps/neuriplo-build"

# Check if libraries exist
if [ ! -f "$DETECTORS_LIB_PATH/libdetectors.so" ]; then
    echo "WARNING: libdetectors.so not found at $DETECTORS_LIB_PATH"
    echo "Have you built the project? (mkdir build && cd build && cmake .. && cmake --build .)"
fi

# Add build directories to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DETECTORS_LIB_PATH:$NEURIPLO_LIB_PATH

# Source ROS 2 setup
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Source workspace setup
if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
fi

echo "Environment setup complete!"
echo "LD_LIBRARY_PATH updated with:"
echo "  - $DETECTORS_LIB_PATH"
echo "  - $NEURIPLO_LIB_PATH"
