#!/bin/bash
set -e

echo "========================================="
echo "ROS2 Jazzy Installation for Ubuntu 24.04"
echo "========================================="

# 1. Set up locale
echo "Step 1/7: Setting up locale..."
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. Setup sources
echo "Step 2/7: Setting up software sources..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update && sudo apt install -y curl

# 3. Add ROS2 GPG key
echo "Step 3/7: Adding ROS2 GPG key..."
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 4. Add ROS2 repository
echo "Step 4/7: Adding ROS2 repository..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 5. Install ROS2 Jazzy Desktop
echo "Step 5/7: Installing ROS2 Jazzy Desktop (this will take several minutes)..."
sudo apt update
sudo apt install -y ros-jazzy-desktop

# 6. Install development tools
echo "Step 6/7: Installing ROS2 development tools..."
sudo apt install -y ros-dev-tools

# 7. Install required packages for object detection
echo "Step 7/7: Installing vision packages for object detection..."
sudo apt install -y \
    ros-jazzy-vision-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-usb-cam

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "To use ROS2, run:"
echo "  source /opt/ros/jazzy/setup.bash"
echo ""
echo "To add ROS2 to your shell automatically, run:"
echo "  echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc"
echo ""