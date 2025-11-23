#!/bin/bash
set -e

echo "========================================="
echo "Installing APT 3rd Party Dependencies"
echo "========================================="

# Update package list
echo "Step 1/3: Updating package list..."
sudo apt update

# Install core build dependencies
echo "Step 2/3: Installing core build dependencies..."
sudo apt install -y \
    cmake \
    wget \
    tar \
    unzip \
    curl \
    build-essential \
    pkg-config

# Install library dependencies
echo "Step 3/3: Installing library dependencies..."
sudo apt install -y \
    libopencv-dev \
    libgoogle-glog-dev

# Optional: Install video capture dependencies (GStreamer and FFmpeg)
echo ""
read -p "Install optional video capture libraries (GStreamer & FFmpeg)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing GStreamer dependencies..."
    sudo apt install -y \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly

    echo "Installing FFmpeg dependencies..."
    sudo apt install -y \
        libavformat-dev \
        libavcodec-dev \
        libavutil-dev \
        libswscale-dev \
        libavfilter-dev

    echo "Video capture libraries installed!"
fi

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Installed dependencies:"
echo "  - Build tools: cmake, wget, tar, unzip, curl"
echo "  - OpenCV development libraries"
echo "  - Google glog logging library"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  - GStreamer development libraries"
    echo "  - FFmpeg development libraries"
fi
echo ""
echo "You can now build the object detection library."
echo ""
