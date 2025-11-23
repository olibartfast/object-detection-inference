cmake_minimum_required(VERSION 3.15)
project(object_detection_ros)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)

# Set the path to the object-detection-inference library
set(OBJECT_DETECTION_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../" CACHE PATH "Path to object-detection-inference root")

# Include directories
include_directories(
  include
  ${OBJECT_DETECTION_ROOT}/detectors/inc
  ${OBJECT_DETECTION_ROOT}/detectors/src
  ${OBJECT_DETECTION_ROOT}/detectors/src/models
  ${OBJECT_DETECTION_ROOT}/build/_deps/neuriplo-src/backends/src
  ${OBJECT_DETECTION_ROOT}/build/_deps/neuriplo-src/include
  ${OBJECT_DETECTION_ROOT}/app/inc
  ${OpenCV_INCLUDE_DIRS}
)

# Link directories
link_directories(
  ${OBJECT_DETECTION_ROOT}/build
  ${OBJECT_DETECTION_ROOT}/build/detectors
  ${OBJECT_DETECTION_ROOT}/build/_deps/neuriplo-build
)

# Create the ROS2 node executable
add_executable(object_detection_node
  src/object_detection_node.cpp
)

ament_target_dependencies(object_detection_node
  rclcpp
  std_msgs
  sensor_msgs
  vision_msgs
  cv_bridge
  image_transport
  message_filters
)

target_link_libraries(object_detection_node
  ${OpenCV_LIBS}
  detectors
  neuriplo
  glog
)

# Install targets
install(TARGETS
  object_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
