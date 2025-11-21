from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Launch object detection node with camera input"""
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'model_type',
            default_value='yolov8',
            description='Type of detection model (yolov4, yolov5, yolov8, rtdetr, etc.)'
        ),
        DeclareLaunchArgument(
            'weights_path',
            description='Path to model weights file'
        ),
        DeclareLaunchArgument(
            'labels_path',
            description='Path to class labels file'
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.25',
            description='Minimum confidence threshold for detections'
        ),
        DeclareLaunchArgument(
            'use_gpu',
            default_value='false',
            description='Enable GPU acceleration'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Input image topic'
        ),
        DeclareLaunchArgument(
            'detection_topic',
            default_value='/detections',
            description='Output detection topic'
        ),
        DeclareLaunchArgument(
            'debug_image_topic',
            default_value='/detection_image',
            description='Output debug image topic'
        ),
        DeclareLaunchArgument(
            'publish_debug_image',
            default_value='true',
            description='Publish annotated debug images'
        ),
        DeclareLaunchArgument(
            'input_sizes',
            default_value='[3, 640, 640]',
            description='Model input sizes [C, H, W]'
        ),
        
        # Object detection node
        Node(
            package='object_detection_ros',
            executable='object_detection_node',
            name='object_detection_node',
            output='screen',
            parameters=[{
                'model_type': LaunchConfiguration('model_type'),
                'weights_path': LaunchConfiguration('weights_path'),
                'labels_path': LaunchConfiguration('labels_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'use_gpu': LaunchConfiguration('use_gpu'),
                'image_topic': LaunchConfiguration('image_topic'),
                'detection_topic': LaunchConfiguration('detection_topic'),
                'debug_image_topic': LaunchConfiguration('debug_image_topic'),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'input_sizes': LaunchConfiguration('input_sizes'),
            }]
        ),
    ])
