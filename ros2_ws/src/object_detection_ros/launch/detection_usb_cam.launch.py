from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    """Launch object detection with USB camera"""
    
    return LaunchDescription([
        # Camera launch arguments
        DeclareLaunchArgument(
            'camera_device',
            default_value='/dev/video0',
            description='Camera device path'
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='640',
            description='Camera image width'
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='480',
            description='Camera image height'
        ),
        
        # Detection model arguments
        DeclareLaunchArgument(
            'model_type',
            default_value='yolov8',
            description='Type of detection model'
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
            description='Minimum confidence threshold'
        ),
        DeclareLaunchArgument(
            'use_gpu',
            default_value='false',
            description='Enable GPU acceleration'
        ),
        
        # USB camera node (using usb_cam package)
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device': LaunchConfiguration('camera_device'),
                'image_width': LaunchConfiguration('image_width'),
                'image_height': LaunchConfiguration('image_height'),
                'pixel_format': 'yuyv',
                'camera_frame_id': 'camera_link',
                'io_method': 'mmap',
            }]
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
                'image_topic': '/image_raw',
                'detection_topic': '/detections',
                'debug_image_topic': '/detection_image',
                'publish_debug_image': True,
                'input_sizes': [3, 640, 640],
            }]
        ),
    ])
