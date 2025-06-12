# from launch import LaunchDescription
# from launch_ros.actions import Node
# from ament_index_python.packages import get_package_share_directory
# import os

# def generate_launch_description():
#     yolov5_rknn_ros_dir = get_package_share_directory('yolov5_rknn_ros')

#     model_path = os.path.join(yolov5_rknn_ros_dir, 'yolov5_rknn_ros', 'sleep_v5_epoch30.rknn')
#     anchors_path = os.path.join(yolov5_rknn_ros_dir, 'yolov5_rknn_ros', 'anchors_yolov5.txt')

#     return LaunchDescription([
#         Node(
#             package='usb_cam',
#             executable='usb_cam_node_exe',
#             name='usb_cam'
#         ),
#         Node(
#             package='yolov5_rknn_ros',
#             executable='yolov5_node',
#             name='yolov5_node',
#             parameters=[
#                 {'model_path': model_path},
#                 {'anchors_path': anchors_path},
#                 {'device_id': '0'},
#                 {'target': 'rk3588'}
#             ]
#         )
#     ])



import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 获取当前包的共享目录路径
    yolov5_pkg_dir = get_package_share_directory('yolov5_rknn_ros')
    
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam_node'
        ),
        Node(
            package='yolov5_rknn_ros',
            executable='yolov5_node',
            name='yolov5_node',
            parameters=[
                {
                    'model_path': os.path.join(yolov5_pkg_dir, 'models', 'sleep_v5_epoch30_30.rknn'),
                    'anchors_path': os.path.join(yolov5_pkg_dir, 'models', 'anchors_yolov5.txt'),
                    'device_id': '0',
                    'target': 'rk3588'
                }
            ]
        )
    ])