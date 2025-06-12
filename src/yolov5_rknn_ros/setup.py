import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolov5_rknn_ros'

setup(
    name=package_name,
    version='0.0.0',
    # packages=[package_name],
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # 安装models文件夹下的所有文件
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='YOLOv5 RKNN ROS2 package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_node = yolov5_rknn_ros.yolov5_node:main',
        ],
    },
)

# import os
# from setuptools import find_packages, setup

# package_name = 'yolov5_rknn_ros'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#         (os.path.join('share', package_name, 'launch'), ['launch/detect.launch.py']),
#         (os.path.join('share', package_name), ['sleep_v5_epoch30.rknn']),
#         (os.path.join('share', package_name), ['anchors_yolov5.txt']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='orangepi',
#     maintainer_email='orangepi@todo.todo',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'yolov5_node = yolov5_rknn_ros.yolov5_node:main', 
#         ],
#     },
# )


# from setuptools import setup

# package_name = 'yolov5_rknn_ros'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=[package_name],
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#         (os.path.join('share', package_name, 'launch'), ['launch/yolov5.launch.py']),
#         (os.path.join('share', package_name), ['sleep_v5_epoch30.rknn']),
#         (os.path.join('share', package_name), ['anchors_yolov5.txt']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='your_name',
#     maintainer_email='your_email@example.com',
#     description='YOLOv5 RKNN ROS2 package',
#     license='Apache License 2.0',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'yolov5_node = yolov5_rknn_ros.yolov5_node:main',
#         ],
#     },
# )