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
    maintainer='neckprotecter',
    maintainer_email='644277376@qq.com',
    description='YOLOv5 RKNN ROS2 package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_node = yolov5_rknn_ros.yolov5_node:main',
        ],
    },
)
