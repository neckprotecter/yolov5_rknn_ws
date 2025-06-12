import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

from yolov5_rknn_ros.yolov5 import yoloV5RKNN  # 保持你原来 yolov5.py 结构
from yolov5_rknn_ros.utils import draw_detections

CLASSES = ("sleep",)

class yolov5Node(Node):
    def __init__(self):
        super().__init__('yolov5_node')
        self.declare_parameter('model_path', '')
        self.declare_parameter('anchors_path', '')
        self.declare_parameter('device_id', '0')
        self.declare_parameter('target', '3588')

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        anchors_path = self.get_parameter('anchors_path').get_parameter_value().string_value
        device_id = self.get_parameter('device_id').get_parameter_value().string_value
        target = self.get_parameter('target').get_parameter_value().string_value

        self.get_logger().info(f"Successfully loaded RKNN model from {model_path}")
        self.get_logger().info(f"Successfully loaded RKNN anchors from {anchors_path}")
        self.get_logger().info(f"Device_id is {device_id}")
        self.get_logger().info(f"Target is {target}")

        self.get_logger().info(f'Loading RKNN model from {model_path}...')
        self.model = yoloV5RKNN(model_path, anchors_path, target, device_id)
        # self.model.init()

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.result_pub = self.create_publisher(String, '/yolov5/result', 10)
        self.image_pub = self.create_publisher(Image, '/yolov5/image_result', 10)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        orig = frame.copy()
        boxes, class_ids, scores = self.model.detect(frame)

        if class_ids is None or len(class_ids) == 0:
            result_msg = self.bridge.cv2_to_imgmsg(orig, encoding='bgr8')
            self.image_pub.publish(result_msg)
            # self.get_logger().info("No detections")
            return

        # Publish detection result text
        result_text = f'Detected: {[f"{class_ids[i]}({scores[i]:.2f})" for i in range(len(class_ids))]}'
        self.result_pub.publish(String(data=result_text))

        # Draw boxes and publish image
        result_img = orig.copy()
        self.model.draw(result_img, boxes, class_ids, scores)
        # result_img = draw_detections(orig, boxes, scores, class_ids)
        result_msg = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
        self.image_pub.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = yolov5Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()