import cv2
import random

# 如果有类别名称文件，替换下面的类别名列表
CLASS_NAMES = [f'class_{i}' for i in range(80)]

def draw_detections(image, boxes, scores, class_ids, conf_threshold=0.3):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = f'{CLASS_NAMES[cls_id]} {score:.2f}'
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image