import numpy as np
import supervision as sv

CLASS_NAMES = {3: "motor", 2: "mobil", 7: "truk", 5: "bus"}
IMAGE_SIZE = 960
MIN_CONF = 0.1
NMS_IOU = 0.5
UNCERTAIN_CONF = (0.25, 0.5)
LINE_DISTANCE_PX = 80


def detect(model, frame):
    result = model.predict(frame, imgsz=IMAGE_SIZE, conf=MIN_CONF, iou=NMS_IOU, classes=list(CLASS_NAMES),
                           verbose=False)[0]
    return sv.Detections.from_ultralytics(result)


def distance_to_line(point, line):
    p, a, b = np.array(point, float), np.array(line[0], float), np.array(line[1], float)
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return np.linalg.norm(p - (a + t * (b - a)))


def needs_heavy_model(detections, lines):
    for (x1, y1, x2, y2), conf in zip(detections.xyxy, detections.confidence):
        if UNCERTAIN_CONF[0] <= conf < UNCERTAIN_CONF[1]:
            bottom_center = ((x1 + x2) / 2, y2)
            if any(distance_to_line(bottom_center, line) <= LINE_DISTANCE_PX for line in lines):
                return True
    return False
