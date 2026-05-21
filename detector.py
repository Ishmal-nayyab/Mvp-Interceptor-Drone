"""
detector.py
-----------
OpenCV se detection - PyTorch/YOLO ki zaroorat NAHI.
Sirf opencv-python chahiye.
"""

import cv2
import numpy as np

class DroneDetector:
    def __init__(self, confidence=0.3):
        self.confidence = confidence
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=40,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 50000:
                continue
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2 = x1 + w
            y2 = y1 + h
            cx = x1 + w // 2
            cy = y1 + h // 2
            conf = min(area / 5000.0, 1.0)
            detections.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": cx,  "cy": cy,
                "conf": round(conf, 2),
                "label": "moving_object"
            })

        detections.sort(key=lambda d: d["conf"], reverse=True)
        return detections[:3]

    def draw(self, frame, detections):
        for d in detections:
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)
            text = f"Target {d['conf']:.2f}"
            cv2.putText(frame, text, (d["x1"], d["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.circle(frame, (d["cx"], d["cy"]), 4, (0, 255, 0), -1)
        return frame
