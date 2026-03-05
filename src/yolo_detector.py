import cv2
import torch
import numpy as np
from ultralytics import YOLO

class YOLOFaceDetector:
    def __init__(self, model_path="od_model/yolov8n-face.pt", conf_threshold=0.5, device="cuda"):
        self.device = 'cuda' if torch.cuda.is_available() and device == "cuda" else 'cpu'
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        bboxes = []
        
        if len(results) > 0:
            for result in results[0].boxes:
                x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
                conf = float(result.conf[0].cpu())
                cls = float(result.cls[0].cpu())
                bboxes.append([x1, y1, x2, y2, conf, cls])

        return bboxes