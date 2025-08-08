from ultralytics import YOLO
import os

class Detector:
    def __init__(self, model_path, conf_thresh=0.5, iou_thresh=0.35, imgsz=1280, allowed_classes=None):
        # Switch to yolo11m if file not found
        if not os.path.exists(model_path):
            print(f"[WARNING] Model '{model_path}' not found. Using 'yolo11m.pt' instead.")
            model_path = "yolo11m.pt"

        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.allowed_classes = allowed_classes

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = self.model.names[cls_id]

            # Filter by allowed classes if set
            if self.allowed_classes and cls_name not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            detections.append(((x1, y1, x2 - x1, y2 - y1), conf, cls_name))

        return detections
