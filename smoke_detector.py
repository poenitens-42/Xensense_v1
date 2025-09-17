import cv2
import numpy as np
from ultralytics import YOLO

class SmokeDetector:
    def __init__(self, model_path=None, use_yolo=True, min_area=5000):
        """
        Args:
            model_path (str): path to YOLO model trained on smoke/fog
            use_yolo (bool): whether to use YOLO or fallback to classical method
            min_area (int): minimum area for classical contour detection
        """
        self.use_yolo = use_yolo
        self.model = None
        self.min_area = min_area

        if use_yolo and model_path:
            self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Detect smoke regions in a frame.
        Returns: list of dicts [{x1, y1, x2, y2, conf}]
        """
        detections = []

        if self.use_yolo and self.model:
            results = self.model(frame)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Assuming "smoke" is class 0
                if cls == 0:
                    detections.append({
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "conf": conf
                    })

        else:
            # Classical fallback: haze/smoke-like region detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            diff = cv2.absdiff(gray, blur)
            _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > self.min_area:
                    detections.append({
                        "x1": x, "y1": y,
                        "x2": x + w, "y2": y + h,
                        "conf": 0.5  # placeholder confidence
                    })

        return detections
