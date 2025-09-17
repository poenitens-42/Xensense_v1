import cv2
from ultralytics import YOLO
import numpy as np
from utils import inpainting as inpaint_utils

class Detector:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45,
                 allowed_classes=None, prompt_filter=None,
                 enable_inpainting=False, inpainting_method="telea"):

        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.allowed_classes = allowed_classes if allowed_classes else []
        self.prompt_filter = prompt_filter.lower() if prompt_filter else None
        self.enable_inpainting = enable_inpainting
        self.inpainting_method = inpainting_method

    def detect(self, frame):
        """Runs YOLO detection and applies optional prompt filter & inpainting."""
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh
        )

        detections = []
        if not results:
            return detections

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = self.model.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                                                                                        # Allowed classes filter
                if self.allowed_classes and cls_name not in self.allowed_classes:
                    continue

                                                                                        # Prompt-based detection filter ( to be implemented in th future )
                if self.prompt_filter and self.prompt_filter not in cls_name.lower():
                    continue

                                                                                        # Optional inpainting( using LaMa instead of telia )
                if self.enable_inpainting:
                    roi = frame[y1:y2, x1:x2]
                    inpainted_roi = inpaint_utils.inpaint_object(
                        roi, method=self.inpainting_method
                    )
                    frame[y1:y2, x1:x2] = inpainted_roi

                                                                                       # Save detection in [x1, y1, x2, y2, conf, cls_name] format
                detections.append([x1, y1, x2, y2, conf, cls_name])

        return detections
