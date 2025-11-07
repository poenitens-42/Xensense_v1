# detector.py — unified dual-model (YOLO11 general + pothole) with flicker smoothing
from ultralytics import YOLO
import numpy as np

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    a_area = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    b_area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = a_area + b_area - inter + 1e-6
    return inter / union

class Detector:
    """
    Unified detector:
      - Model A: general YOLO11 (yolo11n.pt) → normal classes
      - Model B: pothole model (best.pt) → class name 'pothole'
    Returns [x1, y1, x2, y2, conf, cls_name] in *input frame* coordinates.
    """
    def __init__(
        self,
        general_model_path="yolo11n.pt",
        conf_thresh=0.45,
        pothole_model_path="models/best.pt",
        pothole_conf=0.30,
        frame_skip=2,
        iou_merge=0.6,
        min_box=6
    ):
        self.general = YOLO(general_model_path)
        self.pothole = YOLO(pothole_model_path)
        self.conf_thresh = float(conf_thresh)
        self.pothole_conf = float(pothole_conf)
        self.frame_skip = int(frame_skip)
        self.iou_merge = float(iou_merge)
        self.min_box = int(min_box)

        self.allowed_classes = None  # set from main if needed
        self._frame_id = 0
        self._last_dets = []         # for frame-skipping persistence

        # cache names for speed
        # Ultralytics models expose .names dict
        self.general_names = getattr(self.general, "names", None)
        if self.general_names is None:
            self.general_names = {i: str(i) for i in range(1000)}

    def _filter_small(self, x1, y1, x2, y2):
        return (x2 - x1) >= self.min_box and (y2 - y1) >= self.min_box

    def _keep_class(self, cls_name):
        if not self.allowed_classes:
            return True
        return (cls_name in self.allowed_classes)

    def detect(self, frame_bgr):
        """
        frame_bgr: already-resized frame (we DO NOT resize inside).
        Returns list of [x1, y1, x2, y2, conf, cls_name] in this frame's coordinates.
        """
        self._frame_id += 1

        # Cheap frame skip with persistence (reduces flicker + boosts FPS)
        if self.frame_skip > 1 and (self._frame_id % self.frame_skip != 0):
            return list(self._last_dets)

        dets = []

        # ---- Model A: general ----
        ra = self.general.predict(frame_bgr, verbose=False)[0]
        if ra.boxes is not None:
            xyxy = ra.boxes.xyxy.cpu().numpy()
            confs = ra.boxes.conf.cpu().numpy()
            clss = ra.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                if float(c) < self.conf_thresh:
                    continue
                if not self._filter_small(int(x1), int(y1), int(x2), int(y2)):
                    continue
                cls_name = self.general_names.get(k, str(k))
                if not self._keep_class(cls_name):
                    continue
                dets.append([int(x1), int(y1), int(x2), int(y2), float(c), cls_name])

        # ---- Model B: pothole-only ----
        rb = self.pothole.predict(frame_bgr, verbose=False)[0]
        if rb.boxes is not None:
            xyxy = rb.boxes.xyxy.cpu().numpy()
            confs = rb.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                if float(c) < self.pothole_conf:
                    continue
                if not self._filter_small(int(x1), int(y1), int(x2), int(y2)):
                    continue
                # Merge logic: don't suppress potholes against general classes.
                dets.append([int(x1), int(y1), int(x2), int(y2), float(c), "pothole"])

        # ---- Light duplicate suppression within same class (except pothole)
        # (Ultralytics already did NMS per model; this only removes overlaps across the two streams *within same class*)
        merged = []
        for d in dets:
            keep = True
            for m in merged:
                if d[5] == m[5] and d[5] != "pothole":  # same class, not pothole
                    if _iou_xyxy(d[:4], m[:4]) >= self.iou_merge:
                        # keep the higher-confidence one
                        if d[4] <= m[4]:
                            keep = False
                            break
                        else:
                            # replace
                            m[:] = d
                            keep = False
                            break
            if keep:
                merged.append(d)

        self._last_dets = merged
        return merged
