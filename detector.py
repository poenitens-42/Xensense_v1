# detector.py — Clean dual YOLO detector (general + optional pothole model)
# Stable + NMS + top-5 potholes + no crashes

from ultralytics import YOLO
import numpy as np
import torch


def nms_pytorch(boxes, scores, iou_thresh=0.5):
    """Safe NMS even if torchvision.ops is unavailable."""
    if len(boxes) == 0:
        return []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    try:
        keep = torch.ops.torchvision.nms(boxes, scores, float(iou_thresh))
        return keep.cpu().numpy().tolist()
    except Exception:
        # fallback slow NMS
        keep = []
        idxs = scores.argsort(descending=True)

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(int(i))

            if len(idxs) == 1:
                break

            cur = boxes[i]
            others = boxes[idxs[1:]]

            xx1 = torch.maximum(cur[0], others[:, 0])
            yy1 = torch.maximum(cur[1], others[:, 1])
            xx2 = torch.minimum(cur[2], others[:, 2])
            yy2 = torch.minimum(cur[3], others[:, 3])

            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            area_i = (cur[2] - cur[0]) * (cur[3] - cur[1])
            area_o = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
            union = area_i + area_o - inter + 1e-6
            iou = inter / union

            idxs = idxs[1:][iou <= iou_thresh]

        return keep


class Detector:
    """
    Clean dual-model detector:
      - YOLO general (cars, trucks, people, etc.)
      - YOLO pothole model (optional)
    Output format:
      [x1, y1, x2, y2, conf, cls_name]
    """

    def __init__(
        self,
        general_model_path="yolo11n.pt",
        conf_thresh=0.45,
        pothole_model_path="models/best.pt",
        pothole_conf=0.60,
        frame_skip=1,
        pothole_iou=0.45,
        min_pothole_size=40,
        max_potholes=5,
    ):
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models
        self.general = YOLO(general_model_path)
        self.general.to(self.device)

        self.pothole = YOLO(pothole_model_path)
        self.pothole.to(self.device)

        # Parameters
        self.conf_thresh = float(conf_thresh)
        self.pothole_conf = float(pothole_conf)
        self.pothole_iou = float(pothole_iou)
        self.min_pothole_size = int(min_pothole_size)
        self.max_potholes = int(max_potholes)

        self.frame_skip = int(frame_skip)
        self._frame_id = 0
        self._last_dets = []

        # Toggles
        self.pothole_enabled = True
        self.allowed_classes = None

        self.general_names = self.general.names or {}


    def _bbox_ok(self, x1, y1, x2, y2):
        """Reject very small boxes."""
        return (x2 - x1) >= 8 and (y2 - y1) >= 8


    def detect(self, frame_bgr):
        """Run detection (YOLO general + optional pothole)."""

        self._frame_id += 1

        # FRAME SKIP
        if self.frame_skip > 1 and (self._frame_id % self.frame_skip != 0):
            return list(self._last_dets)

        dets = []

        # ==========================================================
        # 1) GENERAL YOLO MODEL
        # ==========================================================
        try:
            ra = self.general.predict(
                frame_bgr, device=self.device, verbose=False
            )[0]
        except Exception as e:
            print("[ERROR] General YOLO detection failed:", e)
            return list(self._last_dets)

        if ra.boxes is not None:
            xyxy = ra.boxes.xyxy.cpu().numpy()
            confs = ra.boxes.conf.cpu().numpy()
            clss = ra.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                if float(c) < self.conf_thresh:
                    continue
                if not self._bbox_ok(x1, y1, x2, y2):
                    continue

                cls_name = self.general_names.get(k, str(k))

                # allow filtering
                if self.allowed_classes and cls_name not in self.allowed_classes:
                    continue

                dets.append([int(x1), int(y1), int(x2), int(y2), float(c), cls_name])


        # ==========================================================
        # 2) POTHOLE MODEL (OPTIONAL)
        # ==========================================================
        potholes = []

        if self.pothole_enabled:
            try:
                rb = self.pothole.predict(
                    frame_bgr, device=self.device, verbose=False
                )[0]
            except Exception as e:
                print("[ERROR] Pothole model failed:", e)
                rb = None

            if rb and rb.boxes is not None:
                xyxy = rb.boxes.xyxy.cpu().numpy()
                confs = rb.boxes.conf.cpu().numpy()

                H = frame_bgr.shape[0]

                boxes = []
                scores = []

                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    if float(c) < self.pothole_conf:
                        continue
                    if (x2 - x1) < self.min_pothole_size or (y2 - y1) < self.min_pothole_size:
                        continue
                    if y1 < H * 0.45:  # bottom half only
                        continue

                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(c))

                # NMS
                keep = nms_pytorch(boxes, scores, iou_thresh=self.pothole_iou)

                # Sort by confidence and limit to top-N potholes
                selected = sorted(
                    [(boxes[i], scores[i]) for i in keep],
                    key=lambda x: x[1],
                    reverse=True
                )[: self.max_potholes]

                for (b, s) in selected:
                    x1, y1, x2, y2 = b
                    potholes.append([int(x1), int(y1), int(x2), int(y2), float(s), "pothole"])

        dets.extend(potholes)

        # Save for frame-skip reuse
        self._last_dets = dets
        return dets
