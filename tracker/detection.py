import numpy as np

class Detection:
    def __init__(self, tlwh, confidence, cls):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.cls = cls

    def to_tlbr(self):
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    def to_xyah(self):
        x, y, w, h = self.tlwh
        cx = x + w / 2
        cy = y + h / 2
        aspect_ratio = w / float(h)
        return np.array([cx, cy, aspect_ratio, h], dtype=np.float32)
