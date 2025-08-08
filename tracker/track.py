import numpy as np


class Track:
    """
    A single target track with its state, managed by the Kalman filter.
    """

    def __init__(self, mean, covariance, track_id, n_init=3, max_age=30, cls="unknown"):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = "tentative"
        self.n_init = n_init
        self.max_age = max_age
        self.cls = cls

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.hits += 1
        self.time_since_update = 0

        if self.state == "tentative" and self.hits >= self.n_init:
            self.state = "confirmed"

        self.cls = detection.cls

    def mark_missed(self):
        if self.state == "tentative":
            self.state = "deleted"
        elif self.time_since_update > self.max_age:
            self.state = "deleted"

    def is_tentative(self):
        return self.state == "tentative"

    def is_confirmed(self):
        return self.state == "confirmed"

    def is_deleted(self):
        return self.state == "deleted"

    def to_tlbr(self):
        """Convert state mean to (x1, y1, x2, y2)."""
        x, y, a, h = self.mean[:4]
        w = a * h
        return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)

    def to_tlwh(self):
        """Convert state mean to (x, y, w, h)."""
        x, y, a, h = self.mean[:4]
        w = a * h
        return np.array([x - w / 2, y - h / 2, w, h], dtype=np.float32)
