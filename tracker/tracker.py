import numpy as np
from .kalman_filter import KalmanFilter

class Track:
    def __init__(self, detection, track_id, n_init=3):
        self.track_id = track_id
        self.cls = detection.cls
        self.mean, self.cov = KalmanFilter().initiate(detection.to_xyah())
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.confirmed = False

    def update(self, detection):
        self.mean, self.cov = KalmanFilter().update(self.mean, self.cov, detection.to_xyah())
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.n_init:
            self.confirmed = True

    def to_tlwh(self):
        cx, cy, ar, h = self.mean[:4]
        w = ar * h
        x = cx - w / 2
        y = cy - h / 2
        return np.array([x, y, w, h], dtype=np.float32)

    def is_confirmed(self):
        return self.confirmed


class Tracker:
    def __init__(self, max_age=30, n_init=3):
        self.tracks = []
        self.next_id = 1
        self.kf = KalmanFilter()
        self.max_age = max_age
        self.n_init = n_init

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            match = None
            for track in self.tracks:
                if track.cls == det.cls:
                    match = track
                    break
            if match:
                match.update(det)
                updated_tracks.append(match)
            else:
                new_track = Track(det, self.next_id, self.n_init)
                self.next_id += 1
                updated_tracks.append(new_track)

        self.tracks = [t for t in updated_tracks if t.time_since_update <= self.max_age]
        return self.tracks
