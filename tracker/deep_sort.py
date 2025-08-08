from .tracker import Tracker
from .detection import Detection

class DeepSort:
    def __init__(self, max_age=30, n_init=3, conf_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.tracker = Tracker(max_age=max_age, n_init=n_init)

    def update_tracks(self, detections, frame=None):
        formatted_dets = []
        for tlwh, conf, cls in detections:
            if conf >= self.conf_thresh:
                formatted_dets.append(Detection(tlwh, conf, cls))
        self.tracker.update(formatted_dets)
        return self.tracker.tracks
