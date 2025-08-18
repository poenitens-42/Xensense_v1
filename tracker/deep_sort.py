import numpy as np

class Track:
    def __init__(self, track_id, bbox, cls_name):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.cls_name = cls_name
        self.cls = cls_name  #  for compatibility with old code
        self.age = 0
        self.missed = 0

    def update(self, bbox):
        self.bbox = bbox
        self.missed = 0
        self.age += 1

    def to_tlwh(self):
        return np.array(self.bbox, dtype=float)

    def is_confirmed(self):
        return True


class DeepSort:
    def __init__(self, max_age=30, n_init=3, conf_thresh=0.5):
        self.max_age = max_age
        self.n_init = n_init
        self.conf_thresh = conf_thresh
        self.next_id = 1
        self.tracks = []

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update_tracks(self, detections, frame):
        updated_ids = set()

        for det in detections:
            x1, y1, x2, y2, conf, cls_name = det
            if conf < self.conf_thresh:
                continue
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # Match with existing track
            best_iou = 0
            best_track = None
            for track in self.tracks:
                if track.cls_name != cls_name:
                    continue
                iou_score = self.iou(bbox, track.bbox)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track = track

            if best_track and best_iou > 0.3:
                best_track.update(bbox)
                updated_ids.add(best_track.track_id)
            else:
                # New track
                new_track = Track(self.next_id, bbox, cls_name)
                self.tracks.append(new_track)
                updated_ids.add(self.next_id)
                self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.track_id in updated_ids or t.missed <= self.max_age]

        return self.tracks
