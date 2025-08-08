import math
from collections import defaultdict

class SpeedEstimator:
    def __init__(self, fps=30, pixel_to_meter=0.05):
        self.last_positions = defaultdict(lambda: None)
        self.pixel_to_meter = pixel_to_meter
        self.fps = fps

    def estimate_speed(self, track_id, center, height, cls_name):
        last = self.last_positions[track_id]
        self.last_positions[track_id] = (center, height)

        if last is None:
            return 0.0

        (x1, y1), _ = last
        (x2, y2) = center
        dist_pixels = math.hypot(x2 - x1, y2 - y1)
        dist_meters = dist_pixels * self.pixel_to_meter
        speed_mps = dist_meters * self.fps
        return speed_mps * 3.6
