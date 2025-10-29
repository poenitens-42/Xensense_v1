import math
from collections import defaultdict

class SpeedEstimator:
    def __init__(self, fps=30, pixel_to_meter=0.05, mode="depth"):
        """
        Args:
            fps (int): video FPS
            pixel_to_meter (float): scaling factor from pixels to meters
            mode (str): "pixel", "depth", or "hybrid"
        """
        self.last_states = defaultdict(lambda: None)  # track_id -> (center, height, depth)
        self.pixel_to_meter = pixel_to_meter
        self.fps = fps
        self.mode = mode

    def estimate_speed(self, track_id, center, height, cls_name, depth=None):
        """
        Estimate speed of tracked object.
        """
        last = self.last_states[track_id]
        self.last_states[track_id] = (center, height, depth)

        if last is None:
            return 0.0

        (x1, y1), _, last_depth = last
        (x2, y2) = center

        #   Pixel displacement (lateral motion)
        dist_pixels = math.hypot(x2 - x1, y2 - y1)
        dist_meters_pixel = dist_pixels * self.pixel_to_meter

        #  Depth displacement (forward/backward motion)
        dist_meters_depth = 0.0
        if depth is not None and last_depth is not None:
            dist_meters_depth = abs(depth - last_depth) * self.pixel_to_meter

        #  Mode selection 
        if self.mode == "pixel":
            total_dist_meters = dist_meters_pixel
        elif self.mode == "depth":
            total_dist_meters = dist_meters_depth
        else:  # hybrid
            total_dist_meters = math.sqrt(dist_meters_pixel**2 + dist_meters_depth**2)

        # Convert to speed (m/s → km/h)

        speed_mps = total_dist_meters * self.fps
        return speed_mps * 3.0
