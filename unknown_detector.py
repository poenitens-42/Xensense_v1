import cv2
import numpy as np

class UnknownObjectDetector:
    """
    Detects moving or obstructing objects missed by YOLO
    using optical flow + depth map cues.
    """

    def __init__(self, min_area=400, motion_thresh=1.2, depth_near=3, depth_far=50):
        self.prev_gray = None
        self.prev_depth = None
        self.min_area = min_area
        self.motion_thresh = motion_thresh
        self.depth_near = depth_near
        self.depth_far = depth_far

    def detect_unknowns(self, frame, depth_map=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_depth = depth_map
            return []

        # Optical flow (motion magnitude)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = (mag > self.motion_thresh).astype(np.uint8) * 255

        # Optional depth gating
        if depth_map is not None and self.prev_depth is not None:
            depth_diff = cv2.absdiff(depth_map, self.prev_depth)
            close_mask = ((depth_map > self.depth_near) &
                          (depth_map < self.depth_far)).astype(np.uint8) * 255
            motion_mask = cv2.bitwise_and(motion_mask, close_mask)

        # Morphological cleanup
        motion_mask = cv2.medianBlur(motion_mask, 5)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append((x, y, w, h, area))

        self.prev_gray = gray
        self.prev_depth = depth_map
        return detections
