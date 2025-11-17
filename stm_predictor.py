import numpy as np
import cv2
from utils.math_utils import distance, sigmoid, exponential_smooth

class STMPredictor:
    """
    Hybrid Spatio-Temporal Predictor:
    - Combines motion history (Kalman-like updates)
    - Depth-aware risk estimation
    - Predicts next 1.5s trajectory per track
    - Outputs smoothed risk, TTC, and predicted path
    """

    def __init__(self, fps=30, history_len=30, future_horizon=45):
        self.fps = fps
        self.history_len = history_len
        self.future_horizon = future_horizon
        self.memory = {}  # tid -> [(x, y, dist, speed, time)]
        self.last_risks = {}  # EMA for stable HUD visualization

    # ------------------------------------------------------------------
    def update(self, tracks, depth_map=None):
        preds = {}
        now = 0.0

        for tr, cls, speed, dist in tracks:
            tid = tr.track_id
            x, y, w, h = tr.to_tlwh().astype(int)
            cx, cy = x + w // 2, y + h // 2

            if tid not in self.memory:
                self.memory[tid] = []
            self.memory[tid].append((cx, cy, dist, speed, now))
            self.memory[tid] = self.memory[tid][-self.history_len:]

            if len(self.memory[tid]) < 2:
                continue

            (x1, y1, _, _, _), (x2, y2, _, _, _) = self.memory[tid][-2:]
            vx = (x2 - x1)
            vy = (y2 - y1)

            pred_path = [(int(x2 + vx * k * 0.5), int(y2 + vy * k * 0.5))
                         for k in range(1, self.future_horizon + 1)]

            # Risk calculation
            base_risk = sigmoid((50 - dist) / 10) * sigmoid(speed / 10) if dist and speed else 0.0
            depth_risk = 0.0
            if depth_map is not None and 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                local = depth_map[max(0, cy - 4):cy + 4, max(0, cx - 4):cx + 4]
                if local.size > 0:
                    var = np.var(local)
                    depth_risk = sigmoid(var * 8)

            total_risk = 0.7 * base_risk + 0.3 * depth_risk
            prev_risk = self.last_risks.get(tid, total_risk)
            smooth_risk = exponential_smooth(total_risk, 0.5 * prev_risk + 0.5 * total_risk)
            self.last_risks[tid] = smooth_risk

            preds[tid] = {
                "pred_path": pred_path,
                "risk_score": smooth_risk,
                "pred_ttc": dist / (speed / 3.6 + 1e-6) if dist and speed else None,
            }

        return preds

    # ------------------------------------------------------------------
    def draw_predictions(self, frame, stm_results):
        """
        Draw predicted trajectories and risk indicators.
        - Green: low risk
        - Yellow: medium
        - Red: high
        """
        if not stm_results:
            return frame

        for tid, info in stm_results.items():
            path = info.get("pred_path", [])
            risk = info.get("risk_score", 0.0)

            if risk < 0.3:
                color = (0, 255, 0)
            elif risk < 0.6:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], color, 2)

            if path:
                px, py = path[-1]
                cv2.putText(frame, f"Risk:{risk:.2f}", (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame
