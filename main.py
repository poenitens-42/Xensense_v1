# ===========================================================
# XenSense main.py — High-FPS Clean ADAS (No Flicker Edition)
# - YOLO11n (general) every 2 frames, best.pt (potholes) every 3
# - MiDaS_small depth every 6 frames (CPU)
# - Medium retention (5 frames) to prevent alternating detections
# - Clean HUD: class + speed + dist; TTC only if < 3s
# - Lane-only + near-only + approaching + moving filters
# - Smoke enhancement toggle 'd'
# ===========================================================

import time
from collections import defaultdict, deque

import cv2
import yaml
import torch
import numpy as np
import simpleaudio as sa
from PIL import Image

from detector import Detector                  # must return [x1,y1,x2,y2,conf,cls] at input scale
from tracker.deep_sort import DeepSort
from speed_estimator import SpeedEstimator
from depth_estimator import DepthEstimator
from smoke_removal import enhance_smoke_region  # your helper: enhance_smoke_region(frame, mask)

# ---------------- Perf tweaks ----------------
try:
    cv2.setNumThreads(0)
except Exception:
    pass
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---------------- Config ----------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------- Beep ----------------
def play_beep(freq=900, ms=120):
    fs = 44100
    t = np.linspace(0, ms/1000, int(fs * ms / 1000), False)
    audio = (np.sin(freq * 2 * np.pi * t) * 32767).astype(np.int16)
    try:
        sa.play_buffer(audio, 1, 2, fs)
    except Exception:
        pass

# ---------------- Speed (wrapper) ----------------
class SpeedEstimatorOptimized(SpeedEstimator):
    def __init__(self, fps=30, pixel_to_meter=0.05, mode="hybrid"):
        super().__init__(fps=fps, pixel_to_meter=pixel_to_meter, mode=mode)
        if not hasattr(self, "last_states"):
            self.last_states = defaultdict(lambda: None)

# ---------------- Reasoner (EMA-TTC) ----------------
class SituationReasoner:
    def __init__(self, fps=30, center_band=(0.43, 0.57), ttc_warn_s=1.6, max_history=60):
        self.fps = fps
        self.center_band = center_band
        self.ttc_warn_s = ttc_warn_s
        self.max_history = max_history
        self.hist = defaultdict(lambda: deque(maxlen=max_history))
        self.ttc_ema = {}
        self.prev_dist = {}

    def _ema_ttc(self, tid, x):
        if x is None: return None
        prev = self.ttc_ema.get(tid, x)
        new = 0.7 * prev + 0.3 * x
        self.ttc_ema[tid] = new
        return new

    def update(self, W, H, tracks):
        events = []
        cx_min = W * self.center_band[0]
        cx_max = W * self.center_band[1]

        for track, cls_name, speed, dist in tracks:
            tid = track.track_id
            x, y, w, h = track.to_tlwh().astype(int)
            cx = x + w // 2

            self.hist[tid].append({"speed": speed, "dist": dist, "t": time.time()})

            prevd = self.prev_dist.get(tid, None)
            if dist is not None:
                self.prev_dist[tid] = dist

            ttc = None
            if (dist is not None) and (speed is not None) and (speed > 2) and (dist < 60):
                v_ms = speed / 3.6
                raw = dist / max(v_ms, 1e-6)
                if 0 < raw < 8 and (prevd is None or dist < prevd + 0.5):
                    ttc = raw
            ttc = self._ema_ttc(tid, ttc)

            if (ttc is not None) and (cx_min < cx < cx_max) and cls_name in ["car","truck","bus","person","motorbike","bicycle"]:
                if ttc < self.ttc_warn_s:
                    events.append(("imminent_collision", ttc, (x,y,w,h)))

        # return min TTC (if any) for global banner
        min_ttc = None
        if events:
            vals = [e[1] for e in events if e[1] is not None]
            if vals:
                min_ttc = min(vals)
        return events, min_ttc

# ---------------- Draw label ----------------
def draw_label(frame, x, y, text, color=(0,255,0), bg_alpha=0.65, padding=4):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x2 = x + tw + 2*padding
    y2 = y - th - 2*padding
    y2 = max(6, y2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y2), (x2, y), color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1-bg_alpha, 0, frame)
    cv2.putText(frame, text, (x+padding, y - padding - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# ===========================================================
# MAIN
# ===========================================================
def main():
    cfg = load_config()
    video_path = cfg.get("video_path") or input("Enter path to video file: ")

    # Toggles
    try:
        dehaze_enabled = (input("Enable smoke enhancement? (y/n): ").strip().lower() == "y")
    except Exception:
        dehaze_enabled = False

    # Schedules (performance)
    DETECT_EVERY = 2          # general YOLO every 2 frames
    POT_DETECT_EVERY = 3      # pothole YOLO every 3 frames (inside your Detector)
    DEPTH_SKIP = 6            # MiDaS every 6 frames
    RETAIN_FRAMES = 5         # medium retention (no flicker)

    # Filters (clean HUD)
    CENTER_BAND = tuple(cfg.get("reasoner", {}).get("center_band", [0.43, 0.57]))
    MAX_DIST_KEEP = 40.0          # meters
    MIN_MOVING_SPEED = 3.0         # km/h
    SHOW_TTC_IF_LT = 3.0           # seconds

    # Pothole pre-filters
    pothole_min_area = 120
    pothole_aspect_ratio_max = 1.2  # h <= 1.2 * w

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video:", video_path)
        return

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

   
    # -------- Unified Detector (compatible with your Detector class) --------
    det_cfg = cfg.get("detector", {})
    pothole_cfg = cfg.get("pothole_detector", {})

    # general detection model (YOLO11n)
    general_model_path = det_cfg.get("general_model", det_cfg.get("model", "yolo11n.pt"))
    general_conf = float(det_cfg.get("conf_thresh", 0.45))

    # ✅ IMPORTANT: your Detector constructor accepts only 2 positional args
    detector = Detector(
        general_model_path,
        general_conf
    )

    # restrict classes if config specifies
    allowed = det_cfg.get("allowed_classes", [])
    detector.allowed_classes = set(allowed) if allowed else None

    # pothole detector
    from ultralytics import YOLO
    pothole_model_path = pothole_cfg.get("model", "models/best.pt")
    pothole_conf = float(pothole_cfg.get("conf_thresh", 0.3))

    pothole_model = YOLO(pothole_model_path)
    if torch.cuda.is_available():
        pothole_model.to("cuda")



    # -------- Tracking & speed --------
    tracker = DeepSort(
        max_age     = cfg["tracker"]["max_age"],   # also retains tracks internally
        n_init      = cfg["tracker"]["n_init"],
        conf_thresh = cfg["tracker"]["conf_thresh"]
    )
    speed_est = SpeedEstimatorOptimized(
        fps=cfg["speed"]["fps"],
        pixel_to_meter=cfg["speed"]["pixel_to_meter"]
    )

    # -------- Reasoner --------
    reasoner = SituationReasoner(
        fps=cfg["speed"]["fps"],
        center_band=CENTER_BAND,
        ttc_warn_s=float(cfg.get("reasoner", {}).get("ttc_warn_s", 1.6)),
        max_history=int(cfg.get("reasoner", {}).get("max_history", 60))
    )

    # -------- Depth (CPU, MiDaS_small) --------
    depth_est = DepthEstimator(model_type="MiDaS_small", device="cpu")
    depth_resize = tuple(cfg.get("depth", {}).get("resize", [384,384]))
    depth_map = None
    depth_min = depth_max = None

    # Model input size for detectors
    model_w = int(cfg.get("input_size", {}).get("w", 640))
    model_h = int(cfg.get("input_size", {}).get("h", 384))

    # Detection cache to avoid recompute & to stabilize tracker
    last_dets_resized = []  # cached detections at resized scale
    frame_i = 0
    last_time = time.time()
    fps_smooth = 0.0

    # Visual retention (extra on top of DeepSort): keep last drawn tracks for a few frames
    retained = {}  # tid -> {"bbox":(x,y,w,h),"cls":str,"age":int,"speed":float,"dist":float,"ttc":float}

    print("[INFO] Running XenSense — High-FPS, No-Flicker")
    print(f"[INFO] dehaze_enabled={dehaze_enabled}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] End of video.")
            break

        now = time.time()
        inst_fps = 1.0 / max(now - last_time, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps
        last_time = now

        frame_i += 1
        vis = frame.copy()

        # Resize for detector
        resized = cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
        scale_x, scale_y = W / model_w, H / model_h

        # ---- Run detectors on schedule; reuse cache otherwise ----
        run_detector_this_frame = (frame_i % DETECT_EVERY == 0)
        if run_detector_this_frame:
            try:
                raw = detector.detect(resized)  # list of [x1,y1,x2,y2,conf,cls] in resized coords
            except Exception as e:
                print("[ERROR] detector.detect failed:", e)
                raw = last_dets_resized  # fall back to cache
            last_dets_resized = raw
        else:
            raw = last_dets_resized  # reuse previous detections

        # ---- Scale to original & pothole pre-filters ----
        detections = []
        for x1,y1,x2,y2,conf,cls in raw:
            X1 = int(x1 * scale_x); Y1 = int(y1 * scale_y)
            X2 = int(x2 * scale_x); Y2 = int(y2 * scale_y)
            w = X2 - X1; h = Y2 - Y1
            if str(cls).lower() == "pothole":
                area = w * h
                if area < pothole_min_area:
                    continue
                if h > (w * pothole_aspect_ratio_max):
                    continue
                if Y2 < int(H * 0.55):  # must be lower-half
                    continue
            detections.append([X1, Y1, X2, Y2, float(conf), cls])

        # ---- Track ----
        tracks = tracker.update_tracks(detections, vis)

        # ---- Depth update (less frequent) ----
        if frame_i % DEPTH_SKIP == 0:
            try:
                small = cv2.resize(frame, depth_resize, interpolation=cv2.INTER_AREA)
                dsmall = depth_est.estimate_depth(small)
                if dsmall is not None and dsmall.size > 0 and not np.isnan(dsmall).any():
                    depth_map = cv2.resize(dsmall.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    depth_min = float(np.min(depth_map))
                    depth_max = float(np.max(depth_map))
                    if (depth_max - depth_min) < 1e-6:
                        depth_map = None
                # else keep previous depth
            except Exception:
                pass

        # ---- Prepare filters ----
        cx_min_pix = int(W * CENTER_BAND[0])
        cx_max_pix = int(W * CENTER_BAND[1])

        # ---- For reasoner feed + drawing ----
        tracks_for_reasoner = []
        active_ids_this_frame = set()

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x, y, w, h = tr.to_tlwh().astype(int)
            cx, cy = x + w//2, y + h//2
            cls_name = getattr(tr, "cls", "object")

            # Depth → meters (conservative linear)
            approx_m = None
            if depth_map is not None and (depth_max is not None and depth_min is not None) and (depth_max - depth_min > 1e-6):
                cyc = int(np.clip(cy, 0, H-1))
                cxc = int(np.clip(cx, 0, W-1))
                rawd = float(depth_map[cyc, cxc])
                norm = np.clip((rawd - depth_min) / (depth_max - depth_min), 0.0, 1.0)
                approx_m = 2.0 + 30.0 * norm
                if approx_m > 80:  # junk
                    approx_m = None

            # Speed (km/h)
            speed = speed_est.estimate_speed(tr.track_id, (cx, cy), h, cls_name)

            # Approaching?
            prevd = reasoner.prev_dist.get(tr.track_id, None)
            approaching = True
            if (approx_m is not None) and (prevd is not None) and (approx_m >= prevd + 0.5):
                approaching = False

            # Keep only lane, near, moving, approaching
            keep = True
            if approx_m is None or approx_m > MAX_DIST_KEEP: keep = False
            if speed is None or speed < MIN_MOVING_SPEED:     keep = False
            if not (cx_min_pix < cx < cx_max_pix):            keep = False
            if not approaching:                                keep = False

            # Compute TTC (for label, not mandatory)
            ttc = None
            if keep and (speed is not None) and (speed > 2):
                v_ms = speed / 3.6
                raw_ttc = approx_m / max(v_ms, 1e-6)
                if 0 < raw_ttc < 8:
                    ttc = raw_ttc

            # Retention bookkeeping (we keep even if later filtered out visually)
            active_ids_this_frame.add(tr.track_id)
            retained[tr.track_id] = {
                "bbox": (x, y, w, h),
                "cls": cls_name,
                "age": 0,  # reset age when seen
                "speed": speed,
                "dist": approx_m,
                "ttc": ttc
            }

            # Add to reasoner regardless (to maintain TTC smoothing/history)
            tracks_for_reasoner.append((tr, cls_name, speed, approx_m))

        # ---- Age & draw retained tracks (prevents flicker) ----
        to_delete = []
        for tid, info in retained.items():
            if tid not in active_ids_this_frame:
                info["age"] += 1
                # give it a chance up to RETAIN_FRAMES
                if info["age"] > RETAIN_FRAMES:
                    to_delete.append(tid)

            x, y, w, h = info["bbox"]
            cls_name = info["cls"]
            speed = info["speed"]
            dist = info["dist"]
            ttc = info["ttc"]

            # Apply the same visual filters at draw-time to keep HUD clean
            cx = x + w//2
            if dist is None or dist > MAX_DIST_KEEP: continue
            if speed is None or speed < MIN_MOVING_SPEED: continue
            if not (cx_min_pix < cx < cx_max_pix): continue

            color = (0, 200, 0) if cls_name != "pothole" else (0, 0, 200)
            label_parts = [str(cls_name)]
            if speed is not None: label_parts.append(f"{speed:.0f}km/h")
            if dist is not None:  label_parts.append(f"{dist:.1f}m")
            draw_label(vis, x, y, " ".join(label_parts), color=color)

            if (ttc is not None) and (ttc < SHOW_TTC_IF_LT):
                draw_label(vis, x, y + 30, f"TTC {ttc:.1f}s", color=(0,0,230) if ttc >= 1.0 else (0,0,255))

        for tid in to_delete:
            retained.pop(tid, None)

        # ---- Reasoner & global banner ----
        events, min_ttc = reasoner.update(W, H, tracks_for_reasoner)
        if (min_ttc is not None) and (min_ttc < reasoner.ttc_warn_s):
            text = f"IMMINENT COLLISION — TTC {min_ttc:.1f}s"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)
            bx = max(10, (W - tw) // 2)
            by = 50
            overlay = vis.copy()
            cv2.rectangle(overlay, (bx-10, by-th-12), (bx+tw+10, by+10), (0,0,200), -1)
            cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
            cv2.putText(vis, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 4)
            play_beep(950, 120)

        # ---- Smoke enhancement (toggle) ----
        if dehaze_enabled:
            try:
                vis = enhance_smoke_region(vis, np.ones((H, W), dtype=np.uint8) * 255)
            except Exception:
                pass

        # ---- FPS HUD ----
        cv2.putText(vis, f"FPS: {fps_smooth:.1f}", (20, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # ---- Show ----
        cv2.imshow("XenSense — High-FPS Clean ADAS", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('d'):
            dehaze_enabled = not dehaze_enabled
            print("[INFO] Smoke enhancement:", dehaze_enabled)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("[ERROR] Unhandled exception in main():")
        traceback.print_exc()
