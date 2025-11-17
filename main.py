import os
import csv
from datetime import datetime
import time
import cv2
import yaml
import torch
import numpy as np
import json


from collections import defaultdict, deque
# from unknown_detector import UnknownObjectDetector
from detector import Detector
from tracker.deep_sort import DeepSort
from speed_estimator import SpeedEstimator
from depth_estimator import DepthEstimator
from stm_predictor import STMPredictor
from stm_lstm_predictor import STMLSTMPredictor
from prompt_detector import PromptedDetector
from video_prompt_processor import VideoPromptProcessor



# ---------------- Perf tweaks ----------------
try:
    cv2.setNumThreads(0)
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ---------------- Config loader ----------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------- Non-blocking beep ----------------
def play_beep(freq=1000, ms=120):
    import threading, platform
    def _beep():
        try:
            if platform.system().lower().startswith("win"):
                import winsound
                winsound.Beep(int(freq), int(ms))
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


def load_prompt_profile(name):
    with open("prompt_profiles.json", "r") as f:
        profiles = json.load(f)
    return profiles.get(name, [])

def save_prompt_profile(name, prompt_list):
    with open("prompt_profiles.json", "r") as f:
        profiles = json.load(f)
    profiles[name] = prompt_list
    with open("prompt_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

# ---------------- Quick dehaze ----------------
def quick_dehaze_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ---------------- Draw label ----------------
def draw_label(frame, x, y, text, color=(0,255,0), bg_alpha=0.7, padding=6, font_scale=0.55, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x2 = int(x + tw + padding * 2)
    y2 = int(y - th - padding * 2)
    y2 = max(6, y2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (int(x), y2), (x2, int(y)), color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (int(x + padding), int(y - padding/2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

# ---------------- Telemetry Logger ----------------

class TelemetryLogger:
    def __init__(self, base_dir="logs"):
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = os.path.join(base_dir, f"session_{ts}.csv")
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
    "timestamp", "frame", "track_id", "source", "class",
    "dist_m", "speed_kmh", "TTC_s", "ml_risk", "stm_risk", "global_risk",
    "crash_side",
    "x", "y", "w", "h"
])


        print(f"[LOG] Telemetry file created: {self.filepath}")

    def log(self, frame_i, track, cls, dist, speed, ttc, ml_risk, stm_risk, global_risk, source="YOLO"):

        x, y, w, h = (0, 0, 0, 0)
        try:
            x, y, w, h = track.to_tlwh().astype(int)
        except Exception:
            pass
        ts = time.time()
        crash_side = getattr(track, "crash_side", "") if hasattr(track, "crash_side") or isinstance(track, dict) else ""
# Note: track may be track_obj or the retained entry; safer to check retained by track_id outside of this function.
        self.writer.writerow([
            ts, frame_i, getattr(track, "track_id", None), cls,
            dist if dist is not None else "",
            speed if speed is not None else "",
            ttc if ttc is not None else "",
            ml_risk if ml_risk is not None else "",
            stm_risk if stm_risk is not None else "",
            global_risk if global_risk is not None else "",
            crash_side,
            x, y, w, h
        ])



    def close(self):
        if self.file:
            self.file.close()
            print(f"[LOG] Telemetry file saved: {self.filepath}")






# ---------------- Speed est wrapper ----------------
class SpeedEstimatorOptimized(SpeedEstimator):
    def __init__(self, fps=30, pixel_to_meter=0.05, mode="hybrid"):
        super().__init__(fps=fps, pixel_to_meter=pixel_to_meter, mode=mode)
        if not hasattr(self, "last_states"):
            self.last_states = defaultdict(lambda: None)
        self.speed_ema = {}
        self.ema_alpha = 0.85

# ---------------- Reasoner (TTC smoothing) ----------------
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
        if x is None:
            return None
        prev = self.ttc_ema.get(tid, x)
        new = 0.7 * prev + 0.3 * x
        self.ttc_ema[tid] = new
        return new

    def update(self, W, H, tracks):
        # global retained
        events = []
        cx_min = W * self.center_band[0]
        cx_max = W * self.center_band[1]

        for track, cls_name, speed, dist in tracks:
            tid = track.track_id
            x, y, w, h = track.to_tlwh().astype(int)
            cx = x + w // 2

            prevd = self.prev_dist.get(tid, None)
            if dist is not None:
                self.prev_dist[tid] = dist

            # TTC formula
            ttc = None
            if speed is not None and dist is not None:
                v_ms = max(speed / 3.6, 0.01)
                ttc = dist / v_ms

                # allow up to 15s now
                if 0 < ttc < 15:
                    prev_ttc = getattr(track, "ttc_prev", ttc)
                    ttc = 0.65 * prev_ttc + 0.35 * ttc
                    setattr(track, "ttc_prev", ttc)



            ttc = self._ema_ttc(tid, ttc)

            if (ttc is not None) and (cx_min < cx < cx_max):
                # Broaden collision check beyond vehicles
                risky_classes = ["car","truck","bus","motorbike","bicycle","person","deer","animal","dog","cow","horse"]
                if cls_name.lower() in risky_classes and ttc < self.ttc_warn_s:
                    events.append(("collision_risk", ttc, (x,y,w,h)))
                # Also catch any new OwlViT prompt-based classes dynamically
                elif "unknown" not in cls_name.lower() and ttc < (self.ttc_warn_s * 0.8):
                    events.append(("collision_risk_generic", ttc, (x,y,w,h)))


        min_ttc = min((e[1] for e in events), default=None)
        return events, min_ttc

# ---------------- Helper: draw dotted polyline ----------------
def draw_dotted_polyline(img, pts, color=(0,255,0), thickness=2, gap=6, dot_len=4, alpha=1.0):
    # pts: list of (x,y)
    if len(pts) < 2:
        return
    overlay = img.copy()
    for i in range(len(pts)-1):
        x1,y1 = int(pts[i][0]), int(pts[i][1])
        x2,y2 = int(pts[i+1][0]), int(pts[i+1][1])
        seg_len = int(np.hypot(x2-x1, y2-y1))
        if seg_len == 0:
            continue
        # number of dots on this segment
        step = dot_len + gap
        num = max(1, seg_len // step)
        for j in range(num):
            t0 = j * step / seg_len
            t1 = min((j * step + dot_len) / seg_len, 1.0)
            sx = int(x1 + (x2-x1) * t0)
            sy = int(y1 + (y2-y1) * t0)
            ex = int(x1 + (x2-x1) * t1)
            ey = int(y1 + (y2-y1) * t1)
            cv2.line(overlay, (sx,sy), (ex,ey), color, thickness, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        img[:] = overlay




# MAIN

def main():
    cfg = load_config()
    video_path = cfg.get("video_path") or input("Enter path to video file: ")

    try:
        dehaze_enabled = (input("Enable smoke enhancement? (y/n): ").strip().lower() == "y")
    except Exception:
        dehaze_enabled = False

    DETECT_EVERY = int(cfg.get("detector", {}).get("detect_every_n", 2))
    DEPTH_SKIP   = int(cfg.get("depth", {}).get("skip", 6))
    CENTER_BAND = tuple(cfg.get("reasoner", {}).get("center_band", [0.40, 0.60]))
    SHOW_TTC_IF_LT = float(cfg.get("reasoner", {}).get("ttc_warn_s", 2.0))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video:", video_path)
        return

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Detector
    # --- Detector: Dual-Model (General YOLO + Pothole Model) ---
    det_cfg = cfg.get("detector", {})
    general_model = det_cfg.get("general_model", "yolo11n.pt")
    pothole_model = det_cfg.get("pothole_model", "models/best.pt")
    conf = float(det_cfg.get("conf_thresh", 0.45))
    pothole_conf = float(det_cfg.get("pothole_conf", 0.30))

    detector = Detector(
        general_model_path=general_model,
        conf_thresh=conf,
        pothole_model_path=pothole_model,
        pothole_conf=pothole_conf,
        frame_skip=1
    )

    allowed = det_cfg.get("allowed_classes", [])
    detector.allowed_classes = set(allowed) if allowed else None


    # Tracker & speed
    tracker = DeepSort(
        max_age     = cfg["tracker"]["max_age"],
        n_init      = cfg["tracker"]["n_init"],
        conf_thresh = cfg["tracker"]["conf_thresh"]
    )
    speed_est = SpeedEstimatorOptimized(
        fps=cfg["speed"]["fps"],
        pixel_to_meter=cfg["speed"]["pixel_to_meter"]
    )

    # Reasoner / predictors
    reasoner = SituationReasoner(
        fps=cfg["speed"]["fps"],
        center_band=CENTER_BAND,
        ttc_warn_s=SHOW_TTC_IF_LT
    )
    stm = STMPredictor(future_horizon=30, fps=cfg["speed"]["fps"])
    stm_lstm = STMLSTMPredictor()

    # Depth
    depth_device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_est = DepthEstimator("MiDaS_small", depth_device)
    depth_resize = tuple(cfg.get("depth", {}).get("resize", [384,384]))
    depth_map = None
    depth_min = depth_max = None

    print("[INFO] Initializing Prompt-based Detection...")
    prompt_detector = PromptedDetector()
    video_prompt = VideoPromptProcessor()
    prompt_enabled = True   # default ON



    # ---------------------------------------
    # Heatmap Runtime Toggles
    # ---------------------------------------
    motion_heatmap_enabled = True
    depth_heatmap_enabled = True

    prev_gray = None
    motion_heat = None


    # 🔹 Ask user for detection prompt (works even in VS Code)
    try:
        print("\n---------------------------------------")
        print("   Custom Zero-Shot Object Detection")
        print("---------------------------------------")
        user_prompt = input("Enter detection prompt (comma-separated): ").strip()
    except EOFError:
        # fallback when input() doesn't work (e.g., non-interactive shell)
        user_prompt = ""

    # ---------------------------------------
    # Video Enhancement Prompt Input
    # ---------------------------------------
    try:
        print("\n---------------------------------------")
        print("   Video Enhancement Prompt System")
        print("---------------------------------------")
        enh = input("Describe how to enhance video (e.g., 'remove fog', 'make clearer'): ").strip()

    except EOFError:
        enh = ""

    if enh:
        video_prompt.set_prompt(enh)
        video_prompt.enabled = True
        print(f"[INFO] Video Enhancement Enabled: {enh}")
    else:
        print("[INFO] No enhancement prompt applied.")



    if not user_prompt.strip():
        # User pressed ENTER → disable prompt detection entirely
        print("[INFO] No prompt entered. Prompt-based detection DISABLED.")
        prompt_enabled = False
        prompt_list = []
    else:
        print(f"[INFO] Using custom prompt: {user_prompt}")
        prompt_enabled = True
        prompt_list = [p.strip() for p in user_prompt.split(",") if p.strip()]

    # Convert string -> list for OwlViT
    prompt_list = [p.strip() for p in user_prompt.split(",") if p.strip()]
    print(f"[INFO] Final prompt list: {prompt_list}\n")



    # Model size
    model_w = int(cfg.get("input_size", {}).get("w", 640))
    model_h = int(cfg.get("input_size", {}).get("h", 384))

    # State
    last_dets = []
    frame_i = 0
    retained = {}
    # global ML risk EMA
    ml_risk_ema = 0.0
    ml_risk_alpha = 0.85  # smooth quick changes
    ml_risk_peak = 0.0

    print("[INFO] Running XenSense — Pro HUD (Hybrid Audi-Tesla)")
    print(f"[INFO] depth_device: {depth_device}")

    telemetry = TelemetryLogger()
    # unknown_detector = UnknownObjectDetector()


    while True:
        ok, frame = cap.read()
        # Apply user-selected video enhancement to frame
        frame = video_prompt.process(frame)

        if not ok:
            print("[INFO] End of video.")
            break

        # -------------------------------------------------------
        # Prompt-based object detection (OwlViT — Zero-Shot Vision)
        # -------------------------------------------------------
        if prompt_enabled and frame_i % 10 == 0:  # every 10th frame for efficiency
            try:
                prompt_dets = prompt_detector.detect(frame, prompt=prompt_list)
                frame = prompt_detector.draw(frame, prompt_dets)

                # ⚠️ Integrate with ML risk system
                for box, label, score in zip(
                    prompt_dets["boxes"], prompt_dets["labels"], prompt_dets["scores"]
                ):
                    risk_signal = 0.0  # <-- Initialize here every iteration
                    label = str(label).lower()
                    if score < 0.15:
                        continue

                    # estimate approximate TTC from bounding box size
                    x1, y1, x2, y2 = map(int, box)
                    box_area = (x2 - x1) * (y2 - y1)
                    normalized_area = box_area / (W * H)

                    # crude distance estimate (smaller box = farther)
                    est_dist = max(2.0, 80.0 * (1 - normalized_area * 10))
                    ttc = None

                    # Assign a virtual TTC for animals/pedestrians
                    if label in ["deer", "animal", "person", "cow", "dog", "horse"]:
                        ttc = 3.0 * (1.0 - normalized_area * 10)
                        ttc = max(0.5, min(ttc, 10.0))  # clamp 0.5–10s range

                        # add risk bump for near static animal detections
                        if ttc is not None and ttc < 3.0:
                            # Higher risk when TTC is small
                            risk_signal = 1.0 - (ttc / 3.0)
                            risk_signal = float(np.clip(risk_signal, 0.0, 1.0))

                            # Smoothly update risk using exponential moving average (EMA)
                            ml_risk_ema = (1 - ml_risk_alpha) * risk_signal + ml_risk_alpha * ml_risk_ema
                            ml_risk_ema = float(np.clip(ml_risk_ema, 0.0, 1.0))

                            # Track peak risk (for visualization)
                            ml_risk_peak = max(ml_risk_peak, ml_risk_ema)


                            # 🔹 make it globally persistent for HUD blending
                            globals()["prompt_risk_signal"] = max(globals().get("prompt_risk_signal", 0.0), risk_signal)

                            # Visual alert
                            cv2.putText(frame, f"⚠️ {label.upper()} DETECTED", (x1, max(30, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            play_beep(900, 100)


            except Exception as e:
                print(f"[ERROR] OwlViT prompt detection failed: {e}")






        vis = frame.copy()
                # -------------------------------------------------------
        # DETECTION — FINAL FIX
        # -------------------------------------------------------
        if frame_i % DETECT_EVERY == 0:
            try:
                dets = detector.detect(vis)
                last_dets = dets
            except Exception as e:
                print("[WARN] detector failed:", e)
                dets = last_dets
        else:
            dets = last_dets

        sx = 1.0
        sy = 1.0









        # -------------------------------------------------------
        # FIX 8 — OPTICAL FLOW MOTION HEATMAP (toggle: M)
        # -------------------------------------------------------
        if motion_heatmap_enabled:
            if prev_gray is None:
                prev_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                motion_heat = np.zeros((H, W), dtype=np.float32)
            else:
                gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                motion_heat = 0.75 * motion_heat + 0.25 * mag
                prev_gray = gray

            motion_norm = cv2.normalize(motion_heat, None, 0, 255, cv2.NORM_MINMAX)
            motion_colored = cv2.applyColorMap(motion_norm.astype(np.uint8), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.85, motion_colored, 0.35, 0)




        frame_i += 1
        reasoner_inputs = []  # ensure it's defined each frame

        # optional dehaze
        if dehaze_enabled:
            vis = quick_dehaze_bgr(vis)

        # detection resize



        # Run detector on schedule

        # ---------------------------------------------------
        # # Fallback: detect unknown objects if YOLO finds none
        # # ---------------------------------------------------
        # center_lane_x = W // 2
        # road_band = (int(W * 0.3), int(W * 0.7))  # focus on center 40% width

        # # Check if any detection in road band
        # in_path = any((road_band[0] < (x1 + x2) // 2 < road_band[1]) for x1, y1, x2, y2, _, _ in dets)

        # if not in_path:  # YOLO sees nothing blocking road
        #     unknowns = unknown_detector.detect_unknowns(vis, depth_map)
        #     for (ux, uy, uw, uh, area) in unknowns:
        #         # Append synthetic detection
        #         dets.append([ux, uy, ux + uw, uy + uh, 0.4, "unknown_obstacle"])

        #         # Draw temporary bounding box (debug)
        #         cv2.rectangle(vis, (ux, uy), (ux + uw, uy + uh), (255, 200, 0), 2)
        #         cv2.putText(vis, "Unknown Obstacle", (ux, uy - 5),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)




        # Scale boxes up
        scaled = []
        for x1, y1, x2, y2, confv, clv in dets:
            X1, Y1 = int(x1 * sx), int(y1 * sy)
            X2, Y2 = int(x2 * sx), int(y2 * sy)
            scaled.append([X1, Y1, X2, Y2, confv, clv])



        # -------------------------------------------------------
        # LARGE OBJECT MODE — detect extremely close objects
        # -------------------------------------------------------
        huge_boxes = []
        for (X1, Y1, X2, Y2, confv, clv) in scaled:
            box_w = X2 - X1
            box_h = Y2 - Y1
            box_area = box_w * box_h

            if box_area > 0.35 * (W * H):     # object covers >35% of screen
                huge_boxes.append((X1, Y1, X2, Y2, confv, clv))

        # If ANY huge close object appears → treat as hazard even before classification
        if huge_boxes:
            ml_risk_ema = 1.0
            ml_risk_peak = 1.0

            for (X1, Y1, X2, Y2, confv, clv) in huge_boxes:
                cv2.putText(vis, "CLOSE OBJECT!", (X1, Y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                play_beep(1000, 200)




        



        # Track
        tracks = tracker.update_tracks(scaled, vis)

        # Depth update (infrequent)
        if frame_i % DEPTH_SKIP == 0:
            try:
                small = cv2.resize(vis, depth_resize)
                dsmall = depth_est.estimate_depth(small)
                if dsmall is not None and not np.isnan(dsmall).any():
                    depth_map = cv2.resize(dsmall.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    depth_min = float(np.min(depth_map)); depth_max = float(np.max(depth_map))
                    if (depth_max - depth_min) < 1e-6:
                        depth_map = None
            except Exception:
                pass

        

        # -------------------------------------------------------
        # FIX 9 — DEPTH VARIATION HEATMAP
        # -------------------------------------------------------
        # -------------------------------------------------------
        # FIX 9 — DEPTH VARIATION HEATMAP (toggle: N)
        # -------------------------------------------------------
        if depth_heatmap_enabled and depth_map is not None:
            dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad = cv2.magnitude(dx, dy)

            depth_norm = cv2.normalize(depth_grad, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_HOT)

            vis = cv2.addWeighted(vis, 0.87, depth_colored, 0.33, 0)

        

        # Smooth bounding boxes (reduce jitter)
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            cur = tr.to_tlwh().astype(float)
            if not hasattr(tr, "smooth_tlwh"):
                tr.smooth_tlwh = cur
            tr.smooth_tlwh = 0.60 * tr.smooth_tlwh + 0.40 * cur

        # -------------------------------------------------------
        # Build retained dictionary: distance + smoothed speed
        # -------------------------------------------------------
        active_now = set()
        cx_band_min = int(W * CENTER_BAND[0])
        cx_band_max = int(W * CENTER_BAND[1])

        for tr in tracks:
            if not tr.is_confirmed():
                continue

            x, y, w, h = tr.smooth_tlwh.astype(int)
            cx, cy = x + w // 2, y + h // 2
            tid = tr.track_id
            cls_name = getattr(tr, "cls", "object")

            # Depth → meters (3x3 median)
            approx_m = None
            if depth_map is not None and depth_min is not None and depth_max is not None:
                px = int(np.clip(cx, 1, W - 2)); py = int(np.clip(cy, 1, H - 2))
                patch = depth_map[py-1:py+2, px-1:px+2]
                raw = float(np.median(patch))
                norm = (raw - depth_min) / max(depth_max - depth_min, 1e-6)
                approx_m = 2 + 30 * np.clip(norm, 0, 1)
                if approx_m > 200:
                    approx_m = None

            # Speed smoothing
            raw_speed = speed_est.estimate_speed(tid, (cx, cy), h, cls_name)
            if raw_speed is not None:
                prev = speed_est.speed_ema.get(tid, raw_speed)
                speed = speed_est.ema_alpha * prev + (1 - speed_est.ema_alpha) * raw_speed
                speed_est.speed_ema[tid] = speed
            else:
                speed = None

            active_now.add(tid)
            retained[tid] = {
                "bbox": (x, y, w, h),
                "cls": cls_name,
                "age": 0,
                "dist": approx_m,
                "speed": speed,
                "track_obj": tr
            }

        # Age retained entries
        # -------------------------------------------------------
        # Age retained entries (robust TTL + tracker staleness)
        # -------------------------------------------------------
        to_del = []
        RETAIN_TTL = int(cfg.get("tracker", {}).get("retain_ttl_frames", 6))  # default: 6 frames

        for tid, info in list(retained.items()):
            if tid in active_now:
                # reset metadata when track is active
                info["age"] = 0
                info["last_seen"] = frame_i
                info["time_since_update"] = getattr(info.get("track_obj", None), "time_since_update", 0)
                continue

            # fallback values
            last_seen = info.get("last_seen", frame_i)
            frames_missing = frame_i - last_seen

            # tracker staleness if available
            tsu = info.get("time_since_update", getattr(info.get("track_obj", None), "time_since_update", None))
            if tsu is None:
                info["age"] = info.get("age", 0) + 1
            else:
                info["age"] = int(tsu)

            # delete if gone too long
            if frames_missing > RETAIN_TTL or info["age"] > RETAIN_TTL:
                to_del.append(tid)

        for tid in to_del:
            try:
                del retained[tid]
            except KeyError:
                pass

        # -------------------------------------------------------
        # Run STM (heuristic) + draw predicted trajectories (sparse)
        # -------------------------------------------------------
        stm_results = {}
        if frame_i % 2 == 0:
            stm_input = []
            for tid, info in retained.items():
                if info["cls"] not in ["car","truck","bus","motorbike","bicycle"]:
                    continue
                if info["dist"] is None or info["speed"] is None:
                    continue
                if info["speed"] < 2.0 or info["dist"] > 80.0:
                    continue
                tr_obj = info.get("track_obj", None)
                if tr_obj is None:
                    continue
                stm_input.append((tr_obj, info["cls"], info["speed"], info["dist"]))

            if stm_input:
                try:
                    stm_results = stm.update(stm_input, depth_map)
                except Exception:
                    stm_results = {}
            # stm.draw_predictions is optional — we'll draw our own polished traces below

        # -------------------------------------------------------
        # Build veh_list and pick lead
        # -------------------------------------------------------
        veh_list = []
        for tid, info in retained.items():
            # Skip stale or long-missing entries
            last_seen = info.get("last_seen", 0)
            if frame_i - last_seen > 6:  # matches RETAIN_TTL default
                continue

            x, y, w, h = info["bbox"]
            cls_name = info["cls"]

            # Draw potholes separately
            if cls_name.lower() == "pothole":
                cv2.rectangle(vis, (x, y), (x + w, y + h), (128, 0, 255), 3)
                draw_label(vis, x + 4, y + 18, "POTHOLE", color=(128, 0, 255), font_scale=0.6)
                play_beep(700, 120)
                continue

            if cls_name in ["car","truck","bus","motorbike","bicycle"]:
                d = info["dist"]
                s = info["speed"]
                cx = x + w // 2
                veh_list.append((d if d else 999, tid, x, y, w, h, cls_name, s, cx))

        # -------------------------------------------------------
        # FIX 1 — FILTER VEHICLES TO REDUCE CLUTTER
        # -------------------------------------------------------
        # Remove small, far, and irrelevant boxes
        filtered = []
        for v in veh_list:
            dist, tid, x, y, w, h, cls_name, speed, cx = v

            if dist is None:
                continue
            if dist > 80:          # ignore objects far away
                continue
            if w < 40 or h < 40:   # ignore tiny boxes
                continue

            filtered.append(v)

        veh_list = filtered


        lead = None
        if veh_list:
            in_band = [v for v in veh_list if cx_band_min < v[8] < cx_band_max and v[0] != 999]
            if in_band:
                lead = min(in_band, key=lambda v: v[0])
            else:
                valid = [v for v in veh_list if v[0] != 999]
                if valid:
                    lead = min(valid, key=lambda v: v[0])

        # -------------------------------------------------------
        # Draw HUD: render near -> far (draw far first so near overlays)
        # We'll sort by distance descending so nearer vehicles draw last
        # -------------------------------------------------------
        veh_list_sorted = sorted(veh_list, key=lambda v: (v[0] if v[0] is not None else 999), reverse=True)

        # For ML panel calculations
        ml_risks = []

        # Draw each vehicle
        for dist, tid, x, y, w, h, cls_name, speed, cx in veh_list_sorted:
            # Skip extremely far or static small objects to reduce clutter
            if dist is None:
                continue
            # Special highlighting for potholes
            if cls_name.lower() == "pothole":
                cv2.rectangle(vis, (x, y), (x + w, y + h), (128, 0, 255), 3)  # purple box
                draw_label(vis, x + 4, y + 18, "POTHOLE", color=(128, 0, 255), font_scale=0.6)
                play_beep(700, 120)
                continue  # skip distance/TTC logic for potholes
            if dist > 120:
                continue

            is_lead = (lead and tid == lead[1])

            # Determine TTC if available
            ttc = None
            if speed and dist:
                v_ms = speed / 3.6
                raw = dist / max(v_ms, 1e-6)
                if 0 < raw < 15:
                    ttc = raw

            # -------------------------------------------------------
            # FIX 2 — HARD COLLISION BOOST FOR VERY CLOSE OBJECTS
            # -------------------------------------------------------
            if dist is not None and dist < 8.0:
                # Extremely close object → treat as collision hazard
                ml_risk_ema = 1.0
                ml_risk_peak = 1.0

                # Red warning on HUD box
                cv2.putText(
                    vis,
                    "CRITICAL!",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                )


            # -------------------------------------------------------
            # FIX 3 — CRASH DETECTION VIA BOUNDING BOX GROWTH RATE
            # -------------------------------------------------------
            area = w * h
            prev_area = getattr(retained[tid].get("track_obj"), "prev_box_area", None)

            if prev_area is not None:
                growth = (area - prev_area) / max(prev_area, 1)

                # If object suddenly expands >35% in one frame → collision or very near collision
                if growth > 0.60:
                    ml_risk_ema = 1.0
                    ml_risk_peak = 1.0

                    cv2.putText(
                        vis,
                        "CRASH!",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

            # Save current area for next frame comparison
            setattr(retained[tid]["track_obj"], "prev_box_area", area)

            
                        # -------------------------------------------------------
            # FIX 4 — CRASH TRAJECTORY / SIDE ESTIMATION (centroid velocity)
            # -------------------------------------------------------
            try:
                # compute centroid for this bbox
                cur_cx = int(x + w // 2)
                cur_cy = int(y + h // 2)

                # previous centroid stored on the track object (in pixels)
                prev_cx = getattr(retained[tid]["track_obj"], "prev_cx", None)
                prev_cy = getattr(retained[tid]["track_obj"], "prev_cy", None)

                # pixel motion since last frame
                dx = None if prev_cx is None else (cur_cx - prev_cx)
                dy = None if prev_cy is None else (cur_cy - prev_cy)

                # normalize motion relative to box width/height to be scale-invariant
                dx_norm = None
                if dx is not None and w > 0:
                    dx_norm = dx / float(w)

                # thresholds (tunable)
                PIXEL_MOVE_THRESH = max(8, int(0.08 * W))    # absolute px threshold or 8px min
                NORM_MOVE_THRESH = 0.12                      # normalized relative to box width

                # Determine lateral approach towards center
                collision_side = None
                approaching_center = False
                if dx is not None:
                    # object moving towards image center?
                    img_center_x = W // 2
                    # if centroid is left of center and moving right (dx>0), it's approaching center from left
                    if (cur_cx < img_center_x and dx > 0) or (cur_cx > img_center_x and dx < 0):
                        approaching_center = True

                    # check magnitude
                    if abs(dx) > PIXEL_MOVE_THRESH or (dx_norm is not None and abs(dx_norm) > NORM_MOVE_THRESH):
                        # decide side relative to ego center
                        if cur_cx < img_center_x:
                            collision_side = "LEFT->CENTER"
                        else:
                            collision_side = "RIGHT->CENTER"

                # If we previously flagged a CRASH via growth or close distance, annotate direction
                if (collision_side is not None) and (ml_risk_ema >= 0.9):
                    # overlay direction label
                    cv2.putText(vis, f"CRASH {collision_side}", (x, max(20, y - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                    # optional: nudge ml_risk_peak to ensure panel shows final severity
                    ml_risk_peak = max(ml_risk_peak, ml_risk_ema)

                    # store for telemetry (attach to retained entry)
                    retained[tid]["crash_side"] = collision_side

                # Save centroids for next-frame velocity
                retained[tid]["track_obj"].prev_cx = cur_cx
                retained[tid]["track_obj"].prev_cy = cur_cy

            except Exception:
                # keep robust in case track_obj is missing; do not crash the main loop
                pass











            # Box color by TTC level
            if ttc is None or ttc > 4.0:
                box_color = (34, 177, 76)   # green
            elif 2.0 < ttc <= 4.0:
                box_color = (0, 215, 255)   # yellow
            else:
                box_color = (0, 64, 255)    # red-ish (BGR)

            # Highlight animals or pedestrians more visibly
            if cls_name.lower() in ["deer", "animal", "dog", "cow", "horse", "person"]:
                if ttc and ttc < 3.0:
                    box_color = (0, 0, 255)  # vivid red
                elif ttc and ttc < 6.0:
                    box_color = (0, 165, 255)  # orange


            # Fade with distance slightly (closer brighter)
            fade = int(np.clip(255 * (1 - (dist / 80.0)), 60, 255))
            box_color = (int(box_color[0]*fade/255), int(box_color[1]*fade/255), int(box_color[2]*fade/255))

            thickness = 3 if is_lead else 1
            cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, thickness, cv2.LINE_AA)

            # Multi-line info inside/near box (class / dist | speed / TTC)
            # Top line: class
            draw_label(vis, x + 4, y + 18, f"{cls_name}", color=box_color, font_scale=0.55)

            # Middle: distance | speed
            sp_text = "--"
            if speed is not None:
                sp_text = f"{int(speed)} km/h"
            dist_text = f"{dist:.1f} m" if dist is not None else "-- m"
            draw_label(vis, x + 4, y + 40, f"{dist_text} | {sp_text}", color=(255,255,255), font_scale=0.52)

            # Bottom small: TTC if relevant
            if ttc is not None and ttc < 6.0:
                draw_label(vis, x + 4, y + h - 6, f"TTC: {ttc:.1f}s", color=(0,0,255), font_scale=0.5)

            # STM dotted trajectory (if available in stm_results)
            # stm_results keyed by tid -> {pred_path: [(x,y),...], risk_score: ...}
            if stm_results and (tid in stm_results):
                entry = stm_results[tid]
                pred_path = entry.get("pred_path", None)
                risk_score = entry.get("risk_score", 0.0)
                if pred_path:
                    # Scale predicted points are already in pixel coords of original track object in stm.update,
                    # but if they were in scaled coords, you may need to adjust. We'll assume they're in image coords.
                    # Compute color gradient based on risk_score
                    r = int(255 * min(1.0, risk_score))
                    g = int(200 * (1 - min(1.0, risk_score)))
                    col = (0, g, r)
                    # Draw dotted polyline with fading alpha (future points more transparent)
                    # We'll draw multiple short segments with decreasing alpha
                    n = len(pred_path)
                    if n >= 2:
                        # Prepare alpha per segment decreasing
                        for i in range(n-1):
                            alpha = 0.9 * (1 - (i / max(1, n-1)) * 0.75)  # start ~0.9 -> ~0.15
                            seg_pts = [pred_path[i], pred_path[i+1]]
                            draw_dotted_polyline(vis, seg_pts, color=col, thickness=2, gap=6, dot_len=4, alpha=alpha)

            # LSTM collision prediction — record for panel
            if "stm_lstm" in locals():
                prev = stm_lstm.buffers.get(tid, [])
                accel = 0.0
                if len(prev) >= 2:
                    try:
                        accel = (speed or 0.0) - float(prev[-1][3])
                    except Exception:
                        accel = 0.0
                state_vec = [x + w / 2, y + h / 2, dist or 0.0, speed or 0.0, accel]
                try:
                    p_coll = stm_lstm.update(tid, state_vec)
                except Exception:
                    p_coll = None
                if p_coll is not None:
                    ml_risks.append((p_coll, tid))
                    # optionally mark high risk bounding box (overlay)
                    if p_coll > 0.75:
                        cv2.putText(vis, f"!!! {int(p_coll*100)}%", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

            # Add to reasoner inputs (for global TTC)
            # (use the actual track object where possible)
            tr_obj = retained.get(tid, {}).get("track_obj") if retained.get(tid) else None
            if tr_obj is None:
                # create a small FakeTrack so reasoner.update can unpack
                class FakeTrack:
                    def __init__(self, tid, x, y, w, h):
                        self.track_id = tid
                        self._bb = np.array([x, y, w, h])
                    def to_tlwh(self): return self._bb
                reasoner_inputs.append((FakeTrack(tid, x, y, w, h), cls_name, speed, dist))
            else:
                reasoner_inputs.append((tr_obj, cls_name, speed, dist))
             
            telemetry.log(
                    frame_i=frame_i,
                    track=retained[tid]["track_obj"],
                    cls=cls_name,
                    dist=dist,
                    speed=speed,
                    ttc=ttc,
                    ml_risk=p_coll if 'p_coll' in locals() else None,
                    stm_risk=stm_results.get(tid, {}).get("risk_score", None) if stm_results else None,
                    global_risk=ml_risk_ema)

      
        # Adaptive ML risk blending — YOLO/STM + OwlViT prompt risk
        # -------------------------------------------------------
        # Ensure prompt_risk_signal always defined and clipped
        prompt_risk_signal = float(np.clip(globals().get("prompt_risk_signal", 0.0), 0.0, 1.0))

        if ml_risks:
            risks = [r for r, _ in ml_risks]
            peak_yolo = max(risks)
            avg_yolo = float(np.mean(risks))

            # Weighted combination: YOLO (vehicle) dominates, prompt adds context
            # --- More responsive risk fusion (reacts instantly on high signals) ---
            # Use the strongest short-term signal (YOLO peak or prompt) rather than a slow weighted average
            combined = float(np.clip(max(avg_yolo, peak_yolo, 0.6 * prompt_risk_signal), 0.0, 1.0))

            # Faster rise, controlled decay
            if combined > ml_risk_ema:
                ml_risk_ema = 0.70 * ml_risk_ema + 0.35 * combined  # fast rise
            else:
                ml_risk_ema = 0.90 * ml_risk_ema + 0.10 * combined  # slow decay

            ml_risk_ema = float(np.clip(ml_risk_ema, 0.0, 1.0))
            ml_risk_peak = max(ml_risk_peak, ml_risk_ema, peak_yolo, prompt_risk_signal)


        else:
            # No YOLO risk — rely purely on prompt detections
            if prompt_risk_signal > ml_risk_ema:
                ml_risk_ema = 0.5 * ml_risk_ema + 0.5 * prompt_risk_signal
            else:
                ml_risk_ema = 0.93 * ml_risk_ema  # decay slightly faster in prompt-only mode
            ml_risk_ema = float(np.clip(ml_risk_ema, 0.0, 1.0))
            ml_risk_peak = max(ml_risk_peak, ml_risk_ema, prompt_risk_signal)


        # Gradual fade of OwlViT contribution
        globals()["prompt_risk_signal"] = prompt_risk_signal * 0.92


        # Build bottom-right panel
        panel_w = 300
        panel_h = 90
        pad = 12
        bx1 = W - panel_w - pad
        by1 = H - panel_h - pad
        bx2 = W - pad
        by2 = H - pad

        overlay = vis.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        # Title
        title = "ML COLLISION PANEL"
        cv2.putText(vis, title, (bx1 + 12, by1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2, cv2.LINE_AA)

        # Risk bar background
        bar_x1 = bx1 + 12
        bar_x2 = bx1 + panel_w - 24
        bar_y = by1 + 40
        cv2.rectangle(vis, (bar_x1, bar_y), (bar_x2, bar_y + 12), (80,80,80), -1)
        # filled portion (based on ml_risk_ema)
        fill_w = int((bar_x2 - bar_x1) * np.clip(ml_risk_ema, 0.0, 1.0))
        if fill_w > 0:
            # gradient color from green->yellow->red
            r = int(255 * np.clip(ml_risk_ema, 0.0, 1.0))
            g = int(255 * np.clip(1.0 - ml_risk_ema, 0.0, 1.0))
            cv2.rectangle(vis, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + 12), (0, g, r), -1)

        # Risk text
        cv2.putText(vis, f"Current Risk: {ml_risk_ema*100:.1f}%", (bx1 + 12, by1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)
        cv2.putText(vis, f"Peak: {ml_risk_peak*100:.1f}%  Tracked: {len(veh_list)}", (bx1 + 12, by1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # -------------------------------------------------------
        # Ego speed estimate (median of moving objects) bottom-left
        # -------------------------------------------------------
        speeds = [info["speed"] for info in retained.values() if info["speed"] is not None and info["speed"] > 2.0]
        ego_speed = float(np.median(speeds)) if speeds else 0.0
        ego_text = f"Ego speed (est): {int(ego_speed)} km/h    FPS: {cap.get(cv2.CAP_PROP_FPS):.0f}"
        draw_label(vis, 12, H - 10, ego_text, color=(50,50,50), font_scale=0.55)

        # -------------------------------------------------------
        # Global collision banner if TTC risk or ML risk high
        # -------------------------------------------------------
        _, risk_ttc = reasoner.update(W, H, reasoner_inputs)
        # If either physics TTC critical or ML peak high -> show banner
        banner_show = False
        if (risk_ttc is not None and risk_ttc < 1.0) or (ml_risk_ema > 0.85) or (ml_risk_peak > 0.92):
            banner_show = True

        if banner_show:
            if risk_ttc is not None and risk_ttc < 1.5:
                banner_text = "DANGER  OBJECT AHEAD  (LOW TTC)"
            elif ml_risk_ema > 0.7:
                banner_text = "AI COLLISION RISK — HIGH CONFIDENCE"
            else:
                banner_text = "IMMINENT COLLISION DETECTED — TAKE ACTION"

            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            bx = max(10, (W - tw) // 2)
            overlay = vis.copy()
            # red gradient-ish bar
            cv2.rectangle(overlay, (bx-14, 40-th-14), (bx + tw + 14, 40 + 14), (0,0,200), -1)
            cv2.addWeighted(overlay, 0.9, vis, 0.1, 0, vis)
            cv2.putText(vis, banner_text, (bx, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3, cv2.LINE_AA)
            # alarm
            if ml_risk_ema > 0.6 or (risk_ttc is not None and risk_ttc < 1.2):
                play_beep(950, 150)

        
        # Display result window
        
        cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
        key = cv2.waitKey(1) & 0xFF


        if key == ord('h'):
            print("\n==================== XenSense Help Menu ====================")
            print(" q  - Quit the program")
            print(" d  - Toggle dehaze enhancement")
            print(" v  - Toggle video enhancement (user prompt-based)")
            print(" o  - Toggle OwlViT prompt-based object detection")
            print()
            print(" t  - Enter new detection prompts (real-time)")
            print(" u  - Voice-activated prompt update")
            print(" l  - Load prompt profile")
            print(" s  - Save current prompt profile")
            print()
            print(" m  - Toggle Motion Heatmap (optical flow)")
            print(" n  - Toggle Depth Heatmap (depth gradient)")
            print()
            print("=============================================================\n")

            # Create dark overlay box
            help_overlay = vis.copy()
            box_w = 500
            box_h = 420
            x1, y1 = 40, 40
            x2, y2 = x1 + box_w, y1 + box_h

            cv2.rectangle(help_overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
            cv2.addWeighted(help_overlay, 0.85, vis, 0.15, 0, vis)

            # Render lines
            y = y1 + 40
            # for line in help_lines:
            #     cv2.putText(vis, line, (x1 + 20, y), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            #     y += 32

            # Freeze overlay until key press
            cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
            cv2.waitKey(0)
            continue





        







        if key == ord('o'):  # Toggle prompt-based detection
            prompt_enabled = not prompt_enabled
            status = "ON" if prompt_enabled else "OFF"
            print(f"[INFO] Prompt Detection: {status}")

            # HUD popup
            overlay = vis.copy()
            cv2.rectangle(overlay, (20, 20), (350, 70), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            cv2.putText(vis, f"Prompt Detection: {status}", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
            cv2.waitKey(350)

        if key == ord('p'):
            detector.pothole_enabled = det_cfg.get("enable_pothole", True)

            status = "ON" if detector.pothole_enabled else "OFF"
            print(f"[INFO] Pothole detection toggled {status}")
            # Optional on-screen message
            overlay = vis.copy()
            cv2.rectangle(overlay, (20, 20), (340, 70), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            cv2.putText(vis, f"Pothole Detection: {status}", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
            cv2.waitKey(400)  # show message briefly

        # Toggle video enhancement prompt
        if key == ord('v'):
            video_prompt.enabled = not video_prompt.enabled
            status = "ON" if video_prompt.enabled else "OFF"

            print(f"[INFO] Video Enhancement: {status}")

            # HUD popup
            overlay = vis.copy()
            cv2.rectangle(overlay, (20, 80), (420, 130), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            cv2.putText(vis, f"Enhancement: {status}", (30, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.83, (255,255,255), 2)
            cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
            cv2.waitKey(350)

        
        if key == ord('m'):
            motion_heatmap_enabled = not motion_heatmap_enabled
            status = "ON" if motion_heatmap_enabled else "OFF"
            print(f"[INFO] Motion Heatmap: {status}")

            overlay = vis.copy()
            cv2.rectangle(overlay, (20, 260), (380, 310), (50,50,50), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            cv2.putText(vis, f"Motion Heatmap {status}", (30, 295),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
            cv2.waitKey(350)


        if key == ord('n'):
            depth_heatmap_enabled = not depth_heatmap_enabled
            status = "ON" if depth_heatmap_enabled else "OFF"
            print(f"[INFO] Depth Heatmap: {status}")

            overlay = vis.copy()
            cv2.rectangle(overlay, (20, 320), (360, 370), (50,50,50), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            cv2.putText(vis, f"Depth Heatmap {status}", (30, 355),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
            cv2.waitKey(350)



        # Load profile (L key)
        if key == ord('l'):
            profile = input("Load profile name: ").strip()
            prompt_list = load_prompt_profile(profile)
            prompt_enabled = bool(prompt_list)
            print(f"[PROFILE] Loaded profile '{profile}': {prompt_list}")

        # Save profile (S key)
        if key == ord('s'):
            name = input("Save prompt profile as: ").strip()
            save_prompt_profile(name, prompt_list)
            print(f"[PROFILE] Saved profile '{name}'.")

        
        # Runtime OwlViT prompt update (keypress: T)
        if key == ord('t'):
            print("\n---------------------------------------")
            print(" ENTER NEW REAL-TIME PROMPT (comma-separated)")
            print(" Press ENTER to disable prompt detection")
            print("---------------------------------------")

            new_prompt = input("Prompt: ").strip()

            if not new_prompt:
                prompt_list = []
                prompt_enabled = False
                print("[INFO] Prompt detection DISABLED.")
            else:
                prompt_list = [p.strip() for p in new_prompt.split(",") if p.strip()]
                prompt_enabled = True
                print(f"[INFO] Updated prompt list: {prompt_list}")

            # HUD popup
            overlay = vis.copy()
            cv2.rectangle(overlay, (20, 150), (520, 200), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            msg = "Prompt Updated" if prompt_enabled else "Prompt Detection Off"
            cv2.putText(vis, msg, (30, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("XenSense — Pro HUD (Hybrid)", vis)
            cv2.waitKey(500)


        # Voice-activated prompt update
        if key == ord('u'):  # 'u' for voice-update
            import speech_recognition as sr

            print("[VOICE] Say your detection prompt...")
            r = sr.Recognizer()

            with sr.Microphone() as source:
                audio = r.listen(source)

            try:
                spoken = r.recognize_google(audio).lower()
                print("You said:", spoken)

                if "disable" in spoken or "off" in spoken:
                    prompt_enabled = False
                    prompt_list = []
                else:
                    # Split by spaces or commas
                    prompt_list = [p.strip() for p in spoken.replace("and", ",").split(",")]
                    prompt_enabled = True

                print(f"[VOICE] Updated prompts: {prompt_list}")

            except:
                print("[VOICE] Could not understand. Try again.")




        if key == ord('q'):
            break
        if key == ord('d'):
            dehaze_enabled = not dehaze_enabled

    cap.release()
    telemetry.close()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
