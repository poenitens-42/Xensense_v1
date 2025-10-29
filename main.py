import cv2
import yaml
import torch
from ultralytics import YOLO
from detector import Detector
from tracker.deep_sort import DeepSort
from speed_estimator import SpeedEstimator
from utils.inpainting import Inpainter
from smoke_detector import SmokeDetector
from depth_estimator import DepthEstimator
import numpy as np
import sys
import simpleaudio as sa
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

sys.path.append("lama")

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# LLM Setup (CPU-friendly)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "rC:\Users\arjun\.cache\huggingface\hub\models--microsoft--phi-4-mini-instruct"

print("[INFO] Loading LLM model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,   # or float32
        device_map="auto",
        trust_remote_code=True
    )

    print("[INFO] LLM model loaded successfully!")

except Exception as e:
    print(f"[ERROR] Failed to load LLM model: {e}")
    llm_model = None  # prevent NameError later


    llm_pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        device=-1                  # -1 = CPU, change if you want GPU
    )
    print("[INFO] LLM model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load LLM model: {e}")
    llm_pipe = None


# Config

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Utilities

def create_object_mask(frame, x, y, w, h):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return mask

def enhance_smoke_region(frame, mask):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    mask_3c = cv2.merge([mask, mask, mask]).astype(np.float32)/255
    output = frame*(1-mask_3c) + enhanced*mask_3c
    return output.astype(np.uint8)

class SpeedEstimatorOptimized(SpeedEstimator):
    def __init__(self, fps=30, pixel_to_meter=0.05, mode="hybrid"):
        super().__init__(fps=fps, pixel_to_meter=pixel_to_meter, mode=mode)
        if not hasattr(self, "last_states"):
            from collections import defaultdict
            self.last_states = defaultdict(lambda: None)

    def is_moving(self, track_id, center, threshold=2):
        last = self.last_states[track_id]
        if last is None:
            return True
        (x1, y1), _, _ = last
        (x2, y2) = center
        dist_pixels = np.hypot(x2 - x1, y2 - y1)
        return dist_pixels > threshold

def check_hazard(track, approx_distance_m, frame_w, warning_dist=10, cls_name=None):
    x, y, w, h = track.to_tlwh().astype(int)
    cx = x + w // 2
    center_region = (frame_w * 0.3, frame_w * 0.7)

    if cls_name == "pothole" and approx_distance_m < 15:
        return True, "HAZARD: Pothole Ahead"

    if cls_name in ["car", "truck", "bus", "motorbike", "bicycle", "person"]:
        if center_region[0] < cx < center_region[1] and approx_distance_m < warning_dist:
            return True, "HAZARD! Slow Down"

    return False, ""

def play_beep(frequency=10000, duration_ms=200):
    fs = 44100
    t = np.linspace(0, duration_ms/1000, int(fs * duration_ms / 1000), False)
    note = np.sin(frequency * t * 2 * np.pi)
    audio = (note * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, fs)

# ---------------------------
# LLM Reasoning
# ---------------------------
def llm_reasoning(tracked_objects_json, memory_frames=None):
    if llm_pipe is None:
        return "LLM not available"

    try:
        prompt = "You are XenSense AI. Analyze detected objects and previous frames for hazards or abnormal behavior.\n"
        prompt += f"Current frame objects: {tracked_objects_json}\n"
        if memory_frames:
            prompt += f"Previous frames: {memory_frames}\n"
        prompt += "Provide any hazards or warnings concisely."

        output = llm_pipe(prompt, max_new_tokens=128, do_sample=False)
        return output[0]['generated_text']
    except Exception as e:
        return f"LLM reasoning failed: {e}"


# Main

def main():
    cfg = load_config()
    video_path = input("Enter path to video file: ")
    dehaze_enabled = input("Enable smoke enhancement? (y/n): ").lower() == 'y'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = Detector(cfg["detector"]["model"], cfg["detector"]["conf_thresh"])
    detector.allowed_classes = cfg["detector"]["allowed_classes"]

    pothole_model = None
    if cfg.get("pothole_detector", {}).get("enabled", False):
        pothole_model = YOLO(cfg["pothole_detector"]["model"])
        pothole_conf_thresh = cfg["pothole_detector"].get("conf_thresh", 0.25)

    pose_model = YOLO(cfg["pose"]["model"]) if cfg["pose"]["enabled"] else None

    tracker = DeepSort(
        max_age=cfg["tracker"]["max_age"],
        n_init=cfg["tracker"]["n_init"],
        conf_thresh=cfg["tracker"]["conf_thresh"]
    )

    speed_estimator = SpeedEstimatorOptimized(
        fps=cfg["speed"]["fps"],
        pixel_to_meter=cfg["speed"]["pixel_to_meter"]
    )

    inpainter = Inpainter(mode=cfg.get("inpainting", {}).get("mode", "hybrid"))
    smoke_detector = SmokeDetector(model_path=cfg.get("smoke_model_path", None), use_yolo=True)

    depth_estimator = DepthEstimator(model_type="MiDaS_small", device="cpu")
    depth_resize = (256, 144)
    depth_skip = 10

    model_h, model_w = 384, 640
    frame_count = 0
    depth_map = None

    print("[INFO] Starting video processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break
        frame_count += 1
        orig_frame = frame.copy()
        resized_frame = cv2.resize(frame, (model_w, model_h))
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h

        # Detection
       
        detections_raw = detector.detect(resized_frame)
        detections = []
        for det in detections_raw:
            x1, y1, x2, y2, conf, cls_name = det
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
            detections.append([x1, y1, x2, y2, conf, cls_name])

        # Pothole detection
        if pothole_model:
            pothole_results = pothole_model(resized_frame)
            for r in pothole_results:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores):
                    if score >= pothole_conf_thresh:
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                        detections.append([x1, y1, x2, y2, score, "pothole"])

        # Smoke detection
        smoke_dets = smoke_detector.detect(resized_frame)
        smoke_mask_frame = np.zeros(orig_frame.shape[:2], dtype=np.uint8)
        for det in smoke_dets:
            sx1, sy1, sx2, sy2 = map(int, [det["x1"]*scale_x, det["y1"]*scale_y, det["x2"]*scale_x, det["y2"]*scale_y])
            smoke_mask_frame[sy1:sy2, sx1:sx2] = 255


        # Tracking

        tracks = tracker.update_tracks(detections, orig_frame)
        tracked_objects_json = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            x, y, w, h = track.to_tlwh().astype(int)
            track_id = track.track_id
            cls_name = track.cls
            cx, cy = x + w//2, y + h//2

            if not speed_estimator.is_moving(track_id, (cx, cy)):
                continue

            speed = speed_estimator.estimate_speed(track_id, (cx, cy), h, cls_name)

            # Depth approximation
            if depth_map is not None:
                cy_clamped = np.clip(cy, 0, depth_map.shape[0]-1)
                cx_clamped = np.clip(cx, 0, depth_map.shape[1]-1)
                dist_norm = depth_map[cy_clamped, cx_clamped]
                approx_distance_m = (1 - dist_norm) * cfg.get("max_depth_m", 50)
            else:
                approx_distance_m = 0.0

            # Hazard check
            hazard, warning_msg = check_hazard(track, approx_distance_m, orig_w, warning_dist=10, cls_name=cls_name)
            if hazard:
                cv2.putText(orig_frame, warning_msg, (x, y-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0,0,255), 3)
                play_beep(frequency=1000 if cls_name != "pothole" else 600)
            else:
                color = (0,255,0) if cls_name != "pothole" else (0,0,255)
                label = f"ID:{track_id} {cls_name} {speed:.1f} km/h {approx_distance_m:.1f} m" \
                        if cls_name != "person" else f"{cls_name} {approx_distance_m:.1f} m"
                cv2.rectangle(orig_frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(orig_frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            smoke_flag = np.any(smoke_mask_frame[y:y+h, x:x+w])

            tracked_objects_json.append({
                "id": track_id,
                "class": cls_name,
                "bbox": [x, y, x+w, y+h],
                "speed": speed,
                "smoke": smoke_flag
            })

        # 
        # Frame memory
        #
        # frame_memory.append(tracked_objects_json)
        # if len(frame_memory) > FRAME_MEMORY_SIZE:
        #     frame_memory.pop(0)

        # #
        # # LLM Reasoning
        # #
        # reasoning_output = llm_reasoning(tracked_objects_json, frame_memory[:-1])
        # print(f"[Frame {frame_count}] LLM: {reasoning_output}")

        #
        # Smoke enhancement toggle
        #
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            dehaze_enabled = not dehaze_enabled
        if dehaze_enabled and np.any(smoke_mask_frame):
            orig_frame = enhance_smoke_region(orig_frame, smoke_mask_frame)

        #
        # Pose skeleton
        # 
        if pose_model:
            pose_results = pose_model(orig_frame)
            for r in pose_results:
                if r.keypoints is not None:
                    kpts = r.keypoints.xy.cpu().numpy()
                    for person in kpts:
                        for (px, py) in person:
                            cv2.circle(orig_frame, (int(px), int(py)), 2, (0,0,255), -1)
                        for i,j in SKELETON_CONNECTIONS:
                            if i < len(person) and j < len(person):
                                x1, y1 = map(int, person[i])
                                x2, y2 = map(int, person[j])
                                cv2.line(orig_frame, (x1,y1), (x2,y2), (255,0,0), 2)

        # 
        # Show frame
        #
        cv2.imshow("XenSensev2", orig_frame)
        if key == ord("q"):
            break

        # ---------------------------
        # Depth map update
        # ---------------------------
        if frame_count % depth_skip == 0:
            small_frame = cv2.resize(orig_frame, depth_resize)
            depth_map_small = depth_estimator.estimate_depth(small_frame)
            depth_map = cv2.resize(depth_map_small, (orig_w, orig_h))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import traceback
    print("[INFO] Script started successfully!")
    try:
        main()
    except Exception as e:
        print("[ERROR] Exception in main:")
        traceback.print_exc()
