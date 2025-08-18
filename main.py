import cv2
import yaml
from ultralytics import YOLO
from detector import Detector
from tracker.deep_sort import DeepSort
from speed_estimator import SpeedEstimator
from utils.inpainting import Inpainter   # NEW
import numpy as np
import sys

sys.path.append("lama")



# Skeleton definition (Human pose detection)
SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),     # Left arm
    (6, 8), (8, 10),    # Right arm
    (5, 6),             # Shoulders
    (11, 12),           # Hips
    (5, 11), (6, 12),   # Torso
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16)  # Right leg
]

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_object_mask(frame, x, y, w, h):
    """Creates a binary mask for the detected object."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return mask

def main():
    cfg = load_config()

    video_path = input("Enter path to video file: ")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Detector
    detector = Detector(cfg["detector"]["model"], cfg["detector"]["conf_thresh"])
    detector.allowed_classes = cfg["detector"]["allowed_classes"]

    # Pose model
    pose_model = None
    if cfg["pose"]["enabled"]:
        pose_model = YOLO(cfg["pose"]["model"])

    # Tracker + Speed Estimator
    tracker = DeepSort(
        max_age=cfg["tracker"]["max_age"],
        n_init=cfg["tracker"]["n_init"],
        conf_thresh=cfg["tracker"]["conf_thresh"]
    )
    speed_estimator = SpeedEstimator(
        fps=cfg["speed"]["fps"],
        pixel_to_meter=cfg["speed"]["pixel_to_meter"]
    )

    # NEW: Inpainter instance
    inpainter = Inpainter(mode=cfg.get("inpainting", {}).get("mode", "hybrid"))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run detector
        detections = detector.detect(frame)

        # Update tracker
        tracks = tracker.update_tracks(detections, frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            x, y, w, h = track.to_tlwh().astype(int)
            track_id = track.track_id
            cls_name = track.cls

            # Speed
            cx, cy = x + w // 2, y + h // 2
            speed = speed_estimator.estimate_speed(track_id, (cx, cy), h, cls_name)

            # Draw only for non-humans (car, truck, etc.)
            if cls_name != "person":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id} {cls_name} {speed:.1f} km/h",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # If user presses 'r' → remove object
            if cv2.waitKey(1) & 0xFF == ord('r'):
                mask = create_object_mask(frame, x, y, w, h)
                job_id = f"{track_id}_{frame_count}"
                frame = inpainter.inpaint(frame, mask, job_id=job_id)

                # HQ LaMa result retrieval (optional)
                hq_frame = inpainter.get_lama_result(job_id)
                if hq_frame is not None:
                    frame = hq_frame

        # Pose skeletons for humans
        if pose_model:
            pose_results = pose_model(frame)
            for r in pose_results:
                if r.keypoints is not None:
                    kpts = r.keypoints.xy.cpu().numpy()
                    for person in kpts:
                        for (px, py) in person:
                            cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)
                        for i, j in SKELETON_CONNECTIONS:
                            if i < len(person) and j < len(person):
                                x1, y1 = map(int, person[i])
                                x2, y2 = map(int, person[j])
                                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Show frame
        cv2.imshow("XenSensev1", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
