from ultralytics import YOLO
import cv2

MODEL_PATH = "models/best.pt"
SOURCE = "pot_test.jpg"

print("Loading YOLO 11 model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

print("Running inference...")
results = model(SOURCE, show=True)
