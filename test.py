import cv2
import torch
from depth_estimator import DepthEstimator

model = DepthEstimator(model_type="MiDaS_small", device="cpu")

img = cv2.imread("data/tt.jpg")  # any frame from your video
img_small = cv2.resize(img, (384,384))

depth = model.estimate_depth(img_small)

print("DEPTH SHAPE:", depth.shape)
print("DEPTH MIN:", depth.min())
print("DEPTH MAX:", depth.max())
print("DEPTH SAMPLE:", depth[100,100])
