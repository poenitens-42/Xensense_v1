import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class DepthEstimator:
    def __init__(self, model_type="DPT_Large", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Load MiDaS model
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(self.device).eval()

        # Transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "DPT" in model_type:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.model = midas

    def estimate_depth(self, frame):
       
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        # Normalize 0-1
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_norm
