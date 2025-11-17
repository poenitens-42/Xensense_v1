import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class DepthEstimator:
    """
    Optimized MiDaS depth estimator:
      - CUDA if available
      - FP16 autocast for large speed boost
      - Works with MiDaS_small, DPT_Large, DPT_Hybrid, etc.
    """
    def __init__(self, model_type="MiDaS_small", device=None):
        # Auto device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # -------------------------
        # Load MiDaS model
        # -------------------------
        # Uses torch.hub cached copy if available.
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # -------------------------
        # MiDaS transforms
        # -------------------------
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if "DPT" in model_type:  
            self.transform = midas_transforms.dpt_transform     # for larger models
        else:
            self.transform = midas_transforms.small_transform   # for MiDaS_small

        # -------------------------
        # Enable FP16 on CUDA (huge speed boost)
        # -------------------------
        self.use_fp16 = (self.device == "cuda")

    def estimate_depth(self, frame):
        """
        Estimate normalized depth map in range 0-1.
        Returns None if something goes wrong.
        """
        try:
            # Convert BGR → RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform → Tensor
            input_tensor = self.transform(img).to(self.device)

            # Ensure correct shape (NCHW)
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)

            # -------------------------
            # Inference (FP16 if CUDA)
            # -------------------------
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        prediction = self.model(input_tensor)
                else:
                    prediction = self.model(input_tensor)

                # Resize to original frame size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.float().cpu().numpy()

            # -------------------------
            # Normalize depth (0–1)
            # -------------------------
            dmin, dmax = depth.min(), depth.max()
            if (dmax - dmin) < 1e-6:
                return None

            depth_norm = (depth - dmin) / (dmax - dmin)
            return depth_norm

        except Exception:
            return None
