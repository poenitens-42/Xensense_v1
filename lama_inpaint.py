import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import cv2

from lama.saicinpainting.evaluation.data import pad_img_to_modulo
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint


class LaMaInpainter:
    def __init__(self, ckpt_path="lama/big-lama/models/best.ckpt", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model config
        ckpt_path = Path(ckpt_path)
        config_path = ckpt_path.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = OmegaConf.load(config_path)
        self.model = load_checkpoint(ckpt_path, map_location=self.device, strict=False, cfg=config)
        self.model.eval()

    def inpaint(self, image, mask):
        # Ensure BGR to RGB and normalize
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).float()

        batch = {'image': img_tensor, 'mask': mask_tensor}
        batch = move_to_device(batch, self.device)
        batch = pad_img_to_modulo(batch, mod=8)

        with torch.no_grad():
            result = self.model(batch)['inpainted']

        # Back to NumPy + BGR
        result = result[0].permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result
