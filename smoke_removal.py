import cv2
import numpy as np

def enhance_smoke_region(frame, mask):
    
    # Convert to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Blend with original only on smoke
    mask_3c = cv2.merge([mask, mask, mask])
    mask_3c = mask_3c.astype(np.float32)/255
    output = frame*(1-mask_3c) + enhanced*mask_3c
    return output.astype(np.uint8)
