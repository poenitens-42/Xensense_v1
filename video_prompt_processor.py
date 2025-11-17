import cv2
import numpy as np

class VideoPromptProcessor:
    def __init__(self):
        self.enabled = False
        self.current_prompt = None

    def set_prompt(self, text):
        self.current_prompt = text.lower()

    def process(self, frame):
        if not self.enabled or not self.current_prompt:
            return frame

        t = self.current_prompt

        if "fog" in t or "haze" in t or "dehaze" in t:
            return self.dehaze(frame)

        if "clear" in t or "sharp" in t:
            return self.sharpen(frame)

        if "bright" in t:
            return self.adjust_brightness(frame, 1.25)

        if "dark" in t:
            return self.adjust_brightness(frame, 0.75)

        if "contrast" in t:
            return self.adjust_contrast(frame, 1.3)

        return frame

    def sharpen(self, img):
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(img, -1, k)

    def dehaze(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L,A,B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)

    def adjust_brightness(self, img, factor):
        return cv2.convertScaleAbs(img, alpha=factor)

    def adjust_contrast(self, img, factor):
        return cv2.convertScaleAbs(img, alpha=factor)
