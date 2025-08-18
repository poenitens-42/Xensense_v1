import cv2
import torch
import threading
import queue
import os

class Inpainter:
    def __init__(self, mode="hybrid"):
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # used for Lama inpainitng specifically for GPU acceleration 
        self.lama_model = None
        self.job_queue = queue.Queue()
        self.results = {}
        
        if self.mode in ["hybrid", "quality"]:
            try:
                from lama import LaMaInpainter
                self.lama_model = LaMaInpainter(device=self.device)
            except ImportError:
                print("[WARNING] LaMa not installed. Falling back to Telea.")
                self.mode = "fast"

        if self.mode == "hybrid":
            threading.Thread(target=self._lama_worker, daemon=True).start()

    def _lama_worker(self):
        while True:
            job_id, frame, mask = self.job_queue.get()
            if job_id is None:
                break
            result = self.lama_model.inpaint(frame, mask)
            self.results[job_id] = result

    def inpaint(self, frame, mask, job_id=None):
        if self.mode == "fast":
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        if self.mode == "quality":
            return self.lama_model.inpaint(frame, mask)

        if self.mode == "hybrid":
            # Live view: telea
            telea_result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            # Background HQ job
            if job_id:
                self.job_queue.put((job_id, frame.copy(), mask.copy()))
            return telea_result

    def get_lama_result(self, job_id):
        return self.results.pop(job_id, None)
