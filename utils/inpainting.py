import cv2
import numpy as np

# Optional LaMa import. We only enable it when CUDA is available and user chose CUDA_ONLY policy
try:
    from lama import LaMaInpainter  # your package/module providing LaMa inpaint
    _LAMA_AVAILABLE = True
except Exception:
    LaMaInpainter = None
    _LAMA_AVAILABLE = False


class Inpainter:
    """
    Hybrid inpainting with ROI-optimized Telea and optional LaMa.

    Modes:
      - "telea"  : always use OpenCV Telea (fast, stable).
      - "lama"   : use LaMa when available (slow on CPU; we gate to CUDA in code below).
      - "hybrid" : auto-choose. If mask area < area_thresh -> Telea ROI;
                   else, use LaMa *only if* CUDA is available and LaMa is installed,
                   otherwise fall back to Telea.

    Policy: CUDA_ONLY for LaMa (requested by user). If CUDA is not available,
            LaMa is never used.
    """

    def __init__(self, mode: str = "hybrid", area_thresh: float = 0.10):
        self.mode = (mode or "hybrid").lower()
        self.area_thresh = float(area_thresh)
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
        # Note: torch check is another option, but we avoid importing torch here.

        self._lama = None
        if self.mode in ("lama", "hybrid") and _LAMA_AVAILABLE and self.cuda_available:
            try:
                self._lama = LaMaInpainter(device="cuda")
            except Exception:
                self._lama = None

    # ----------------------------- public API -----------------------------
    def inpaint(self, frame: np.ndarray, mask: np.ndarray, job_id=None) -> np.ndarray:
        """Main entry. Returns a valid BGR frame of the same shape as input.
        Uses ROI-based Telea for speed; escalates to LaMa in hybrid if mask is large and LaMa on CUDA is ready.
        """
        if frame is None or mask is None or frame.size == 0:
            return frame

        # Ensure single-channel mask (uint8)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        nonzero = cv2.countNonZero(mask)
        if nonzero == 0:
            return frame  # nothing to inpaint

        H, W = frame.shape[:2]
        area_ratio = float(nonzero) / float(H * W)

        if self.mode == "telea":
            return self._telea_roi(frame, mask)

        if self.mode == "lama":
            if self._lama is not None:
                return self._lama_safe(frame, mask)
            # fallback if LaMa not available
            return self._telea_roi(frame, mask)

        # hybrid mode
        if area_ratio < self.area_thresh:
            return self._telea_roi(frame, mask)
        # large mask -> try LaMa on CUDA
        if self._lama is not None:
            return self._lama_safe(frame, mask)
        # fallback
        return self._telea_roi(frame, mask)

    # ----------------------------- helpers -----------------------------
    def _telea_roi(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fast ROI-based Telea inpainting. Avoids full-frame inpaint to keep FPS high.
        """
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return frame
        x, y, w, h = cv2.boundingRect(np.column_stack((xs, ys)))

        # Inflate ROI slightly to avoid seam artifacts
        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)

        roi_f = frame[y0:y1, x0:x1]
        roi_m = mask[y0:y1, x0:x1]

        # Safety: ensure non-empty
        if roi_f.size == 0 or roi_m.size == 0:
            return frame

        # Telea inpaint on ROI only
        out_roi = cv2.inpaint(roi_f, roi_m, 3, cv2.INPAINT_TELEA)
        out = frame.copy()
        out[y0:y1, x0:x1] = out_roi
        return out

    def _lama_safe(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa with basic guards. Assumes CUDA is available and LaMa was constructed on CUDA.
        If LaMa fails for any reason, falls back to Telea ROI.
        """
        try:
            return self._lama.inpaint(frame, mask)
        except Exception:
            return self._telea_roi(frame, mask)


# Backward-compat shim (if other modules import this function)

def inpaint_object(roi: np.ndarray, method: str = "telea") -> np.ndarray:
    """
    Legacy helper used by old Detector implementation. Keeps behavior simple:
    - telea : cv2.inpaint on full ROI with binary mask generated from non-zeros.
    - lama  : NOT supported here (hybrid handled in class above); falls back to telea.
    """
    if roi is None or roi.size == 0:
        return roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(roi, m.astype(np.uint8), 3, cv2.INPAINT_TELEA)
