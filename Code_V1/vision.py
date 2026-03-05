# vision.py
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from config import (
    W, H,
    ROI_START_FRAC, LOOKAHEAD_Y_FRAC, BAND_THICKNESS,
    MIN_AREA, INVERT_AFTER_OTSU,
    KERNEL_SIZE, MORPH_CLOSE_IT, MORPH_OPEN_IT
)

@dataclass
class LinePick:
    cx_bottom: int
    cx_look: int
    area: int
    y_look: int          # y in ROI coordinates
    roi_start: int       # y offset in full image
    mask: np.ndarray     # binary mask of ROI

class LineSelector:
    def __init__(self, w: int):
        self.w = w
        self.cx_ref = w // 2

    def pick(self, mask: np.ndarray) -> dict | None:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return None

        h, w = mask.shape
        y_la = int(h * LOOKAHEAD_Y_FRAC)
        y0 = max(0, y_la - BAND_THICKNESS)
        y1 = min(h, y_la + BAND_THICKNESS)

        best = None  # (score, comp_id, cx_bottom, cx_look, area)

        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_AREA:
                continue

            comp = (labels == i).astype(np.uint8)

            # bottom slice centroid
            yb0 = int(h * 0.80)
            bottom = comp[yb0:h, :]
            if bottom.sum() == 0:
                continue
            _, xs_b = np.where(bottom > 0)
            cx_bottom = int(xs_b.mean())

            # lookahead band centroid
            band = comp[y0:y1, :]
            if band.sum() == 0:
                cx_look = cx_bottom
                miss_penalty = 1.0
            else:
                _, xs_l = np.where(band > 0)
                cx_look = int(xs_l.mean())
                miss_penalty = 0.0

            score = (
                1.5 * abs(cx_look - self.cx_ref) +
                0.7 * abs(cx_bottom - self.cx_ref) +
                500.0 * miss_penalty -
                0.002 * area
            )

            if (best is None) or (score < best[0]):
                best = (score, i, cx_bottom, cx_look, area)

        if best is None:
            return None

        _, comp_id, cx_bottom, cx_look, area = best

        # continuity (EMA)
        alpha = 0.35
        self.cx_ref = int((1 - alpha) * self.cx_ref + alpha * cx_look)

        return {
            "comp_id": comp_id,
            "cx_bottom": cx_bottom,
            "cx_look": cx_look,
            "area": area,
            "y_look": int(mask.shape[0] * LOOKAHEAD_Y_FRAC),
        }

def compute_mask_from_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Returns (mask, roi_start_y) where mask is binary uint8 (0/255) in ROI coordinates.
    """
    roi_start = int(H * ROI_START_FRAC)
    roi = frame_bgr[roi_start:H, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if INVERT_AFTER_OTSU:
        mask = 255 - mask

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_IT)

    return mask, roi_start

def detect_line(frame_bgr: np.ndarray, selector: LineSelector) -> LinePick | None:
    mask, roi_start = compute_mask_from_frame(frame_bgr)
    pick = selector.pick(mask)
    if pick is None:
        return None

    return LinePick(
        cx_bottom=int(pick["cx_bottom"]),
        cx_look=int(pick["cx_look"]),
        area=int(pick["area"]),
        y_look=int(pick["y_look"]),
        roi_start=int(roi_start),
        mask=mask
    )