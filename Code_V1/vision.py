# vision.py
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from config import (
    W, H, ROI_TOP_FRAC, ROI_BOT_FRAC,
    GAUSS_K, INV_BINARY, THRESH, MIN_CONTOUR_AREA
)

@dataclass
class LineMeasurement:
    found: bool
    e_px: float          # error in pixels: + -> line is to the right of image center
    confidence: float    # 0..1
    debug: dict

def _roi(frame: np.ndarray) -> tuple[np.ndarray, int]:
    y0 = int(H * ROI_TOP_FRAC)
    y1 = int(H * ROI_BOT_FRAC)
    return frame[y0:y1, :], y0

def measure_line(frame_bgr: np.ndarray) -> LineMeasurement:
    # Ensure size
    frame = cv2.resize(frame_bgr, (W, H))
    roi, y0 = _roi(frame)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    k = GAUSS_K if GAUSS_K % 2 == 1 else GAUSS_K + 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    if INV_BINARY:
        _, bw = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)

    # Clean small noise
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return LineMeasurement(False, 0.0, 0.0, {"y0": y0})

    # pick biggest contour
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < MIN_CONTOUR_AREA:
        return LineMeasurement(False, 0.0, 0.0, {"y0": y0, "area": area})

    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-6:
        return LineMeasurement(False, 0.0, 0.0, {"y0": y0, "area": area})

    cx = float(M["m10"] / M["m00"])
    # error wrt image center
    e_px = cx - (W / 2.0)

    # confidence: area normalized (simple heuristic)
    conf = float(np.clip(area / (W * (H * (ROI_BOT_FRAC - ROI_TOP_FRAC))), 0.0, 1.0))

    return LineMeasurement(True, e_px, conf, {"y0": y0, "area": area, "cx": cx, "bw": bw})