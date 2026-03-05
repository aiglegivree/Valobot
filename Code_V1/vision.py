# vision.py
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from config import W, H, ROI_TOP_FRAC, ROI_BOT_FRAC, GAUSS_K, MIN_CONTOUR_AREA

@dataclass
class LineMeasurement:
    found: bool
    e_px: float
    confidence: float
    debug: dict

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def measure_line(frame_bgr: np.ndarray) -> LineMeasurement:
    frame = cv2.resize(frame_bgr, (W, H))

    y0 = int(H * ROI_TOP_FRAC)
    y1 = int(H * ROI_BOT_FRAC)
    roi = frame[y0:y1, :]

    # 1) grayscale + blur (comme ton ancien code)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    k = GAUSS_K if GAUSS_K % 2 == 1 else GAUSS_K + 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 2) Otsu: seuil auto
    # On calcule deux masques (normal et inversé), puis on choisit le plus plausible.
    _, bw_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw_inv  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Heuristique: la ligne doit être une "petite" portion blanche, pas 90% de la ROI
    ratio_norm = float(np.mean(bw_norm > 0))
    ratio_inv  = float(np.mean(bw_inv > 0))

    # Choisit le masque dont le ratio de pixels blancs est le plus raisonnable
    # (typiquement entre 1% et 40% selon épaisseur/ROI)
    def score(r: float) -> float:
        if r < 0.005 or r > 0.70:
            return 0.0
        # bonus autour de 0.05..0.25
        return 1.0 - abs(r - 0.15)

    bw = bw_norm if score(ratio_norm) >= score(ratio_inv) else bw_inv
    white_ratio = float(np.mean(bw > 0))

    # 3) Nettoyage léger (évite de casser la ligne)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # 4) Contours -> plus grand contour (comme beaucoup de codes simples de line follow)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return LineMeasurement(False, 0.0, 0.0, {"y0": y0, "bw": bw, "white_ratio": white_ratio, "reason": "no_contours"})

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < MIN_CONTOUR_AREA:
        return LineMeasurement(False, 0.0, 0.0, {"y0": y0, "bw": bw, "white_ratio": white_ratio, "area": area, "reason": "area_too_small"})

    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-6:
        return LineMeasurement(False, 0.0, 0.0, {"y0": y0, "bw": bw, "white_ratio": white_ratio, "area": area, "reason": "bad_moments"})

    cx = float(M["m10"] / M["m00"])
    e_px = cx - (W / 2.0)

    # Confiance simple: basée sur aire + ratio blanc raisonnable
    roi_area = float(W * (y1 - y0))
    conf_area = _clamp01(area / (0.25 * roi_area))  # 25% de la ROI = confiance max
    conf_ratio = _clamp01(1.0 - abs(white_ratio - 0.15) / 0.20)
    conf = _clamp01(0.7 * conf_area + 0.3 * conf_ratio)

    return LineMeasurement(True, e_px, conf, {
        "y0": y0,
        "bw": bw,
        "white_ratio": white_ratio,
        "area": area,
        "cx": cx
    })