# vision.py
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from config import (
    H, W,
    ROI_START_FRAC, GAUSS_KSIZE, USE_BINARY_INV,
    KERNEL_SIZE, MORPH_CLOSE_IT, MORPH_OPEN_IT,
    BAND_THICKNESS,
    SLICE_TOP_FRAC, SLICE_MID_FRAC, SLICE_BOT_FRAC,
    W_TOP, W_MID, W_BOT
)


@dataclass
class VisionResult:
    cx_top: int | None
    cx_mid: int | None
    cx_bot: int | None

    err_top: float
    err_mid: float
    err_bot: float
    steer_error: float

    seen_top: bool
    seen_mid: bool
    seen_bot: bool
    any_seen: bool
    all_seen: bool

    roi_start: int
    y_top: int
    y_mid: int
    y_bot: int

    mask: np.ndarray


def compute_mask_from_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    roi_start = int(H * ROI_START_FRAC)
    roi = frame_bgr[roi_start:H, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    thresh_type = cv2.THRESH_BINARY_INV if USE_BINARY_INV else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_IT)

    return mask, roi_start


def get_band_center(mask: np.ndarray, y_frac: float, thickness: int) -> tuple[int | None, int]:
    h, w = mask.shape
    y = int(h * y_frac)

    y0 = max(0, y - thickness)
    y1 = min(h, y + thickness + 1)

    band = mask[y0:y1, :]
    ys, xs = np.where(band > 0)

    if xs.size == 0:
        return None, y

    cx = int(xs.mean())
    return cx, y


def detect_line(frame_bgr: np.ndarray) -> VisionResult:
    mask, roi_start = compute_mask_from_frame(frame_bgr)

    h, w = mask.shape
    cx_ref = w // 2

    cx_top, y_top = get_band_center(mask, SLICE_TOP_FRAC, BAND_THICKNESS)
    cx_mid, y_mid = get_band_center(mask, SLICE_MID_FRAC, BAND_THICKNESS)
    cx_bot, y_bot = get_band_center(mask, SLICE_BOT_FRAC, BAND_THICKNESS)

    seen_top = cx_top is not None
    seen_mid = cx_mid is not None
    seen_bot = cx_bot is not None

    any_seen = seen_top or seen_mid or seen_bot
    all_seen = seen_top and seen_mid and seen_bot

    # fallback local pour calculer une erreur même si une tranche manque
    if cx_bot is None and cx_mid is not None:
        cx_bot_use = cx_mid
    elif cx_bot is None and cx_top is not None:
        cx_bot_use = cx_top
    elif cx_bot is None:
        cx_bot_use = cx_ref
    else:
        cx_bot_use = cx_bot

    if cx_mid is None:
        cx_mid_use = cx_bot_use
    else:
        cx_mid_use = cx_mid

    if cx_top is None:
        cx_top_use = cx_mid_use
    else:
        cx_top_use = cx_top

    err_top = float(cx_top_use - cx_ref)
    err_mid = float(cx_mid_use - cx_ref)
    err_bot = float(cx_bot_use - cx_ref)

    steer_error = (
        W_TOP * err_top +
        W_MID * err_mid +
        W_BOT * err_bot
    )

    return VisionResult(
        cx_top=cx_top,
        cx_mid=cx_mid,
        cx_bot=cx_bot,
        err_top=err_top,
        err_mid=err_mid,
        err_bot=err_bot,
        steer_error=steer_error,
        seen_top=seen_top,
        seen_mid=seen_mid,
        seen_bot=seen_bot,
        any_seen=any_seen,
        all_seen=all_seen,
        roi_start=roi_start,
        y_top=y_top,
        y_mid=y_mid,
        y_bot=y_bot,
        mask=mask
    )


def draw_debug(frame_bgr: np.ndarray, res: VisionResult) -> np.ndarray:
    out = frame_bgr.copy()
    h_roi, w_roi = res.mask.shape
    roi_y0 = res.roi_start
    roi_y1 = roi_y0 + h_roi

    # rectangle ROI
    cv2.rectangle(out, (0, roi_y0), (w_roi - 1, roi_y1 - 1), (0, 255, 255), 2)

    # centre image
    cx_ref = w_roi // 2
    cv2.line(out, (cx_ref, roi_y0), (cx_ref, roi_y1 - 1), (255, 0, 0), 2)

    # lignes des tranches
    for y in [res.y_top, res.y_mid, res.y_bot]:
        yy = roi_y0 + y
        cv2.line(out, (0, yy), (w_roi - 1, yy), (180, 180, 180), 1)

    # points détectés
    if res.cx_top is not None:
        cv2.circle(out, (res.cx_top, roi_y0 + res.y_top), 6, (0, 255, 0), -1)
    if res.cx_mid is not None:
        cv2.circle(out, (res.cx_mid, roi_y0 + res.y_mid), 6, (0, 200, 255), -1)
    if res.cx_bot is not None:
        cv2.circle(out, (res.cx_bot, roi_y0 + res.y_bot), 6, (0, 0, 255), -1)

    return out

def draw_debug_slices(frame_bgr, mask, roi_start, y_top, y_mid, y_bot,
                      cx_top, cx_mid, cx_bot, band_thickness):
    """
    Affiche sur l'image :
    - la ROI
    - les 3 tranches
    - les centres détectés pour chaque tranche
    - le centre de l'image
    """
    out = frame_bgr.copy()
    h_roi, w_roi = mask.shape

    # rectangle ROI
    cv2.rectangle(out, (0, roi_start), (w_roi - 1, roi_start + h_roi - 1), (0, 255, 255), 2)

    # centre image
    cx_ref = w_roi // 2
    cv2.line(out, (cx_ref, roi_start), (cx_ref, roi_start + h_roi - 1), (255, 0, 0), 2)

    # fonction locale pour dessiner une tranche
    def draw_one_slice(y, cx, color, name):
        yy = roi_start + y

        # bande horizontale
        y0 = max(roi_start, yy - band_thickness)
        y1 = min(roi_start + h_roi - 1, yy + band_thickness)

        cv2.rectangle(out, (0, y0), (w_roi - 1, y1), color, 1)
        cv2.line(out, (0, yy), (w_roi - 1, yy), color, 1)

        # centre détecté
        if cx is not None:
            cv2.circle(out, (cx, yy), 6, color, -1)
            cv2.putText(out, f"{name}: {cx}", (10, yy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(out, f"{name}: None", (10, yy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    draw_one_slice(y_top, cx_top, (0, 255, 0), "TOP")
    draw_one_slice(y_mid, cx_mid, (0, 200, 255), "MID")
    draw_one_slice(y_bot, cx_bot, (0, 0, 255), "BOT")

    return out

def draw_mask_slices(mask, y_top, y_mid, y_bot, cx_top, cx_mid, cx_bot):
    """
    Retourne une image couleur du masque pour visualiser les tranches.
    """
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def draw_one(y, cx, color):
        cv2.line(vis, (0, y), (mask.shape[1] - 1, y), color, 1)
        if cx is not None:
            cv2.circle(vis, (cx, y), 5, color, -1)

    draw_one(y_top, cx_top, (0, 255, 0))
    draw_one(y_mid, cx_mid, (0, 200, 255))
    draw_one(y_bot, cx_bot, (0, 0, 255))

    cx_ref = mask.shape[1] // 2
    cv2.line(vis, (cx_ref, 0), (cx_ref, mask.shape[0] - 1), (255, 0, 0), 1)

    return vis