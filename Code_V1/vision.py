# vision_top_down.py
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
    W_TOP, W_MID, W_BOT,
    MAX_DIST_FROM_BOTTOM_CENTER, MIN_CONTOUR_AREA
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


def correct_top_down(mask: np.ndarray, angle_deg: float = 20) -> np.ndarray:
    """
    Approximate top-down view of a flat surface given the camera tilt.
    Stretches the vertical axis to undo foreshortening.
    """
    scale = 1 / np.cos(np.radians(angle_deg))
    h, w = mask.shape
    new_h = int(h * scale)
    top_down = cv2.resize(mask, (w, new_h))
    return top_down


def contour_min_distance_to_point(cnt: np.ndarray, px: int, py: int) -> float:
    """
    Return the minimum Euclidean distance between a contour and a point.
    """
    pts = cnt[:, 0, :].astype(np.float32)   # shape (N, 2)
    dx = pts[:, 0] - px
    dy = pts[:, 1] - py
    d2 = dx * dx + dy * dy
    return float(np.sqrt(d2.min()))


def compute_mask_from_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Compute a cleaned mask for the ROI (line detection).
    Keep only contours that pass close enough to the bottom-center anchor.
    """
    roi_start = int(H * ROI_START_FRAC)
    roi = frame_bgr[roi_start:H, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 100, 100])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_IT)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)

    h_roi, w_roi = mask.shape
    anchor_x = w_roi // 2
    anchor_y = h_roi - 1

    best_contour = None
    best_score = float("inf")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # contrainte de distance au centre bas
        min_dist = contour_min_distance_to_point(cnt, anchor_x, anchor_y)
        if min_dist > MAX_DIST_FROM_BOTTOM_CENTER:
            continue

        # parmi les contours valides, on garde celui qui est le plus proche
        # du centre bas et qui descend le plus bas
        bottom_point = cnt[cnt[:, :, 1].argmax()][0]
        x_bottom, y_bottom = bottom_point

        score = min_dist - 0.2 * y_bottom
        if score < best_score:
            best_score = score
            best_contour = cnt

    if best_contour is not None:
        cv2.drawContours(clean_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    return clean_mask, roi_start


def get_band_center(mask: np.ndarray, y_frac: float, thickness: int) -> tuple[int | None, int]:
    """
    Compute the horizontal center of a band in the mask.
    """
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


def detect_line(frame_bgr: np.ndarray, angle_deg: float = 20) -> VisionResult:
    """
    Detect the line and compute top-down slices and steering error.
    """
    mask, roi_start = compute_mask_from_frame(frame_bgr)

    mask_td = correct_top_down(mask, angle_deg)
    h_td, w_td = mask_td.shape
    cx_ref = w_td // 2

    y_top = int(SLICE_TOP_FRAC * h_td)
    y_mid = int(SLICE_MID_FRAC * h_td)
    y_bot = int(SLICE_BOT_FRAC * h_td)

    cx_top, _ = get_band_center(mask_td, SLICE_TOP_FRAC, BAND_THICKNESS)
    cx_mid, _ = get_band_center(mask_td, SLICE_MID_FRAC, BAND_THICKNESS)
    cx_bot, _ = get_band_center(mask_td, SLICE_BOT_FRAC, BAND_THICKNESS)

    seen_top = cx_top is not None
    seen_mid = cx_mid is not None
    seen_bot = cx_bot is not None
    any_seen = seen_top or seen_mid or seen_bot
    all_seen = seen_top and seen_mid and seen_bot

    cx_bot_use = cx_bot or cx_mid or cx_top or cx_ref
    cx_mid_use = cx_mid or cx_bot_use
    cx_top_use = cx_top or cx_mid_use

    err_top = float(cx_top_use - cx_ref)
    err_mid = float(cx_mid_use - cx_ref)
    err_bot = float(cx_bot_use - cx_ref)

    steer_error = W_TOP * err_top + W_MID * err_mid + W_BOT * err_bot

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
        mask=mask_td
    )


def draw_debug_slices(frame_bgr, mask, roi_start, y_top, y_mid, y_bot,
                      cx_top, cx_mid, cx_bot, band_thickness, steer_error):
    """
    Draw ROI, slices, centers, steering error, and anchor zone on the frame.
    """
    out = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)
    h_roi, w_roi = mask.shape
    cx_ref = w_roi // 2

    cv2.rectangle(out, (0, roi_start), (w_roi - 1, roi_start + h_roi - 1), (0, 255, 255), 2)
    cv2.line(out, (cx_ref, roi_start), (cx_ref, roi_start + h_roi - 1), (255, 0, 0), 2)

    # cercle de validation autour du centre bas
    anchor = (cx_ref, roi_start + h_roi - 1)
    cv2.circle(out, anchor, MAX_DIST_FROM_BOTTOM_CENTER, (255, 255, 0), 2)
    cv2.circle(out, anchor, 5, (255, 255, 0), -1)

    def draw_one_slice(y, cx, color, name):
        y0 = max(0, y - band_thickness)
        y1 = min(h_roi - 1, y + band_thickness)
        cv2.rectangle(out, (0, roi_start + y0), (w_roi - 1, roi_start + y1), color, 1)
        cv2.line(out, (0, roi_start + y), (w_roi - 1, roi_start + y), color, 1)
        if cx is not None:
            cv2.circle(out, (cx, roi_start + y), 6, color, -1)
            cv2.putText(out, f"{name}: {cx}", (10, roi_start + y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(out, f"{name}: None", (10, roi_start + y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    draw_one_slice(y_top, cx_top, (0, 255, 0), "TOP")
    draw_one_slice(y_mid, cx_mid, (0, 200, 255), "MID")
    draw_one_slice(y_bot, cx_bot, (0, 0, 255), "BOT")

    cv2.putText(out, f"Steer error: {steer_error:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def draw_mask_slices(mask, y_top, y_mid, y_bot, cx_top, cx_mid, cx_bot):
    """
    Return a color image of the mask with slice overlays.
    """
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cx_ref = mask.shape[1] // 2

    def draw_one(y, cx, color):
        cv2.line(vis, (0, y), (mask.shape[1] - 1, y), color, 1)
        if cx is not None:
            cv2.circle(vis, (cx, y), 5, color, -1)

    draw_one(y_top, cx_top, (0, 255, 0))
    draw_one(y_mid, cx_mid, (0, 200, 255))
    draw_one(y_bot, cx_bot, (0, 0, 255))
    cv2.line(vis, (cx_ref, 0), (cx_ref, mask.shape[0] - 1), (255, 0, 0), 1)
    return vis