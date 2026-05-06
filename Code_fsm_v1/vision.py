from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    BAND_THICKNESS,
    BLACK_H_HIGH,
    BLACK_H_LOW,
    BLACK_S_HIGH,
    BLACK_S_LOW,
    BLACK_V_HIGH,
    BLACK_V_LOW,
    BOTTOM_SLICE_MAX_DIST,
    CENTER_EXCEPTION_DIST,
    CORNER_APPROX_EPSILON_FRAC,
    CORNER_MAX_ANGLE_DEG,
    CORNER_MIN_CONTOUR_AREA,
    CORNER_PADDING,
    GAUSS_KSIZE,
    H,
    KERNEL_SIZE,
    MORPH_CLOSE_IT,
    ROI_START_FRAC,
    SHAPE_MORPH_MODE,
    SLICE_BOT_FRAC,
    SLICE_POINT_HISTORY_FRAMES,
    SLICE_MID_FRAC,
    SLICE_TOP_FRAC,
    T_BOTTOM_SLICE_WHITE_RATIO,
    T_CORNERS,
    T_SIDE_MIN_X_OFFSET,
    W_BOT,
    W_MID,
    W_TOP,
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
    width_top: int
    width_mid: int
    width_bot: int
    bottom_slice_full_white: bool
    t_detected: bool
    corner_count: int
    corners: list[tuple[int, int]]
    t_left_bar_x: int | None
    t_right_bar_x: int | None
    mask: np.ndarray
    connected_mask: np.ndarray
    shape_mask: np.ndarray


class SlicePointHistory:
    def __init__(self, maxlen: int):
        self._history = {
            "top": deque(maxlen=maxlen),
            "mid": deque(maxlen=maxlen),
            "bot": deque(maxlen=maxlen),
            "bot_gated": deque(maxlen=maxlen),
        }

    def update(self, key: str, value: int | None) -> int | None:
        history = self._history[key]
        if value is None:
            history.clear()
            return None

        history.append(value)
        valid = list(history)
        return int(round(sum(valid) / len(valid)))


SLICE_POINT_HISTORY = SlicePointHistory(SLICE_POINT_HISTORY_FRAMES)


def remove_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    h, w = flood.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def apply_shape_morphology(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    if SHAPE_MORPH_MODE == "remove_holes":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return remove_holes(mask)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)


def compute_shape_mask_from_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    roi_start = int(H * ROI_START_FRAC)
    roi = frame_bgr[roi_start:H, :]
    roi_blur = cv2.GaussianBlur(roi, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    lower_black = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    upper_black = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = apply_shape_morphology(mask)
    return mask, roi_start


def get_band_info(
    mask: np.ndarray,
    y_frac: float,
    thickness: int,
    x_center: int | None = None,
    max_x_dist: int | None = None,
) -> tuple[int | None, int, int | None, int | None, int, int]:
    h, w = mask.shape
    y = int(h * y_frac)
    y0 = max(0, y - thickness)
    y1 = min(h, y + thickness + 1)
    band = mask[y0:y1, :]

    x_offset = 0
    if x_center is not None and max_x_dist is not None:
        x0 = max(0, x_center - max_x_dist)
        x1 = min(w, x_center + max_x_dist + 1)
        band = band[:, x0:x1]
        x_offset = x0

    _, xs = np.where(band > 0)
    if xs.size == 0:
        return None, y, None, None, 0, 0

    xs_full = xs + x_offset
    x_min = int(xs_full.min())
    x_max = int(xs_full.max())
    cx = int(xs_full.mean())
    width = int(x_max - x_min + 1)
    count = int(xs_full.size)
    return cx, y, x_min, x_max, width, count


def is_band_full_white(mask: np.ndarray, y_frac: float, thickness: int) -> bool:
    h, w = mask.shape
    y = int(h * y_frac)
    y0 = max(0, y - thickness)
    y1 = min(h, y + thickness + 1)

    pad = min(CORNER_PADDING, h // 2, w // 2)
    x0_inner = pad
    x1_inner = w - pad
    y0_inner = max(y0, pad)
    y1_inner = min(y1, h - pad)
    if x1_inner <= x0_inner or y1_inner <= y0_inner:
        return False

    band = mask[y0_inner:y1_inner, x0_inner:x1_inner]
    white_ratio = float(np.count_nonzero(band > 0)) / float(band.size)
    return white_ratio >= T_BOTTOM_SLICE_WHITE_RATIO


def keep_shape_connected_to_bottom_rect(mask: np.ndarray, x_center: int, max_x_dist: int) -> np.ndarray:
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    pad = min(CORNER_PADDING, max(0, h - 1))
    anchor_bottom = max(1, h - pad)
    x0 = max(0, x_center - max_x_dist)
    x1 = min(w, x_center + max_x_dist + 1)
    bottom_band = labels[max(0, anchor_bottom - 2):anchor_bottom, x0:x1]
    candidate_labels = np.unique(bottom_band)
    candidate_labels = candidate_labels[candidate_labels != 0]
    if candidate_labels.size == 0:
        return np.zeros_like(mask)

    best_label = None
    best_area = -1
    for lab in candidate_labels:
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = lab

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def add_corner_padding(mask: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(
        mask,
        CORNER_PADDING,
        CORNER_PADDING,
        CORNER_PADDING,
        CORNER_PADDING,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )


def apply_black_border(mask: np.ndarray) -> np.ndarray:
    if CORNER_PADDING <= 0:
        return mask.copy()

    bordered = mask.copy()
    pad = min(CORNER_PADDING, bordered.shape[0] // 2, bordered.shape[1] // 2)
    if pad <= 0:
        return bordered

    bordered[:pad, :] = 0
    bordered[-pad:, :] = 0
    bordered[:, :pad] = 0
    bordered[:, -pad:] = 0
    return bordered


def is_near_frame_edge(x: int, y: int, width: int, height: int, margin: int) -> bool:
    return x <= margin or y <= margin or x >= width - 1 - margin or y >= height - 1 - margin


def count_t_side_corners(
    corners: list[tuple[int, int]],
) -> tuple[int, int, list[tuple[int, int]], int | None, int | None]:
    if len(corners) < 2:
        return 0, 0, [], None, None

    bottom_pair = sorted(corners, key=lambda p: (p[1], p[0]), reverse=True)[:2]
    bottom_x_min = min(point[0] for point in bottom_pair)
    bottom_x_max = max(point[0] for point in bottom_pair)
    left_bar_x = bottom_x_min - T_SIDE_MIN_X_OFFSET
    right_bar_x = bottom_x_max + T_SIDE_MIN_X_OFFSET
    bottom_pair_set = set(bottom_pair)

    left_count = 0
    right_count = 0
    for point in corners:
        if point in bottom_pair_set:
            continue
        x, _ = point
        if x <= left_bar_x:
            left_count += 1
        elif x >= right_bar_x:
            right_count += 1

    return left_count, right_count, bottom_pair, left_bar_x, right_bar_x


def extract_corners_from_connected_mask(
    connected_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[float], np.ndarray | None]:
    padded = add_corner_padding(connected_mask)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], [], None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < CORNER_MIN_CONTOUR_AREA:
        return [], [], biggest

    perimeter = cv2.arcLength(biggest, True)
    epsilon = CORNER_APPROX_EPSILON_FRAC * perimeter
    approx = cv2.approxPolyDP(biggest, epsilon, True)
    approx_points = approx.reshape(-1, 2)
    if len(approx_points) < 3:
        return [], [], biggest

    corners: list[tuple[int, int]] = []
    corner_angles: list[float] = []
    edge_margin = max(1, 2 * CORNER_PADDING)
    mask_h, mask_w = connected_mask.shape

    for i, point in enumerate(approx_points):
        px = int(point[0])
        py = int(point[1])
        x = px - CORNER_PADDING
        y = py - CORNER_PADDING

        if is_near_frame_edge(x, y, mask_w, mask_h, edge_margin):
            corners.append((x, y))
            corner_angles.append(0.0)
            continue

        prev_point = approx_points[(i - 1) % len(approx_points)].astype(np.float32)
        curr_point = point.astype(np.float32)
        next_point = approx_points[(i + 1) % len(approx_points)].astype(np.float32)

        v1 = prev_point - curr_point
        v2 = next_point - curr_point
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = float(np.dot(v1, v2) / (norm1 * norm2))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if angle_deg > CORNER_MAX_ANGLE_DEG:
            continue

        corners.append((x, y))
        corner_angles.append(angle_deg)

    return corners, corner_angles, biggest


def detect_line(frame_bgr: np.ndarray, angle_deg: float = 20) -> VisionResult:
    shape_mask_raw, roi_start = compute_shape_mask_from_frame(frame_bgr)
    shared_mask = apply_black_border(shape_mask_raw)
    shared_mask = keep_shape_connected_to_bottom_rect(
        shared_mask,
        x_center=shared_mask.shape[1] // 2,
        max_x_dist=BOTTOM_SLICE_MAX_DIST,
    )
    shared_mask = apply_black_border(shared_mask)
    h_td, w_td = shared_mask.shape
    cx_ref = w_td // 2

    y_top = int(SLICE_TOP_FRAC * h_td)
    y_mid = int(SLICE_MID_FRAC * h_td)
    y_bot = int(SLICE_BOT_FRAC * h_td)

    raw_cx_top, _, _, _, width_top, _ = get_band_info(shared_mask, SLICE_TOP_FRAC, BAND_THICKNESS)
    raw_cx_mid, _, _, _, width_mid, _ = get_band_info(shared_mask, SLICE_MID_FRAC, BAND_THICKNESS)
    raw_cx_bot, _, _, _, width_bot, _ = get_band_info(shared_mask, SLICE_BOT_FRAC, BAND_THICKNESS)
    raw_cx_bot_gated, _, _, _, width_bot_gated, _ = get_band_info(
        shared_mask,
        SLICE_BOT_FRAC,
        BAND_THICKNESS,
        x_center=cx_ref,
        max_x_dist=BOTTOM_SLICE_MAX_DIST,
    )

    cx_top = SLICE_POINT_HISTORY.update("top", raw_cx_top)
    cx_mid = SLICE_POINT_HISTORY.update("mid", raw_cx_mid)
    cx_bot = SLICE_POINT_HISTORY.update("bot", raw_cx_bot)
    cx_bot_gated = SLICE_POINT_HISTORY.update("bot_gated", raw_cx_bot_gated)
    bottom_slice_full_white = is_band_full_white(shared_mask, SLICE_BOT_FRAC, BAND_THICKNESS)

    seen_bot = cx_bot_gated is not None
    top_near_center = cx_top is not None and abs(cx_top - cx_ref) <= CENTER_EXCEPTION_DIST
    mid_near_center = cx_mid is not None and abs(cx_mid - cx_ref) <= CENTER_EXCEPTION_DIST
    allow_top = seen_bot or top_near_center
    allow_mid = seen_bot or mid_near_center

    if not allow_top:
        cx_top = None
        width_top = 0
    if not allow_mid:
        cx_mid = None
        width_mid = 0

    seen_top = cx_top is not None
    seen_mid = cx_mid is not None
    any_seen = seen_bot or seen_mid or seen_top
    all_seen = seen_bot and seen_mid and seen_top

    if seen_bot:
        cx_bot_use = cx_bot
        cx_mid_use = cx_mid if cx_mid is not None else cx_bot
        cx_top_use = cx_top if cx_top is not None else cx_mid_use
    elif seen_mid:
        cx_mid_use = cx_mid
        cx_top_use = cx_top if cx_top is not None else cx_mid
        cx_bot_use = cx_mid
    elif seen_top:
        cx_top_use = cx_top
        cx_mid_use = cx_top
        cx_bot_use = cx_top
    else:
        cx_top_use = cx_ref
        cx_mid_use = cx_ref
        cx_bot_use = cx_ref

    err_top = float(cx_top_use - cx_ref)
    err_mid = float(cx_mid_use - cx_ref)
    err_bot = float(cx_bot_use - cx_ref)
    steer_error = W_TOP * err_top + W_MID * err_mid + W_BOT * err_bot

    corners, _, _ = extract_corners_from_connected_mask(shared_mask)
    corner_count = len(corners)
    left_side_corners, right_side_corners, bottom_pair, t_left_bar_x, t_right_bar_x = count_t_side_corners(corners)

    shape_detection_allowed = seen_bot and width_bot_gated > 0
    if shape_detection_allowed:
        t_detected = (
            corner_count >= T_CORNERS
            and len(bottom_pair) == 2
            and left_side_corners >= 2
            and right_side_corners >= 2
        )
        shape_connected_mask = shared_mask
    else:
        t_detected = False
        corner_count = 0
        corners = []
        shape_connected_mask = np.zeros_like(shared_mask)

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
        width_top=width_top,
        width_mid=width_mid,
        width_bot=width_bot,
        bottom_slice_full_white=bottom_slice_full_white,
        t_detected=t_detected,
        corner_count=corner_count,
        corners=corners,
        t_left_bar_x=t_left_bar_x,
        t_right_bar_x=t_right_bar_x,
        mask=shared_mask,
        connected_mask=shared_mask,
        shape_mask=shape_connected_mask,
    )


def draw_debug_slices(
    frame_bgr: np.ndarray,
    vision_result: VisionResult,
    band_thickness: int,
    wheel_left: float | None = None,
    wheel_right: float | None = None,
):
    out = frame_bgr.copy()
    mask = vision_result.connected_mask
    h_roi, w_roi = mask.shape
    roi_start = vision_result.roi_start
    cx_ref = w_roi // 2

    cv2.rectangle(out, (0, roi_start), (w_roi - 1, roi_start + h_roi - 1), (0, 255, 255), 2)
    cv2.line(out, (cx_ref, roi_start), (cx_ref, roi_start + h_roi - 1), (255, 0, 0), 2)

    x0 = max(0, cx_ref - BOTTOM_SLICE_MAX_DIST)
    x1 = min(w_roi - 1, cx_ref + BOTTOM_SLICE_MAX_DIST)
    cv2.rectangle(
        out,
        (x0, roi_start + max(0, vision_result.y_bot - band_thickness)),
        (x1, roi_start + min(h_roi - 1, vision_result.y_bot + band_thickness)),
        (255, 255, 0),
        2,
    )

    for bar_x in [vision_result.t_left_bar_x, vision_result.t_right_bar_x]:
        if bar_x is None:
            continue
        bar_x = max(0, min(w_roi - 1, bar_x))
        cv2.line(out, (bar_x, roi_start), (bar_x, roi_start + h_roi - 1), (255, 0, 255), 2)

    def draw_one(y: int, cx: int | None, color: tuple[int, int, int], name: str):
        cv2.line(out, (0, roi_start + y), (w_roi - 1, roi_start + y), color, 1)
        if cx is not None:
            cv2.circle(out, (cx, roi_start + y), 5, color, -1)
        cv2.putText(out, f"{name}: {cx}", (10, roi_start + y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

    draw_one(vision_result.y_top, vision_result.cx_top, (0, 255, 0), "TOP")
    draw_one(vision_result.y_mid, vision_result.cx_mid, (0, 200, 255), "MID")
    draw_one(vision_result.y_bot, vision_result.cx_bot, (0, 0, 255), "BOT")

    width_lines = [
        (f"TOP W: {vision_result.width_top}", (0, 255, 0)),
        (f"MID W: {vision_result.width_mid}", (0, 200, 255)),
        (f"BOT W: {vision_result.width_bot}", (0, 0, 255)),
    ]
    width_x = max(10, w_roi - 120)
    width_y = 22
    for index, (label, color) in enumerate(width_lines):
        y = width_y + index * 22
        cv2.putText(out, label, (width_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, label, (width_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    cv2.putText(out, f"Corners: {vision_result.corner_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(
        out,
        f"T: {'YES' if vision_result.t_detected else 'NO'}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.325,
        (0, 255, 0) if vision_result.t_detected else (0, 0, 255),
        1,
    )
    cv2.putText(
        out,
        f"Steer error: {vision_result.steer_error:.2f}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 255, 0),
        1,
    )

    if wheel_left is not None and wheel_right is not None:
        wheel_label = f"L {wheel_left:.2f}  R {wheel_right:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(wheel_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = max(0, (out.shape[1] - text_w) // 2)
        text_y = max(text_h + 8, 22)
        pad_x = 8
        pad_y = 6
        cv2.rectangle(
            out,
            (max(0, text_x - pad_x), max(0, text_y - text_h - pad_y)),
            (min(out.shape[1] - 1, text_x + text_w + pad_x), min(out.shape[0] - 1, text_y + baseline + pad_y)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(out, wheel_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def draw_mask_slices(vision_result: VisionResult):
    vis = cv2.cvtColor(vision_result.connected_mask, cv2.COLOR_GRAY2BGR)
    cx_ref = vis.shape[1] // 2

    for y, cx, color in [
        (vision_result.y_top, vision_result.cx_top, (0, 255, 0)),
        (vision_result.y_mid, vision_result.cx_mid, (0, 200, 255)),
        (vision_result.y_bot, vision_result.cx_bot, (0, 0, 255)),
    ]:
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), color, 1)
        if cx is not None:
            cv2.circle(vis, (cx, y), 5, color, -1)

    x0 = max(0, cx_ref - BOTTOM_SLICE_MAX_DIST)
    x1 = min(vis.shape[1] - 1, cx_ref + BOTTOM_SLICE_MAX_DIST)
    cv2.rectangle(
        vis,
        (x0, max(0, vision_result.y_bot - BAND_THICKNESS)),
        (x1, min(vis.shape[0] - 1, vision_result.y_bot + BAND_THICKNESS)),
        (255, 255, 0),
        2,
    )
    cv2.line(vis, (cx_ref, 0), (cx_ref, vis.shape[0] - 1), (255, 0, 0), 1)

    for bar_x in [vision_result.t_left_bar_x, vision_result.t_right_bar_x]:
        if bar_x is None:
            continue
        bar_x = max(0, min(vis.shape[1] - 1, bar_x))
        cv2.line(vis, (bar_x, 0), (bar_x, vis.shape[0] - 1), (255, 0, 255), 1)

    for x, y in vision_result.corners:
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)

    cv2.putText(vis, f"Corners: {vision_result.corner_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    if vision_result.t_detected:
        cv2.putText(vis, "T DETECTED", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    return vis
