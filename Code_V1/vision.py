# vision_top_down.py
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from config import (
    H, W,
    ROI_START_FRAC,
    KERNEL_SIZE, MORPH_CLOSE_IT, MORPH_OPEN_IT,
    BAND_THICKNESS,
    SLICE_TOP_FRAC, SLICE_MID_FRAC, SLICE_BOT_FRAC,
    W_TOP, W_MID, W_BOT,

    # bottom gating
    BOTTOM_SLICE_MAX_DIST,
    CENTER_EXCEPTION_DIST,

    # T detection
    T_WIDTH_RATIO,
    T_MIN_ABS_WIDTH,
    T_CENTER_MAX_ERR,
    T_MIN_SIDE_EXT,
    T_MAX_SIDE_RATIO,
    T_MIN_WIDTH_GAIN,
    T_STEM_MIN_ROWS,
    T_STEM_X_TOL,

    # vertical scan
    T_SCAN_Y_START_FRAC,
    T_SCAN_Y_END_FRAC,
    T_MIN_BAR_ROWS,

    # anti false-T
    T_MIN_BALANCE_RATIO,
    T_UP_CHECK_ROWS,
    T_MAX_UP_CONTINUITY_ROWS,
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

    t_detected: bool
    t_row: int | None

    mask: np.ndarray
    connected_mask: np.ndarray


def correct_top_down(mask: np.ndarray, angle_deg: float = 20) -> np.ndarray:
    scale = 1 / np.cos(np.radians(angle_deg))
    h, w = mask.shape
    new_h = int(h * scale)
    return cv2.resize(mask, (w, new_h))


def compute_mask_from_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    roi_start = int(H * ROI_START_FRAC)
    roi = frame_bgr[roi_start:H, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 70], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_IT)

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

    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None, y, None, None, 0, 0

    xs_full = xs + x_offset
    x_min = int(xs_full.min())
    x_max = int(xs_full.max())
    cx = int(xs_full.mean())
    width = x_max - x_min
    count = int(xs_full.size)

    return cx, y, x_min, x_max, width, count


def keep_bottom_connected_component(mask: np.ndarray, cx_ref: int) -> np.ndarray:
    """
    Keep only the connected component touching the bottom allowed window.
    """
    h, w = mask.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    x0 = max(0, cx_ref - BOTTOM_SLICE_MAX_DIST)
    x1 = min(w, cx_ref + BOTTOM_SLICE_MAX_DIST + 1)

    bottom_band = labels[max(0, h - 2):h, x0:x1]
    candidate_labels = np.unique(bottom_band)
    candidate_labels = candidate_labels[candidate_labels != 0]

    if candidate_labels.size == 0:
        return np.zeros_like(mask)

    best_label = None
    best_area = -1
    for lab in candidate_labels:
        area = stats[lab, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_label = lab

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def get_row_run_around_x(row_mask: np.ndarray, x_ref: int) -> tuple[int | None, int | None]:
    """
    Return the contiguous white run [x_min, x_max] that contains x_ref
    or the nearest local run around x_ref.
    """
    w = row_mask.shape[0]
    if w == 0:
        return None, None

    x_ref = max(0, min(w - 1, x_ref))

    if not row_mask[x_ref]:
        radius = 12
        x0 = max(0, x_ref - radius)
        x1 = min(w, x_ref + radius + 1)
        xs = np.where(row_mask[x0:x1])[0]
        if xs.size == 0:
            return None, None
        xs_full = x0 + xs
        x_ref = int(xs_full[np.argmin(np.abs(xs_full - x_ref))])

    x_min = x_ref
    while x_min > 0 and row_mask[x_min - 1]:
        x_min -= 1

    x_max = x_ref
    while x_max < w - 1 and row_mask[x_max + 1]:
        x_max += 1

    return int(x_min), int(x_max)


def estimate_local_stem_direction(
    connected_mask: np.ndarray,
    cx_bot: int,
    y_bot: int,
    y_start: int,
) -> tuple[float, list[tuple[int, int]]]:
    """
    Follow the stem upward and estimate local slope dx/dy.
    """
    points: list[tuple[int, int]] = []
    last_x = cx_bot
    h, w = connected_mask.shape

    y0 = min(h - 1, y_bot)
    y1 = max(0, y_start)

    for y in range(y0, y1 - 1, -1):
        row = connected_mask[y, :] > 0
        x_min, x_max = get_row_run_around_x(row, last_x)
        if x_min is None:
            continue

        cx = (x_min + x_max) // 2
        points.append((cx, y))
        last_x = cx

    if len(points) < 2:
        return 0.0, points

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)

    y_mean = ys.mean()
    denom = np.sum((ys - y_mean) ** 2)
    if denom < 1e-6:
        return 0.0, points

    slope = float(np.sum((ys - y_mean) * (xs - xs.mean())) / denom)
    return slope, points


def detect_t_on_connected_mask(
    connected_mask: np.ndarray,
    cx_bot: int | None,
    width_bot: int,
    y_bot: int,
) -> tuple[bool, int | None]:
    """
    T detection tuned to be practical:
    - start from bottom stem
    - scan upward around bottom x
    - require strong width increase
    - require left and right extension
    - require stem continuity below
    - only light anti-turn filtering
    """
    if cx_bot is None or width_bot <= 0:
        return False, None

    h, w = connected_mask.shape
    cx_ref = w // 2

    # Keep this, but not too strict
    if abs(cx_bot - cx_ref) > T_CENTER_MAX_ERR:
        return False, None

    y_start = int(T_SCAN_Y_START_FRAC * h)
    y_end = int(T_SCAN_Y_END_FRAC * h)
    if y_start > y_end:
        y_start, y_end = y_end, y_start

    consecutive_hits = 0
    first_hit_row = None
    candidate_row = None
    candidate_x = None

    # small horizontal drift allowed while scanning upward
    last_x = cx_bot

    for y in range(y_end, y_start - 1, -1):
        row = connected_mask[y, :] > 0
        if not np.any(row):
            consecutive_hits = 0
            first_hit_row = None
            candidate_row = None
            candidate_x = None
            continue

        # search near previous x, not only exactly at cx_bot
        x_min, x_max = get_row_run_around_x(row, last_x)
        if x_min is None or x_max is None:
            consecutive_hits = 0
            first_hit_row = None
            candidate_row = None
            candidate_x = None
            continue

        run_cx = (x_min + x_max) // 2
        width = x_max - x_min

        # allow stem to move a bit
        stem_x = last_x
        stem_x = max(x_min, min(x_max, stem_x))

        left_ext = stem_x - x_min
        right_ext = x_max - stem_x

        width_ok = (
            width >= T_MIN_ABS_WIDTH and
            width >= T_WIDTH_RATIO * max(1, width_bot) and
            (width - width_bot) >= T_MIN_WIDTH_GAIN
        )

        both_sides_present = (
            left_ext >= T_MIN_SIDE_EXT and
            right_ext >= T_MIN_SIDE_EXT
        )

        # lighter balance filter than before
        smaller = min(left_ext, right_ext)
        larger = max(left_ext, right_ext)
        balance_ratio = smaller / max(1, larger)

        balanced_enough = balance_ratio >= 0.18

        bar_ok = width_ok and both_sides_present and balanced_enough

        if bar_ok:
            consecutive_hits += 1
            if first_hit_row is None:
                first_hit_row = y
            candidate_row = y
            candidate_x = run_cx
        else:
            consecutive_hits = 0
            first_hit_row = None
            candidate_row = None
            candidate_x = None

        # update tracking even before T hit
        last_x = run_cx

        if consecutive_hits >= T_MIN_BAR_ROWS:
            break

    if candidate_row is None or candidate_x is None:
        return False, None

    # validate stem continuity below bar
    y0 = candidate_row
    y1 = min(h - 1, candidate_row + T_STEM_MIN_ROWS)

    stem_hits = 0
    total_rows = max(1, y1 - y0 + 1)
    last_x = candidate_x

    for y in range(y0, y1 + 1):
        row = connected_mask[y, :] > 0
        x_min, x_max = get_row_run_around_x(row, last_x)
        if x_min is None:
            continue

        run_cx = (x_min + x_max) // 2
        last_x = run_cx

        x_left = max(0, run_cx - T_STEM_X_TOL)
        x_right = min(w, run_cx + T_STEM_X_TOL + 1)

        if np.any(connected_mask[y, x_left:x_right] > 0):
            stem_hits += 1

    required_hits = max(4, int(0.5 * total_rows))
    if stem_hits < required_hits:
        return False, None

    return True, first_hit_row


def detect_line(frame_bgr: np.ndarray, angle_deg: float = 20) -> VisionResult:
    mask, roi_start = compute_mask_from_frame(frame_bgr)

    mask_td = correct_top_down(mask, angle_deg)
    h_td, w_td = mask_td.shape
    cx_ref = w_td // 2

    connected_mask = keep_bottom_connected_component(mask_td, cx_ref)

    y_top = int(SLICE_TOP_FRAC * h_td)
    y_mid = int(SLICE_MID_FRAC * h_td)
    y_bot = int(SLICE_BOT_FRAC * h_td)

    cx_top, _, _, _, width_top, _ = get_band_info(
        connected_mask, SLICE_TOP_FRAC, BAND_THICKNESS
    )
    cx_mid, _, _, _, width_mid, _ = get_band_info(
        connected_mask, SLICE_MID_FRAC, BAND_THICKNESS
    )
    cx_bot, _, _, _, width_bot, _ = get_band_info(
        connected_mask,
        SLICE_BOT_FRAC,
        BAND_THICKNESS,
        x_center=cx_ref,
        max_x_dist=BOTTOM_SLICE_MAX_DIST
    )

    seen_bot = cx_bot is not None

    top_near_center = (
        cx_top is not None and abs(cx_top - cx_ref) <= CENTER_EXCEPTION_DIST
    )
    mid_near_center = (
        cx_mid is not None and abs(cx_mid - cx_ref) <= CENTER_EXCEPTION_DIST
    )

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
    else:
        if seen_mid:
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

    t_detected, t_row = detect_t_on_connected_mask(
        connected_mask=connected_mask,
        cx_bot=cx_bot,
        width_bot=width_bot,
        y_bot=y_bot,
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
        width_top=width_top,
        width_mid=width_mid,
        width_bot=width_bot,
        t_detected=t_detected,
        t_row=t_row,
        mask=mask_td,
        connected_mask=connected_mask,
    )


def draw_debug_slices(frame_bgr, vision_result, band_thickness):
    out = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)

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
        2
    )

    def draw_one(y, cx, color, name):
        cv2.line(out, (0, roi_start + y), (w_roi - 1, roi_start + y), color, 1)
        if cx is not None:
            cv2.circle(out, (cx, roi_start + y), 5, color, -1)
        cv2.putText(
            out,
            f"{name}: {cx}",
            (10, roi_start + y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    draw_one(vision_result.y_top, vision_result.cx_top, (0, 255, 0), "TOP")
    draw_one(vision_result.y_mid, vision_result.cx_mid, (0, 200, 255), "MID")
    draw_one(vision_result.y_bot, vision_result.cx_bot, (0, 0, 255), "BOT")

    if vision_result.t_row is not None:
        cv2.line(
            out,
            (0, roi_start + vision_result.t_row),
            (w_roi - 1, roi_start + vision_result.t_row),
            (255, 0, 255),
            2
        )

    if vision_result.t_detected:
        cv2.putText(
            out,
            "T DETECTED",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

    cv2.putText(
        out,
        f"Steer error: {vision_result.steer_error:.2f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return out


def draw_mask_slices(vision_result):
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
        2
    )

    cv2.line(vis, (cx_ref, 0), (cx_ref, vis.shape[0] - 1), (255, 0, 0), 1)

    if vision_result.t_row is not None:
        cv2.line(vis, (0, vision_result.t_row), (vis.shape[1] - 1, vision_result.t_row), (255, 0, 255), 2)

    if vision_result.t_detected:
        cv2.putText(
            vis,
            "T DETECTED",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

    return vis