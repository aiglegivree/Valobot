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
    T_CENTER_MAX_ERR,
    T_SCAN_Y_START_FRAC,
    T_SCAN_Y_END_FRAC,
    T_TRACK_MAX_STEP,
    T_MIN_TRACK_POINTS,
    T_CANDIDATE_MIN_WIDTH,
    T_CANDIDATE_MIN_WIDTH_GAIN,
    T_CANDIDATE_MIN_WIDTH_RATIO,
    T_SIDE_PROBE_LEN,
    T_SIDE_PROBE_HALF_WIDTH,
    T_BACK_PROBE_LEN,
    T_BACK_PROBE_HALF_WIDTH,
    T_SIDE_MIN_HITS,
    T_BACK_MIN_HITS,
    T_MIN_BALANCE_RATIO,
    T_DIRECTION_DOT_MAX,
    T_SIDE_ANGLE_DEG,
    T_SIDE_ANGLE_STEP_DEG,
    T_CONFIRM_ROWS,
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

def rotate_vec(v: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c = np.cos(a)
    s = np.sin(a)
    return np.array([
        c * v[0] - s * v[1],
        s * v[0] + c * v[1]
    ], dtype=np.float32)


def best_probe_in_angle_fan(
    mask: np.ndarray,
    p: np.ndarray,
    base_dir: np.ndarray,
    trunk_dir: np.ndarray,
    length: int,
    half_width: int,
    angle_deg: float,
    angle_step_deg: float,
) -> tuple[int, np.ndarray | None]:
    """
    Test several probe directions around base_dir and keep the best one.
    This makes lateral branch detection robust to tilted T bars.
    """
    best_hits = -1
    best_dir = None

    angles = np.arange(-angle_deg, angle_deg + 0.1, angle_step_deg, dtype=np.float32)

    for a in angles:
        d = rotate_vec(base_dir, float(a))
        hits, mean_dir = sample_probe_hits(mask, p, d, length, half_width)

        if hits <= 0:
            continue

        # garde seulement une branche assez latérale par rapport au tronc
        if mean_dir is not None:
            if abs(float(np.dot(mean_dir, trunk_dir))) > T_DIRECTION_DOT_MAX:
                continue

        if hits > best_hits:
            best_hits = hits
            best_dir = mean_dir if mean_dir is not None else d

    if best_hits < 0:
        return 0, None

    return best_hits, best_dir

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


def get_row_run_around_x(row_mask: np.ndarray, x_ref: int, search_radius: int = 20) -> tuple[int | None, int | None]:
    """
    Return the contiguous white run [x_min, x_max] that contains x_ref
    or the nearest local run around x_ref.
    """
    w = row_mask.shape[0]
    if w == 0:
        return None, None

    x_ref = max(0, min(w - 1, x_ref))

    if not row_mask[x_ref]:
        x0 = max(0, x_ref - search_radius)
        x1 = min(w, x_ref + search_radius + 1)
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


def track_stem_points(
    connected_mask: np.ndarray,
    cx_bot: int,
    y_bot: int,
    y_start: int,
) -> list[dict]:
    """
    Track the same connected run upward from the bottom.
    Each point stores center, row, width, and run bounds.
    """
    h, w = connected_mask.shape
    last_x = int(cx_bot)
    points: list[dict] = []

    y0 = min(h - 1, y_bot)
    y1 = max(0, y_start)

    miss_count = 0

    for y in range(y0, y1 - 1, -1):
        row = connected_mask[y, :] > 0
        if not np.any(row):
            miss_count += 1
            if miss_count > 5:
                break
            continue

        x_min, x_max = get_row_run_around_x(row, last_x, search_radius=T_TRACK_MAX_STEP)
        if x_min is None or x_max is None:
            miss_count += 1
            if miss_count > 5:
                break
            continue

        cx = (x_min + x_max) // 2

        if points and abs(cx - last_x) > T_TRACK_MAX_STEP:
            miss_count += 1
            if miss_count > 5:
                break
            continue

        miss_count = 0
        last_x = cx

        points.append({
            "x": int(cx),
            "y": int(y),
            "x_min": int(x_min),
            "x_max": int(x_max),
            "width": int(x_max - x_min),
        })

    return points


def estimate_local_tangent(points: list[dict], idx: int, span: int = 6) -> np.ndarray | None:
    """
    Estimate local tangent of the tracked stem around a candidate point.
    Tangent is oriented from bottom toward top.
    """
    n = len(points)
    i0 = max(0, idx - span)
    i1 = min(n - 1, idx + span)

    if i1 - i0 < 2:
        return None

    xs = np.array([points[i]["x"] for i in range(i0, i1 + 1)], dtype=np.float32)
    ys = np.array([points[i]["y"] for i in range(i0, i1 + 1)], dtype=np.float32)

    # bottom -> top orientation: y decreases
    p0 = np.array([xs[0], ys[0]], dtype=np.float32)
    p1 = np.array([xs[-1], ys[-1]], dtype=np.float32)
    v = p1 - p0

    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return None

    return v / norm


def sample_probe_hits(
    mask: np.ndarray,
    p: np.ndarray,
    direction: np.ndarray,
    length: int,
    half_width: int,
) -> tuple[int, np.ndarray | None]:
    """
    Sample a thin rectangular probe centered on the direction.
    Return number of steps with some white pixels and mean detected direction.
    """
    h, w = mask.shape
    d = direction.astype(np.float32)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-6:
        return 0, None
    d /= nrm

    n = np.array([-d[1], d[0]], dtype=np.float32)

    hits = 0
    hit_pts = []

    for s in range(1, length + 1):
        found_here = False
        base = p + d * s

        for o in range(-half_width, half_width + 1):
            q = base + n * o
            x = int(round(q[0]))
            y = int(round(q[1]))

            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                hits += 1
                hit_pts.append(np.array([x, y], dtype=np.float32))
                found_here = True

        if not found_here:
            # tolérance aux petits trous : on continue
            pass

    if len(hit_pts) < 2:
        return hits, None

    mean_dir = hit_pts[-1] - hit_pts[0]
    norm = float(np.linalg.norm(mean_dir))
    if norm < 1e-6:
        return hits, None

    return hits, mean_dir / norm


def detect_t_on_connected_mask(
    connected_mask: np.ndarray,
    cx_bot: int | None,
    width_bot: int,
    y_bot: int,
) -> tuple[bool, int | None]:
    """
    Detect a T in the local frame of the followed line:
    - track the stem upward,
    - estimate local tangent,
    - verify back continuity,
    - verify one branch on each side,
    - tolerate moderate tilt of the T bar.
    """
    if cx_bot is None or width_bot <= 0:
        return False, None

    h, w = connected_mask.shape
    cx_ref = w // 2

    if abs(cx_bot - cx_ref) > T_CENTER_MAX_ERR:
        return False, None

    y_start = int(T_SCAN_Y_START_FRAC * h)
    y_end = int(T_SCAN_Y_END_FRAC * h)
    if y_start > y_end:
        y_start, y_end = y_end, y_start

    points = track_stem_points(
        connected_mask=connected_mask,
        cx_bot=cx_bot,
        y_bot=y_bot,
        y_start=y_start,
    )

    if len(points) < T_MIN_TRACK_POINTS:
        return False, None

    best_row = None
    consecutive_ok = 0

    # points[0] = bas, points[-1] = haut
    for idx in range(len(points) - 2, max(6, len(points) // 3), -1):
        pt = points[idx]
        y = pt["y"]
        x = pt["x"]
        width = pt["width"]

        below_widths = [p["width"] for p in points[max(0, idx - 6):idx]]
        if len(below_widths) < 3:
            continue

        stem_width_ref = float(np.median(below_widths))

        width_ok = (
            width >= T_CANDIDATE_MIN_WIDTH and
            width >= T_CANDIDATE_MIN_WIDTH_RATIO * stem_width_ref and
            (width - stem_width_ref) >= T_CANDIDATE_MIN_WIDTH_GAIN
        )
        if not width_ok:
            consecutive_ok = 0
            continue

        t = estimate_local_tangent(points, idx, span=5)
        if t is None:
            consecutive_ok = 0
            continue

        n = np.array([-t[1], t[0]], dtype=np.float32)
        p = np.array([x, y], dtype=np.float32)

        # continuité du tronc vers l'arrière
        back_hits, _ = sample_probe_hits(
            connected_mask, p, -t, T_BACK_PROBE_LEN, T_BACK_PROBE_HALF_WIDTH
        )
        if back_hits < T_BACK_MIN_HITS:
            consecutive_ok = 0
            continue

        # branches latérales: on teste plusieurs angles autour de la normale
        left_hits, left_dir = best_probe_in_angle_fan(
            connected_mask,
            p,
            n,
            t,
            T_SIDE_PROBE_LEN,
            T_SIDE_PROBE_HALF_WIDTH,
            T_SIDE_ANGLE_DEG,
            T_SIDE_ANGLE_STEP_DEG,
        )

        right_hits, right_dir = best_probe_in_angle_fan(
            connected_mask,
            p,
            -n,
            t,
            T_SIDE_PROBE_LEN,
            T_SIDE_PROBE_HALF_WIDTH,
            T_SIDE_ANGLE_DEG,
            T_SIDE_ANGLE_STEP_DEG,
        )

        if left_hits < T_SIDE_MIN_HITS or right_hits < T_SIDE_MIN_HITS:
            consecutive_ok = 0
            continue

        balance = min(left_hits, right_hits) / max(left_hits, right_hits)
        if balance < T_MIN_BALANCE_RATIO:
            consecutive_ok = 0
            continue

        # sécurité: il faut bien une direction mesurée de chaque côté
        if left_dir is None or right_dir is None:
            consecutive_ok = 0
            continue

        consecutive_ok += 1
        best_row = y

        if consecutive_ok >= T_CONFIRM_ROWS:
            return True, best_row

    return False, None


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