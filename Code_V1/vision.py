# vision.py
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
import config as cfg


# ============================================================
# CONFIG HELPERS
# ============================================================
def C(name: str, default):
    return getattr(cfg, name, default)


W = C("W", 640)
H = C("H", 480)

ROI_START_FRAC = C("ROI_START_FRAC", 0.50)
GAUSS_KSIZE = C("GAUSS_KSIZE", 5)
USE_BINARY_INV = C("USE_BINARY_INV", True)

KERNEL_SIZE = C("KERNEL_SIZE", 5)
MORPH_CLOSE_IT = C("MORPH_CLOSE_IT", 2)
MORPH_OPEN_IT = C("MORPH_OPEN_IT", 1)
MIN_COMPONENT_AREA = C("MIN_COMPONENT_AREA", 120)

BAND_THICKNESS = C("BAND_THICKNESS", 10)

SLICE_TOP_FRAC = C("SLICE_TOP_FRAC", 0.18)
SLICE_MID_FRAC = C("SLICE_MID_FRAC", 0.50)
SLICE_BOT_FRAC = C("SLICE_BOT_FRAC", 0.82)

W_TOP = C("W_TOP", 0.20)
W_MID = C("W_MID", 0.30)
W_BOT = C("W_BOT", 0.50)

BOTTOM_SLICE_MAX_DIST = C("BOTTOM_SLICE_MAX_DIST", 160)
CENTER_EXCEPTION_DIST = C("CENTER_EXCEPTION_DIST", 55)

# --- T detection based on extracted structural features
T_ENABLE = C("T_ENABLE", True)

# trunk estimation from lower part of ROI
T_TRUNK_BAND_FRAC = C("T_TRUNK_BAND_FRAC", 0.22)
T_MIN_TRUNK_PIXELS = C("T_MIN_TRUNK_PIXELS", 80)

# scan candidates upward along the estimated trunk
T_SCAN_START_FRAC = C("T_SCAN_START_FRAC", 0.08)
T_SCAN_END_FRAC = C("T_SCAN_END_FRAC", 0.78)
T_SCAN_STEPS = C("T_SCAN_STEPS", 18)

# branch / trunk probing
T_BRANCH_HALF_LEN = C("T_BRANCH_HALF_LEN", 95)
T_BRANCH_THICKNESS = C("T_BRANCH_THICKNESS", 2)
T_BRANCH_MIN_LEN = C("T_BRANCH_MIN_LEN", 24)

T_TRUNK_BACK_LEN = C("T_TRUNK_BACK_LEN", 42)
T_TRUNK_FWD_LEN = C("T_TRUNK_FWD_LEN", 28)
T_TRUNK_THICKNESS = C("T_TRUNK_THICKNESS", 2)
T_TRUNK_MIN_BACK = C("T_TRUNK_MIN_BACK", 20)
T_TRUNK_MAX_FWD = C("T_TRUNK_MAX_FWD", 10)

# geometry checks
T_MIN_TOTAL_BAR = C("T_MIN_TOTAL_BAR", 70)
T_SIDE_BALANCE_RATIO = C("T_SIDE_BALANCE_RATIO", 3.0)
T_CENTER_TOL_PX = C("T_CENTER_TOL_PX", 26)

# debug
SHOW_T_DEBUG = C("SHOW_T_DEBUG", True)


# ============================================================
# DATA
# ============================================================
@dataclass
class LineResult:
    found: bool
    error_px: float
    target_x: float
    centers: list  # [(x, y) or None, ...] in ROI coordinates
    slice_ys: list
    roi_start: int
    mask: np.ndarray
    t_detected: bool
    t_center: tuple[int, int] | None
    trunk_dir: tuple[float, float] | None  # unit vector pointing upward in ROI coords
    scan_points: list
    t_score: float


# ============================================================
# BASIC IMAGE OPS
# ============================================================
def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask

    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def _preprocess(frame_roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

    k = _odd(GAUSS_KSIZE)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    thresh_mode = cv2.THRESH_BINARY_INV if USE_BINARY_INV else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, thresh_mode + cv2.THRESH_OTSU)

    k2 = max(1, KERNEL_SIZE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))

    if MORPH_CLOSE_IT > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    if MORPH_OPEN_IT > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_IT)

    mask = _keep_large_components(mask, MIN_COMPONENT_AREA)
    return mask


# ============================================================
# SLICE FEATURES FOR LINE FOLLOWING
# ============================================================
def _slice_y(h: int, frac: float) -> int:
    return int(np.clip(round(frac * (h - 1)), 0, h - 1))


def _band_centroid_x(
    mask: np.ndarray,
    y: int,
    band_thickness: int,
    prev_x: float | None = None,
    max_dist: float | None = None,
) -> tuple[float | None, tuple[int, int] | None]:
    h, w = mask.shape[:2]
    half = max(1, band_thickness // 2)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)

    ys, xs = np.where(mask[y0:y1] > 0)
    if len(xs) == 0:
        return None, None

    xs = xs.astype(np.float32)

    if prev_x is not None and max_dist is not None:
        keep = np.abs(xs - prev_x) <= max_dist
        if np.any(keep):
            xs = xs[keep]
            ys = ys[keep]
        else:
            if abs(prev_x - (w * 0.5)) <= CENTER_EXCEPTION_DIST:
                return None, None

    if len(xs) == 0:
        return None, None

    x_mean = float(np.mean(xs))
    return x_mean, (int(round(x_mean)), y)


# ============================================================
# DIRECTIONAL FEATURE EXTRACTION FOR T
# ============================================================
def _normalize(vx: float, vy: float) -> tuple[float, float] | None:
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return None
    return vx / n, vy / n


def _estimate_trunk_direction(mask: np.ndarray, x_hint: float | None) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Estimate the incoming line direction from the lower band only.
    Returns:
        trunk_dir_up: unit vector pointing upward in ROI coordinates
        anchor: point near lower band center
    """
    h, w = mask.shape[:2]
    y0 = int((1.0 - T_TRUNK_BAND_FRAC) * h)
    band = mask[y0:h, :]

    ys, xs = np.where(band > 0)
    if len(xs) < T_MIN_TRUNK_PIXELS:
        return None, None

    ys = ys + y0
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    if x_hint is not None:
        keep = np.abs(xs - x_hint) <= BOTTOM_SLICE_MAX_DIST
        if np.count_nonzero(keep) >= T_MIN_TRUNK_PIXELS:
            xs = xs[keep]
            ys = ys[keep]

    pts = np.column_stack((xs, ys)).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean[None, :]
    cov = centered.T @ centered / max(len(pts) - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argmax(eigvals)
    vx, vy = eigvecs[:, idx]

    d = _normalize(float(vx), float(vy))
    if d is None:
        return None, None

    # force vector to point upward in image coordinates
    if d[1] > 0:
        d = (-d[0], -d[1])

    return d, (float(mean[0]), float(mean[1]))


def _line_support(
    mask: np.ndarray,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    max_len: int,
    thickness: int,
    allow_initial_gap: int = 1,
    max_total_gap: int = 3,
) -> int:
    """
    Count how far the structure continues from (cx, cy) along direction d.
    We accept a little gap to be robust to noisy masks.
    """
    h, w = mask.shape[:2]
    perp_x, perp_y = -dy, dx

    hits = 0
    gaps = 0
    started = False

    for s in range(1, max_len + 1):
        px = cx + s * dx
        py = cy + s * dy

        ok = False
        for t in range(-thickness, thickness + 1):
            qx = int(round(px + t * perp_x))
            qy = int(round(py + t * perp_y))
            if 0 <= qx < w and 0 <= qy < h and mask[qy, qx] > 0:
                ok = True
                break

        if ok:
            hits = s
            started = True
            gaps = 0
        else:
            gaps += 1
            if not started and gaps > allow_initial_gap:
                break
            if started and gaps > max_total_gap:
                break

    return hits

def _probe_t_candidate(mask: np.ndarray, cx: float, cy: float, trunk_up: tuple[float, float]):
    """
    Features around one candidate center.

    Robust version:
    - incoming trunk below the candidate can be short
    - left/right branches can be unbalanced if the T is not centered
    """
    ux, uy = trunk_up

    # backward = vers le bas de l'image
    bx, by = -ux, -uy

    # directions perpendiculaires
    lx, ly = -uy, ux
    rx, ry = uy, -ux

    trunk_back = _line_support(
        mask, cx, cy, bx, by,
        max_len=T_TRUNK_BACK_LEN,
        thickness=T_TRUNK_THICKNESS,
    )
    trunk_fwd = _line_support(
        mask, cx, cy, ux, uy,
        max_len=T_TRUNK_FWD_LEN,
        thickness=T_TRUNK_THICKNESS,
    )
    left_len = _line_support(
        mask, cx, cy, lx, ly,
        max_len=T_BRANCH_HALF_LEN,
        thickness=T_BRANCH_THICKNESS,
    )
    right_len = _line_support(
        mask, cx, cy, rx, ry,
        max_len=T_BRANCH_HALF_LEN,
        thickness=T_BRANCH_THICKNESS,
    )

    short_side = min(left_len, right_len)
    long_side = max(left_len, right_len)
    bar_total = left_len + right_len
    branch_ratio = long_side / max(1.0, short_side)

    # vraie présence bilatérale, mais pas forcément symétrique
    has_two_sides = (
        short_side >= T_MIN_BRANCH_SHORT and
        long_side >= T_MIN_BRANCH_LONG and
        bar_total >= T_MIN_TOTAL_BAR and
        branch_ratio <= T_MAX_BRANCH_RATIO
    )

    valid = (
        trunk_back >= T_TRUNK_MIN_BACK and
        trunk_fwd <= T_TRUNK_MAX_FWD and
        has_two_sides
    )

    # score moins punitif sur le tronc court
    score = (
        1.4 * short_side +
        0.8 * long_side +
        0.35 * trunk_back -
        1.0 * trunk_fwd
    )

    feats = {
        "trunk_back": trunk_back,
        "trunk_fwd": trunk_fwd,
        "left_len": left_len,
        "right_len": right_len,
        "short_side": short_side,
        "long_side": long_side,
        "bar_total": bar_total,
        "branch_ratio": branch_ratio,
        "valid": valid,
        "score": score,
    }
    return feats
def _detect_t(mask: np.ndarray, bottom_x: float | None):
    if not T_ENABLE or bottom_x is None:
        return False, None, None, [], 0.0

    h, w = mask.shape[:2]
    trunk_dir, anchor = _estimate_trunk_direction(mask, bottom_x)
    if trunk_dir is None or anchor is None:
        return False, None, None, [], 0.0

    ux, uy = trunk_dir
    ax, ay = anchor

    scan_points = []
    best = None
    best_score = -1e9

    start_s = int(T_SCAN_START_FRAC * h)
    end_s = int(T_SCAN_END_FRAC * h)

    for s in np.linspace(start_s, end_s, T_SCAN_STEPS):
        cx = ax + s * ux
        cy = ay + s * uy

        if not (0 <= cx < w and 0 <= cy < h):
            continue

        feats = _probe_t_candidate(mask, cx, cy, trunk_dir)
        scan_points.append((int(round(cx)), int(round(cy)), feats))

        if feats["valid"] and feats["score"] > best_score:
            best_score = feats["score"]
            best = (int(round(cx)), int(round(cy)))

    if best is None:
        return False, None, trunk_dir, scan_points, 0.0

    return True, best, trunk_dir, scan_points, float(best_score)
# ============================================================
# MAIN API
# ============================================================
def detect_line(frame: np.ndarray, angle_deg: float = 0.0) -> LineResult:
    """
    angle_deg kept only to preserve your old API.
    """
    roi_start = int(frame.shape[0] * ROI_START_FRAC)
    frame_roi = frame[roi_start:, :]
    mask = _preprocess(frame_roi)

    h, w = mask.shape[:2]
    y_top = _slice_y(h, SLICE_TOP_FRAC)
    y_mid = _slice_y(h, SLICE_MID_FRAC)
    y_bot = _slice_y(h, SLICE_BOT_FRAC)
    slice_ys = [y_top, y_mid, y_bot]

    x_bot, c_bot = _band_centroid_x(mask, y_bot, BAND_THICKNESS)
    x_mid, c_mid = _band_centroid_x(mask, y_mid, BAND_THICKNESS, prev_x=x_bot, max_dist=BOTTOM_SLICE_MAX_DIST)
    x_top, c_top = _band_centroid_x(mask, y_top, BAND_THICKNESS, prev_x=x_mid if x_mid is not None else x_bot, max_dist=BOTTOM_SLICE_MAX_DIST)

    centers = [c_top, c_mid, c_bot]
    xs = [x_top, x_mid, x_bot]
    ws = [W_TOP, W_MID, W_BOT]

    valid = [(x, ww) for x, ww in zip(xs, ws) if x is not None]
    if valid:
        target_x = sum(x * ww for x, ww in valid) / sum(ww for _, ww in valid)
        found = True
    else:
        target_x = w * 0.5
        found = False

    error_px = float(target_x - (w * 0.5))

    t_detected, t_center, trunk_dir, scan_points, t_score = _detect_t(mask, x_bot)

    return LineResult(
        found=found,
        error_px=error_px,
        target_x=float(target_x),
        centers=centers,
        slice_ys=slice_ys,
        roi_start=roi_start,
        mask=mask,
        t_detected=t_detected,
        t_center=t_center,
        trunk_dir=trunk_dir,
        scan_points=scan_points,
        t_score=t_score,
    )


# ============================================================
# DEBUG DRAWING
# ============================================================
def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def draw_mask_slices(mask: np.ndarray, res: LineResult, band_thickness: int = BAND_THICKNESS) -> np.ndarray:
    out = _ensure_bgr(mask)

    # slices
    for y in res.slice_ys:
        y0 = max(0, y - band_thickness // 2)
        y1 = min(out.shape[0] - 1, y + band_thickness // 2)
        cv2.rectangle(out, (0, y0), (out.shape[1] - 1, y1), (90, 90, 90), 1)

    # centers
    for c in res.centers:
        if c is not None:
            cv2.circle(out, c, 6, (0, 255, 0), -1)

    # target x / center x
    tx = int(round(res.target_x))
    cx = out.shape[1] // 2
    cv2.line(out, (tx, 0), (tx, out.shape[0] - 1), (255, 255, 0), 1)
    cv2.line(out, (cx, 0), (cx, out.shape[0] - 1), (255, 0, 0), 1)

    # T scan points
    if SHOW_T_DEBUG:
        for x, y, feats in res.scan_points:
            col = (0, 120, 255) if feats["valid"] else (80, 80, 180)
            cv2.circle(out, (x, y), 2, col, -1)

    if res.t_center is not None:
        cv2.circle(out, res.t_center, 10, (0, 0, 255), 2)
        cv2.putText(
            out,
            f"T score={res.t_score:.1f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return out

def draw_debug_slices(frame: np.ndarray, res: LineResult, band_thickness: int = BAND_THICKNESS) -> np.ndarray:
    """
    Old-style debug:
    - works directly on the full frame
    - shows ROI box
    - draws slices only inside ROI
    """
    out = frame.copy()
    h_full, w_full = out.shape[:2]

    roi_y0 = res.roi_start
    roi_h = h_full - roi_y0

    # ROI rectangle
    cv2.rectangle(out, (0, roi_y0), (w_full - 1, h_full - 1), (120, 120, 120), 1)

    # slice bands in full-frame coordinates
    for y in res.slice_ys:
        yy = roi_y0 + y
        y0 = max(0, yy - band_thickness // 2)
        y1 = min(h_full - 1, yy + band_thickness // 2)
        cv2.rectangle(out, (0, y0), (w_full - 1, y1), (0, 255, 255), 1)

    # detected centers
    names = ["top", "mid", "bot"]
    for name, c in zip(names, res.centers):
        if c is not None:
            cx, cy = c
            cyf = roi_y0 + cy
            cv2.circle(out, (cx, cyf), 6, (0, 255, 0), -1)
            cv2.putText(
                out, name, (cx + 8, cyf - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
            )

    # target x and image center
    tx = int(round(res.target_x))
    icx = w_full // 2
    cv2.line(out, (tx, roi_y0), (tx, h_full - 1), (255, 255, 0), 2)
    cv2.line(out, (icx, roi_y0), (icx, h_full - 1), (255, 0, 0), 1)

    # scan points for T in ROI coords -> full coords
    if SHOW_T_DEBUG:
        for x, y, feats in res.scan_points:
            yy = roi_y0 + y
            col = (0, 165, 255) if feats["valid"] else (100, 100, 200)
            cv2.circle(out, (x, yy), 2, col, -1)

    # T center
    if res.t_center is not None:
        txc, tyc = res.t_center
        tyc_full = roi_y0 + tyc
        cv2.circle(out, (txc, tyc_full), 10, (0, 0, 255), 2)
        cv2.putText(
            out,
            "T DETECTED",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"err={res.error_px:.1f}px  found={res.found}  T={res.t_detected}",
        (10, h_full - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out