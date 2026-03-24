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
    T_ENABLE,
    T_MIN_PIXELS_MAIN,
    T_MIN_EIG_RATIO,
    T_SCAN_START_FRAC,
    T_SCAN_END_FRAC,
    T_SCAN_STEPS,
    T_BRANCH_HALF_LEN,
    T_BRANCH_MIN_LEN,
    T_BRANCH_THICKNESS,
    T_TRUNK_CHECK_LEN,
    T_TRUNK_MIN_HITS,
    T_BAR_MIN_HITS,
    T_CENTER_TOL_PX,
    T_MIN_DIST_FROM_BOTTOM,
)


@dataclass
class VisionResult:
    found: bool
    cx: float | None
    cy: float | None
    error_px: float
    is_t: bool

    mask: np.ndarray
    roi_start: int

    top_xy: tuple[int, int] | None
    mid_xy: tuple[int, int] | None
    bot_xy: tuple[int, int] | None

    main_dir: np.ndarray | None
    t_center: tuple[int, int] | None


# ============================================================
# helpers
# ============================================================

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v.copy()
    return v / n


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))

    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def _extract_slice_point(mask: np.ndarray, y: int, max_dist: float | None = None) -> tuple[int, int] | None:
    h, w = mask.shape
    y = _clamp(y, 0, h - 1)

    xs = np.where(mask[y] > 0)[0]
    if len(xs) == 0:
        return None

    cx = int(np.mean(xs))

    if max_dist is not None and abs(cx - w // 2) > max_dist:
        return None

    return (cx, y)


def _weighted_center(top_xy, mid_xy, bot_xy) -> tuple[float | None, float | None]:
    pts = []
    weights = []

    if top_xy is not None:
        pts.append(top_xy)
        weights.append(W_TOP)

    if mid_xy is not None:
        pts.append(mid_xy)
        weights.append(W_MID)

    if bot_xy is not None:
        pts.append(bot_xy)
        weights.append(W_BOT)

    if not pts:
        return None, None

    pts = np.asarray(pts, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    s = float(np.sum(weights))
    cx = float(np.sum(pts[:, 0] * weights) / s)
    cy = float(np.sum(pts[:, 1] * weights) / s)

    return cx, cy


# ============================================================
# orientation of main line
# ============================================================

def _fit_main_orientation(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < T_MIN_PIXELS_MAIN:
        return None, None, 0.0

    pts = np.column_stack((xs, ys)).astype(np.float32)
    center = np.mean(pts, axis=0)

    pts0 = pts - center
    cov = (pts0.T @ pts0) / max(len(pts0), 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam0 = float(eigvals[0])
    lam1 = float(eigvals[1]) if len(eigvals) > 1 else 1e-6
    eig_ratio = lam0 / max(lam1, 1e-6)

    u = eigvecs[:, 0].astype(np.float32)
    u = _norm(u)

    # orient upward in image
    if u[1] > 0:
        u = -u

    return center, u, eig_ratio


def _sample_line(mask: np.ndarray, p0: np.ndarray, direction: np.ndarray,
                 half_len: int, thickness: int = 1) -> tuple[int, int, int]:
    """
    Count hits along a segment centered at p0 in 'direction'.
    Returns:
        hit_count, first_hit_t, last_hit_t
    """
    h, w = mask.shape
    d = _norm(direction)

    hits = []

    for t in range(-half_len, half_len + 1):
        px = int(round(p0[0] + t * d[0]))
        py = int(round(p0[1] + t * d[1]))

        found = False
        for k in range(-thickness, thickness + 1):
            qx = int(round(px - k * d[1]))
            qy = int(round(py + k * d[0]))
            if 0 <= qx < w and 0 <= qy < h and mask[qy, qx] > 0:
                found = True
                break

        if found:
            hits.append(t)

    if not hits:
        return 0, 0, 0

    return len(hits), min(hits), max(hits)


def _run_length_from_center(mask: np.ndarray, p0: np.ndarray, direction: np.ndarray,
                            max_len: int, thickness: int = 1) -> tuple[int, int]:
    """
    Measure branch length on both sides from center along direction.
    Returns:
        neg_len, pos_len
    """
    h, w = mask.shape
    d = _norm(direction)

    def one_side(sign: int) -> int:
        run = 0
        gaps = 0
        gap_budget = 2

        for t in range(1, max_len + 1):
            px = int(round(p0[0] + sign * t * d[0]))
            py = int(round(p0[1] + sign * t * d[1]))

            found = False
            for k in range(-thickness, thickness + 1):
                qx = int(round(px - k * d[1]))
                qy = int(round(py + k * d[0]))
                if 0 <= qx < w and 0 <= qy < h and mask[qy, qx] > 0:
                    found = True
                    break

            if found:
                run = t
                gaps = 0
            else:
                gaps += 1
                if gaps > gap_budget:
                    break

        return run

    neg_len = one_side(-1)
    pos_len = one_side(+1)
    return neg_len, pos_len


def _point_on_main_line_for_row(mask: np.ndarray, u: np.ndarray, center: np.ndarray, y: int) -> np.ndarray | None:
    """
    Find the point where the oriented main line crosses row y.
    This is much better than taking mean(xs) on the row when the line is tilted.
    """
    if abs(u[1]) < 1e-6:
        return None

    t = (y - center[1]) / u[1]
    x = center[0] + t * u[0]

    h, w = mask.shape
    xi = int(round(x))
    yi = int(round(y))

    if not (0 <= xi < w and 0 <= yi < h):
        return None

    # refine locally on that row
    search_radius = 20
    x0 = max(0, xi - search_radius)
    x1 = min(w - 1, xi + search_radius)

    xs = np.where(mask[yi, x0:x1 + 1] > 0)[0]
    if len(xs) == 0:
        return None

    xr = x0 + int(np.mean(xs))
    return np.array([xr, yi], dtype=np.float32)


# ============================================================
# T detection using line orientation
# ============================================================

def _detect_t_from_orientation(mask: np.ndarray) -> tuple[bool, tuple[int, int] | None, np.ndarray | None]:
    if not T_ENABLE:
        return False, None, None

    main_mask = _largest_component(mask)
    center, u, eig_ratio = _fit_main_orientation(main_mask)

    if center is None or u is None:
        return False, None, None

    if eig_ratio < T_MIN_EIG_RATIO:
        return False, None, u

    n = np.array([-u[1], u[0]], dtype=np.float32)

    h, w = main_mask.shape
    y0 = int(round(T_SCAN_START_FRAC * h))
    y1 = int(round(T_SCAN_END_FRAC * h))
    ys = np.linspace(y0, y1, T_SCAN_STEPS).astype(int)

    best_score = -1.0
    best_center = None

    for y in ys:
        if y > h - T_MIN_DIST_FROM_BOTTOM:
            continue

        p = _point_on_main_line_for_row(main_mask, u, center, y)
        if p is None:
            continue

        # check the trunk exists around the candidate
        hits_trunk, first_t, last_t = _sample_line(
            main_mask, p, u, T_TRUNK_CHECK_LEN, thickness=T_BRANCH_THICKNESS
        )

        if hits_trunk < T_TRUNK_MIN_HITS:
            continue

        # must continue before and after the crossing
        if first_t > -2 or last_t < 2:
            continue

        # check perpendicular branches
        neg_len, pos_len = _run_length_from_center(
            main_mask, p, n, T_BRANCH_HALF_LEN, thickness=T_BRANCH_THICKNESS
        )

        if neg_len < T_BRANCH_MIN_LEN or pos_len < T_BRANCH_MIN_LEN:
            continue

        if abs(neg_len - pos_len) > T_CENTER_TOL_PX:
            continue

        hits_bar, first_b, last_b = _sample_line(
            main_mask, p, n, T_BRANCH_HALF_LEN, thickness=T_BRANCH_THICKNESS
        )

        if hits_bar < T_BAR_MIN_HITS:
            continue

        score = hits_bar + 0.5 * hits_trunk - 0.2 * abs(neg_len - pos_len)

        if score > best_score:
            best_score = score
            best_center = (int(round(p[0])), int(round(p[1])))

    return best_center is not None, best_center, u


# ============================================================
# main pipeline
# ============================================================

def detect_line(frame: np.ndarray, angle_deg: float = 20.0) -> VisionResult:
    h, w = frame.shape[:2]
    roi_start = int(ROI_START_FRAC * h)
    roi = frame[roi_start:].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))

    if MORPH_CLOSE_IT > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=MORPH_CLOSE_IT)

    if MORPH_OPEN_IT > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=MORPH_OPEN_IT)

    mask = _largest_component(mask)

    mh, mw = mask.shape

    y_top = int(SLICE_TOP_FRAC * mh)
    y_mid = int(SLICE_MID_FRAC * mh)
    y_bot = int(SLICE_BOT_FRAC * mh)

    top_xy = _extract_slice_point(mask, y_top)
    mid_xy = _extract_slice_point(mask, y_mid)
    bot_xy = _extract_slice_point(mask, y_bot, max_dist=BOTTOM_SLICE_MAX_DIST)

    if bot_xy is None:
        tmp = _extract_slice_point(mask, y_bot, max_dist=None)
        if tmp is not None and abs(tmp[0] - mw // 2) <= CENTER_EXCEPTION_DIST:
            bot_xy = tmp

    cx, cy = _weighted_center(top_xy, mid_xy, bot_xy)

    if cx is None:
        found = False
        error_px = 0.0
    else:
        found = True
        error_px = float(cx - (mw / 2.0))

    is_t, t_center, main_dir = _detect_t_from_orientation(mask)

    return VisionResult(
        found=found,
        cx=cx,
        cy=cy,
        error_px=error_px,
        is_t=is_t,
        mask=mask,
        roi_start=roi_start,
        top_xy=top_xy,
        mid_xy=mid_xy,
        bot_xy=bot_xy,
        main_dir=main_dir,
        t_center=t_center,
    )


# ============================================================
# debug drawing
# ============================================================

def draw_debug_slices(img: np.ndarray, res: VisionResult, band_thickness: int = BAND_THICKNESS) -> np.ndarray:
    out = img.copy()

    roi_start = res.roi_start
    mh, mw = res.mask.shape

    cv2.rectangle(out, (0, roi_start), (mw - 1, roi_start + mh - 1), (0, 255, 255), 2)

    y_top = roi_start + int(SLICE_TOP_FRAC * mh)
    y_mid = roi_start + int(SLICE_MID_FRAC * mh)
    y_bot = roi_start + int(SLICE_BOT_FRAC * mh)

    cv2.line(out, (0, y_top), (mw - 1, y_top), (255, 0, 0), 1)
    cv2.line(out, (0, y_mid), (mw - 1, y_mid), (255, 0, 0), 1)
    cv2.line(out, (0, y_bot), (mw - 1, y_bot), (255, 0, 0), 1)

    # show band thickness around each slice
    for y in [y_top, y_mid, y_bot]:
        cv2.line(out, (0, y - band_thickness // 2), (mw - 1, y - band_thickness // 2), (80, 80, 80), 1)
        cv2.line(out, (0, y + band_thickness // 2), (mw - 1, y + band_thickness // 2), (80, 80, 80), 1)

    cv2.line(out, (mw // 2, roi_start), (mw // 2, roi_start + mh - 1), (0, 255, 0), 1)

    def draw_pt(pt, color):
        if pt is None:
            return
        x, y = pt
        cv2.circle(out, (x, roi_start + y), 5, color, -1)

    draw_pt(res.top_xy, (0, 0, 255))
    draw_pt(res.mid_xy, (0, 128, 255))
    draw_pt(res.bot_xy, (255, 0, 255))

    if res.cx is not None and res.cy is not None:
        cv2.circle(out, (int(res.cx), roi_start + int(res.cy)), 7, (0, 255, 0), -1)

    if res.main_dir is not None and res.cx is not None and res.cy is not None:
        p = np.array([res.cx, res.cy], dtype=np.float32)
        u = res.main_dir
        n = np.array([-u[1], u[0]], dtype=np.float32)

        p1 = p - 50 * u
        p2 = p + 50 * u
        cv2.line(
            out,
            (int(p1[0]), roi_start + int(p1[1])),
            (int(p2[0]), roi_start + int(p2[1])),
            (255, 255, 0),
            2
        )

        q1 = p - 40 * n
        q2 = p + 40 * n
        cv2.line(
            out,
            (int(q1[0]), roi_start + int(q1[1])),
            (int(q2[0]), roi_start + int(q2[1])),
            (0, 255, 255),
            2
        )

    if res.t_center is not None:
        tx, ty = res.t_center
        cv2.circle(out, (tx, roi_start + ty), 9, (0, 0, 255), 2)
        cv2.putText(out, "T", (tx + 10, roi_start + ty - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    txt = f"found={res.found} err={res.error_px:.1f} T={res.is_t}"
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4)
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def draw_mask_slices(res: VisionResult) -> np.ndarray:
    mask = res.mask
    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    h, w = mask.shape

    y_top = int(SLICE_TOP_FRAC * h)
    y_mid = int(SLICE_MID_FRAC * h)
    y_bot = int(SLICE_BOT_FRAC * h)

    cv2.line(out, (0, y_top), (w - 1, y_top), (255, 0, 0), 1)
    cv2.line(out, (0, y_mid), (w - 1, y_mid), (255, 0, 0), 1)
    cv2.line(out, (0, y_bot), (w - 1, y_bot), (255, 0, 0), 1)
    cv2.line(out, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)

    def draw_pt(pt, color):
        if pt is None:
            return
        cv2.circle(out, pt, 5, color, -1)

    draw_pt(res.top_xy, (0, 0, 255))
    draw_pt(res.mid_xy, (0, 128, 255))
    draw_pt(res.bot_xy, (255, 0, 255))

    if res.cx is not None and res.cy is not None:
        cv2.circle(out, (int(res.cx), int(res.cy)), 7, (0, 255, 0), -1)

    if res.main_dir is not None and res.cx is not None and res.cy is not None:
        p = np.array([res.cx, res.cy], dtype=np.float32)
        u = res.main_dir
        n = np.array([-u[1], u[0]], dtype=np.float32)

        p1 = p - 50 * u
        p2 = p + 50 * u
        cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)

        q1 = p - 40 * n
        q2 = p + 40 * n
        cv2.line(out, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])), (0, 255, 255), 2)

    if res.t_center is not None:
        cv2.circle(out, res.t_center, 9, (0, 0, 255), 2)
        cv2.putText(out, "T", (res.t_center[0] + 10, res.t_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return out