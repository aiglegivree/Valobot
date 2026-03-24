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
    T_SCAN_OFFSET_PX,
    T_BRANCH_HALF_LEN,
    T_BRANCH_MIN_LEN,
    T_BRANCH_THICKNESS,
    T_TRUNK_CHECK_LEN,
    T_TRUNK_MIN_HITS,
    T_BAR_MIN_HITS,
    T_CENTER_TOL_PX,
    T_MIN_DIST_FROM_BOTTOM,
    T_ANGLE_MIN_DEG,
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
    perp_dir: np.ndarray | None
    t_center: tuple[int, int] | None
    t_candidate: tuple[int, int] | None


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
# orientation from tracked line points
# ============================================================

def _fit_main_direction_from_slices(
    top_xy: tuple[int, int] | None,
    mid_xy: tuple[int, int] | None,
    bot_xy: tuple[int, int] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate trunk direction only from the tracked line points,
    not from the full T component.
    Returns:
        center: np.ndarray([x, y]) or None
        u: unit vector along the trunk or None
    """
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

    if len(pts) < 2:
        return None, None

    pts = np.asarray(pts, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    center = np.sum(pts * weights[:, None], axis=0) / np.sum(weights)

    pts0 = pts - center
    cov = (pts0.T @ pts0) / max(len(pts0), 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    u = _norm(u)

    # force direction to point upward in image coordinates
    if u[1] > 0:
        u = -u

    # reject nearly horizontal main direction
    angle_deg = abs(np.degrees(np.arctan2(u[1], u[0])))
    angle_to_vertical = abs(90.0 - angle_deg)
    if angle_to_vertical > T_ANGLE_MIN_DEG:
        # still keep it, but only if it is not too horizontal
        pass

    return center.astype(np.float32), u


# ============================================================
# line sampling
# ============================================================

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


def _refine_point_on_normal(mask: np.ndarray, p: np.ndarray, normal: np.ndarray, radius: int = 8) -> np.ndarray:
    """
    Small refinement around p, along the normal direction, to center the
    candidate on the visible trunk.
    """
    h, w = mask.shape
    n = _norm(normal)

    pts = []
    for t in range(-radius, radius + 1):
        px = int(round(p[0] + t * n[0]))
        py = int(round(p[1] + t * n[1]))
        if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
            pts.append([px, py])

    if not pts:
        return p

    pts = np.asarray(pts, dtype=np.float32)
    return np.mean(pts, axis=0)


# ============================================================
# T detection using trunk direction from slices
# ============================================================

def _choose_t_candidate(
    top_xy: tuple[int, int] | None,
    mid_xy: tuple[int, int] | None,
    bot_xy: tuple[int, int] | None,
    u: np.ndarray,
) -> np.ndarray | None:
    """
    Candidate point where the T bar is expected.
    Prefer the upper visible part of the trunk.
    """
    if mid_xy is not None:
        p = np.array(mid_xy, dtype=np.float32)
    elif top_xy is not None:
        p = np.array(top_xy, dtype=np.float32)
    elif bot_xy is not None:
        p = np.array(bot_xy, dtype=np.float32)
    else:
        return None

    # move upward along the trunk to search closer to the bar
    p = p + T_SCAN_OFFSET_PX * u
    return p.astype(np.float32)


def _detect_t_from_orientation(
    mask: np.ndarray,
    top_xy: tuple[int, int] | None,
    mid_xy: tuple[int, int] | None,
    bot_xy: tuple[int, int] | None,
) -> tuple[bool, tuple[int, int] | None, tuple[int, int] | None, np.ndarray | None, np.ndarray | None]:
    if not T_ENABLE:
        return False, None, None, None, None

    center, u = _fit_main_direction_from_slices(top_xy, mid_xy, bot_xy)
    if center is None or u is None:
        return False, None, None, None, None

    n = np.array([-u[1], u[0]], dtype=np.float32)

    p = _choose_t_candidate(top_xy, mid_xy, bot_xy, u)
    if p is None:
        return False, None, None, u, n

    h, w = mask.shape
    p = _refine_point_on_normal(mask, p, n, radius=8)

    px = int(round(p[0]))
    py = int(round(p[1]))

    candidate = (px, py)

    if not (0 <= px < w and 0 <= py < h):
        return False, None, candidate, u, n

    if py > h - T_MIN_DIST_FROM_BOTTOM:
        return False, None, candidate, u, n

    # 1) trunk continuity around the candidate
    hits_trunk, first_t, last_t = _sample_line(
        mask, p, u, T_TRUNK_CHECK_LEN, thickness=T_BRANCH_THICKNESS
    )

    if hits_trunk < T_TRUNK_MIN_HITS:
        return False, None, candidate, u, n

    if first_t > -2 or last_t < 2:
        return False, None, candidate, u, n

    # 2) perpendicular branches around the candidate
    neg_len, pos_len = _run_length_from_center(
        mask, p, n, T_BRANCH_HALF_LEN, thickness=T_BRANCH_THICKNESS
    )

    if neg_len < T_BRANCH_MIN_LEN or pos_len < T_BRANCH_MIN_LEN:
        return False, None, candidate, u, n

    if abs(neg_len - pos_len) > T_CENTER_TOL_PX:
        return False, None, candidate, u, n

    hits_bar, first_b, last_b = _sample_line(
        mask, p, n, T_BRANCH_HALF_LEN, thickness=T_BRANCH_THICKNESS
    )

    if hits_bar < T_BAR_MIN_HITS:
        return False, None, candidate, u, n

    t_center = (px, py)
    return True, t_center, candidate, u, n


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

    is_t, t_center, t_candidate, main_dir, perp_dir = _detect_t_from_orientation(
        mask, top_xy, mid_xy, bot_xy
    )

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
        perp_dir=perp_dir,
        t_center=t_center,
        t_candidate=t_candidate,
    )


# ============================================================
# debug drawing
# ============================================================

def draw_debug_slices(img: np.ndarray, res: VisionResult, band_thickness: int = BAND_THICKNESS) -> np.ndarray:
    out = img.copy()

    roi_start = res.roi_start
    mh, mw = res.mask.shape

    # old visuals
    cv2.rectangle(out, (0, roi_start), (mw - 1, roi_start + mh - 1), (0, 255, 255), 2)

    y_top = roi_start + int(SLICE_TOP_FRAC * mh)
    y_mid = roi_start + int(SLICE_MID_FRAC * mh)
    y_bot = roi_start + int(SLICE_BOT_FRAC * mh)

    cv2.line(out, (0, y_top), (mw - 1, y_top), (255, 0, 0), 1)
    cv2.line(out, (0, y_mid), (mw - 1, y_mid), (255, 0, 0), 1)
    cv2.line(out, (0, y_bot), (mw - 1, y_bot), (255, 0, 0), 1)

    for y in [y_top, y_mid, y_bot]:
        cv2.line(out, (0, y - band_thickness // 2), (mw - 1, y - band_thickness // 2), (80, 80, 80), 1)
        cv2.line(out, (0, y + band_thickness // 2), (mw - 1, y + band_thickness // 2), (80, 80, 80), 1)

    cv2.line(out, (mw // 2, roi_start), (mw // 2, roi_start + mh - 1), (0, 255, 0), 1)

    def draw_pt(pt, color, r=5):
        if pt is None:
            return
        x, y = pt
        cv2.circle(out, (x, roi_start + y), r, color, -1)

    draw_pt(res.top_xy, (0, 0, 255))
    draw_pt(res.mid_xy, (0, 128, 255))
    draw_pt(res.bot_xy, (255, 0, 255))

    if res.cx is not None and res.cy is not None:
        cv2.circle(out, (int(res.cx), roi_start + int(res.cy)), 7, (0, 255, 0), -1)

    # new visuals
    if res.main_dir is not None and res.cx is not None and res.cy is not None:
        p = np.array([res.cx, res.cy], dtype=np.float32)
        u = res.main_dir
        p1 = p - 55 * u
        p2 = p + 55 * u
        cv2.line(
            out,
            (int(p1[0]), roi_start + int(p1[1])),
            (int(p2[0]), roi_start + int(p2[1])),
            (255, 255, 0),
            2
        )

    if res.perp_dir is not None and res.t_candidate is not None:
        p = np.array(res.t_candidate, dtype=np.float32)
        n = res.perp_dir
        q1 = p - 55 * n
        q2 = p + 55 * n
        cv2.line(
            out,
            (int(q1[0]), roi_start + int(q1[1])),
            (int(q2[0]), roi_start + int(q2[1])),
            (0, 255, 255),
            2
        )

    if res.t_candidate is not None:
        tx, ty = res.t_candidate
        cv2.circle(out, (tx, roi_start + ty), 8, (0, 165, 255), 2)
        cv2.putText(out, "cand", (tx + 10, roi_start + ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    if res.t_center is not None:
        tx, ty = res.t_center
        cv2.circle(out, (tx, roi_start + ty), 10, (0, 0, 255), 2)
        cv2.putText(out, "T", (tx + 10, roi_start + ty - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    txt = f"found={res.found} err={res.error_px:.1f} T={res.is_t}"
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4)
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def draw_mask_slices(res: VisionResult) -> np.ndarray:
    mask = res.mask
    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    h, w = mask.shape

    # old visuals
    y_top = int(SLICE_TOP_FRAC * h)
    y_mid = int(SLICE_MID_FRAC * h)
    y_bot = int(SLICE_BOT_FRAC * h)

    cv2.line(out, (0, y_top), (w - 1, y_top), (255, 0, 0), 1)
    cv2.line(out, (0, y_mid), (w - 1, y_mid), (255, 0, 0), 1)
    cv2.line(out, (0, y_bot), (w - 1, y_bot), (255, 0, 0), 1)
    cv2.line(out, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)

    def draw_pt(pt, color, r=5):
        if pt is None:
            return
        cv2.circle(out, pt, r, color, -1)

    draw_pt(res.top_xy, (0, 0, 255))
    draw_pt(res.mid_xy, (0, 128, 255))
    draw_pt(res.bot_xy, (255, 0, 255))

    if res.cx is not None and res.cy is not None:
        cv2.circle(out, (int(res.cx), int(res.cy)), 7, (0, 255, 0), -1)

    # new visuals
    if res.main_dir is not None and res.cx is not None and res.cy is not None:
        p = np.array([res.cx, res.cy], dtype=np.float32)
        u = res.main_dir
        p1 = p - 55 * u
        p2 = p + 55 * u
        cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)

    if res.perp_dir is not None and res.t_candidate is not None:
        p = np.array(res.t_candidate, dtype=np.float32)
        n = res.perp_dir
        q1 = p - 55 * n
        q2 = p + 55 * n
        cv2.line(out, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])), (0, 255, 255), 2)

    if res.t_candidate is not None:
        cv2.circle(out, res.t_candidate, 8, (0, 165, 255), 2)
        cv2.putText(out, "cand", (res.t_candidate[0] + 10, res.t_candidate[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    if res.t_center is not None:
        cv2.circle(out, res.t_center, 10, (0, 0, 255), 2)
        cv2.putText(out, "T", (res.t_center[0] + 10, res.t_center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return out