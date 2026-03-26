from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROI_START_FRAC = 0.45

GAUSS_KSIZE = 5
KERNEL_SIZE = 3
MORPH_CLOSE_IT = 2
MORPH_OPEN_IT = 1
MIN_COMPONENT_AREA = 120

BOTTOM_BAND_FRAC = 0.12

TRUNK_LOOKBACK = 10
SCAN_HALF_LEN = 120
SCAN_STEP = 1.0

MIN_SIDE_LEN = 22
MIN_TRUNK_BELOW = 25
SIDE_BALANCE_MIN = 0.45     # min(left,right)/max(left,right)
T_CONFIRM_FRAMES = 2
T_MEMORY_FRAMES = 3

CANDIDATE_START_FRAC = 0.15
CANDIDATE_END_FRAC = 0.85


# ============================================================
# DATA
# ============================================================

@dataclass
class TDetectionResult:
    t_detected: bool
    t_valid_now: bool
    score: float
    node_xy: tuple[int, int] | None
    roi_start: int
    mask: np.ndarray
    skeleton: np.ndarray
    debug: dict


# ============================================================
# IMAGE
# ============================================================

def _build_mask(frame: np.ndarray) -> tuple[np.ndarray, int]:
    h, _ = frame.shape[:2]
    roi_start = int(h * ROI_START_FRAC)
    roi = frame[roi_start:]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_IT)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255

    return clean, roi_start


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    img = binary.copy()
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    skel[skel > 0] = 255
    return skel


# ============================================================
# SKELETON HELPERS
# ============================================================

_NEI8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1)
]


def _neighbors8(x: int, y: int, skel: np.ndarray) -> list[tuple[int, int]]:
    h, w = skel.shape
    out = []
    for dx, dy in _NEI8:
        xx, yy = x + dx, y + dy
        if 0 <= xx < w and 0 <= yy < h and skel[yy, xx] != 0:
            out.append((xx, yy))
    return out


def _find_bottom_seed(skel: np.ndarray) -> tuple[int, int] | None:
    h, w = skel.shape
    y0 = int(h * (1.0 - BOTTOM_BAND_FRAC))
    pts = np.column_stack(np.where(skel[y0:] > 0))
    if len(pts) == 0:
        return None

    ys = pts[:, 0] + y0
    xs = pts[:, 1]

    max_y = ys.max()
    idxs = np.where(ys == max_y)[0]
    cx = w / 2.0
    best_i = idxs[np.argmin(np.abs(xs[idxs] - cx))]
    return int(xs[best_i]), int(ys[best_i])


def _trace_trunk_from_bottom(skel: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]]:
    path = [seed]
    prev = None
    cur = seed

    for _ in range(5000):
        neigh = _neighbors8(cur[0], cur[1], skel)
        if prev is not None:
            neigh = [p for p in neigh if p != prev]
        if not neigh:
            break

        up = [p for p in neigh if p[1] < cur[1]]
        same = [p for p in neigh if p[1] == cur[1]]
        down = [p for p in neigh if p[1] > cur[1]]

        if up:
            up.sort(key=lambda p: abs(p[0] - cur[0]))
            nxt = up[0]
        elif same:
            same.sort(key=lambda p: abs(p[0] - cur[0]))
            nxt = same[0]
        elif down:
            down.sort(key=lambda p: abs(p[0] - cur[0]))
            nxt = down[0]
        else:
            break

        if nxt in path:
            break

        path.append(nxt)
        prev, cur = cur, nxt

    return path


# ============================================================
# GEOMETRIC T TEST
# ============================================================

def _unit(vx: float, vy: float):
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return None
    return np.array([vx / n, vy / n], dtype=np.float32)


def _sample_binary(mask: np.ndarray, x: float, y: float) -> int:
    h, w = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        return 1 if mask[yi, xi] > 0 else 0
    return 0


def _run_length(mask: np.ndarray, x: float, y: float, dx: float, dy: float, max_len: int) -> int:
    cnt = 0
    for k in range(1, max_len + 1):
        xx = x + dx * k
        yy = y + dy * k
        if _sample_binary(mask, xx, yy):
            cnt += 1
        else:
            break
    return cnt


def _evaluate_t_on_trunk(mask: np.ndarray, trunk_path: list[tuple[int, int]]):
    dbg = {"candidates": []}

    if len(trunk_path) < max(TRUNK_LOOKBACK + 5, MIN_TRUNK_BELOW):
        dbg["reject"] = "trunk too short"
        return False, 0.0, None, dbg

    n = len(trunk_path)
    i0 = max(TRUNK_LOOKBACK + 2, int(n * CANDIDATE_START_FRAC))
    i1 = min(n - 2, int(n * CANDIDATE_END_FRAC))

    best_score = 0.0
    best_pt = None
    best_dbg = None

    for i in range(i0, i1):
        p = trunk_path[i]
        p_back = trunk_path[max(0, i - TRUNK_LOOKBACK)]

        t = _unit(p[0] - p_back[0], p[1] - p_back[1])
        if t is None:
            continue

        # normale au tronc
        nvec = np.array([-t[1], t[0]], dtype=np.float32)

        left_len = _run_length(mask, p[0], p[1],  nvec[0],  nvec[1], SCAN_HALF_LEN)
        right_len = _run_length(mask, p[0], p[1], -nvec[0], -nvec[1], SCAN_HALF_LEN)

        below_len = 0
        for j in range(i, -1, -1):
            below_len += math.hypot(
                trunk_path[j][0] - trunk_path[max(0, j - 1)][0],
                trunk_path[j][1] - trunk_path[max(0, j - 1)][1],
            )
            if below_len >= MIN_TRUNK_BELOW:
                break

        balance = min(left_len, right_len) / max(max(left_len, right_len), 1)

        cand = {
            "idx": i,
            "pt": p,
            "left_len": left_len,
            "right_len": right_len,
            "below_len": below_len,
            "balance": balance,
        }
        dbg["candidates"].append(cand)

        if left_len < MIN_SIDE_LEN:
            continue
        if right_len < MIN_SIDE_LEN:
            continue
        if below_len < MIN_TRUNK_BELOW:
            continue
        if balance < SIDE_BALANCE_MIN:
            continue

        score = (
            min(left_len, 80) / 80.0
            + min(right_len, 80) / 80.0
            + balance
        )

        if score > best_score:
            best_score = score
            best_pt = p
            best_dbg = cand

    if best_pt is None:
        dbg["reject"] = "no valid trunk candidate"
        return False, 0.0, None, dbg

    dbg["best"] = best_dbg
    return True, float(best_score), best_pt, dbg


# ============================================================
# DETECTOR
# ============================================================

class TDetector:
    def __init__(self):
        self.confirm_count = 0
        self.memory_count = 0

    def detect(self, frame: np.ndarray) -> TDetectionResult:
        mask, roi_start = _build_mask(frame)
        skel = _skeletonize(mask)

        seed = _find_bottom_seed(skel)

        debug = {
            "seed": seed,
            "trunk_path": [],
        }

        valid_now = False
        best_score = 0.0
        best_node = None

        if seed is not None:
            trunk_path = _trace_trunk_from_bottom(skel, seed)
            debug["trunk_path"] = trunk_path

            ok, score, pt, dbg2 = _evaluate_t_on_trunk(mask, trunk_path)
            debug["geom"] = dbg2

            if ok:
                valid_now = True
                best_score = score
                best_node = pt

        if valid_now:
            self.confirm_count += 1
            self.memory_count = T_MEMORY_FRAMES
        else:
            self.confirm_count = 0
            self.memory_count = max(0, self.memory_count - 1)

        t_detected = (self.confirm_count >= T_CONFIRM_FRAMES) or (self.memory_count > 0)

        return TDetectionResult(
            t_detected=t_detected,
            t_valid_now=valid_now,
            score=best_score,
            node_xy=best_node,
            roi_start=roi_start,
            mask=mask,
            skeleton=skel,
            debug=debug,
        )


# ============================================================
# DRAW
# ============================================================

def draw_t_debug(frame: np.ndarray, res: TDetectionResult) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.line(out, (0, res.roi_start), (w - 1, res.roi_start), (255, 255, 0), 1)

    overlay = np.zeros_like(out)
    sk = cv2.cvtColor(res.skeleton, cv2.COLOR_GRAY2BGR)
    overlay[res.roi_start:] = sk
    out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

    seed = res.debug.get("seed")
    if seed is not None:
        cv2.circle(out, (seed[0], seed[1] + res.roi_start), 4, (0, 255, 255), -1)

    for p in res.debug.get("trunk_path", []):
        cv2.circle(out, (p[0], p[1] + res.roi_start), 1, (0, 255, 0), -1)

    if res.node_xy is not None:
        x, y = res.node_xy
        color = (0, 0, 255) if res.t_detected else (0, 165, 255)
        cv2.circle(out, (x, y + res.roi_start), 8, color, 2)

        geom = res.debug.get("geom", {})
        best = geom.get("best", {})
        txt = f"L={best.get('left_len',0)} R={best.get('right_len',0)} B={best.get('balance',0):.2f}"
        cv2.putText(out, txt, (x + 10, y + res.roi_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    state = "T DETECTED" if res.t_detected else ("T candidate" if res.t_valid_now else "No T")
    color = (0, 0, 255) if res.t_detected else ((0, 165, 255) if res.t_valid_now else (180, 180, 180))
    cv2.putText(out, state, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    return out