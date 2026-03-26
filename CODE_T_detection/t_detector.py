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
MIN_COMPONENT_AREA = 150

# HSV threshold for black
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0

BLACK_H_HIGH = 180
BLACK_S_HIGH = 170
BLACK_V_HIGH = 110

# trunk extraction
BOTTOM_BAND_FRAC = 0.12
TRUNK_LOOKBACK = 10

# T scan
SCAN_HALF_LEN = 140
MIN_SIDE_LEN = 26
MIN_TRUNK_BELOW = 28
SIDE_BALANCE_MIN = 0.55

# where to search on the trunk
CANDIDATE_START_FRAC = 0.18
CANDIDATE_END_FRAC = 0.82

# stable spatial run on trunk
MIN_VALID_RUN = 7

# temporal hysteresis
T_ENTER_SCORE = 2.25
T_EXIT_SCORE = 1.55
T_ENTER_FRAMES = 3
T_EXIT_FRAMES = 4


# ============================================================
# DATA
# ============================================================

@dataclass
class TDetectionResult:
    t_detected: bool
    t_valid_now: bool
    score: float
    point_xy: tuple[int, int] | None
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

    blur = cv2.GaussianBlur(roi, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)

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
# SKELETON
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
# GEOMETRY
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


def _moving_average(vals: list[float], k: int = 5) -> list[float]:
    if not vals:
        return vals
    half = k // 2
    out = []
    for i in range(len(vals)):
        a = max(0, i - half)
        b = min(len(vals), i + half + 1)
        out.append(sum(vals[a:b]) / max(1, b - a))
    return out


def _evaluate_t_on_trunk(mask: np.ndarray, trunk_path: list[tuple[int, int]]):
    dbg = {"samples": []}

    if len(trunk_path) < max(TRUNK_LOOKBACK + 5, MIN_TRUNK_BELOW):
        dbg["reject"] = "trunk too short"
        return False, 0.0, None, dbg

    n = len(trunk_path)
    i0 = max(TRUNK_LOOKBACK + 2, int(n * CANDIDATE_START_FRAC))
    i1 = min(n - 2, int(n * CANDIDATE_END_FRAC))

    raw = []

    for i in range(i0, i1):
        p = trunk_path[i]
        p_back = trunk_path[max(0, i - TRUNK_LOOKBACK)]

        t = _unit(p[0] - p_back[0], p[1] - p_back[1])
        if t is None:
            raw.append(None)
            continue

        nvec = np.array([-t[1], t[0]], dtype=np.float32)

        left_len = _run_length(mask, p[0], p[1],  nvec[0],  nvec[1], SCAN_HALF_LEN)
        right_len = _run_length(mask, p[0], p[1], -nvec[0], -nvec[1], SCAN_HALF_LEN)

        below_len = 0.0
        for j in range(i, 0, -1):
            dx = trunk_path[j][0] - trunk_path[j - 1][0]
            dy = trunk_path[j][1] - trunk_path[j - 1][1]
            below_len += math.hypot(dx, dy)
            if below_len >= MIN_TRUNK_BELOW:
                break

        balance = min(left_len, right_len) / max(max(left_len, right_len), 1)

        raw.append({
            "idx": i,
            "pt": p,
            "left_len": left_len,
            "right_len": right_len,
            "below_len": below_len,
            "balance": balance,
        })

    if not raw:
        dbg["reject"] = "no samples"
        return False, 0.0, None, dbg

    lefts = [0 if r is None else r["left_len"] for r in raw]
    rights = [0 if r is None else r["right_len"] for r in raw]
    balances = [0 if r is None else r["balance"] for r in raw]

    lefts_s = _moving_average(lefts, 5)
    rights_s = _moving_average(rights, 5)
    balances_s = _moving_average(balances, 5)

    valid_flags = []
    scores = []

    for k, r in enumerate(raw):
        if r is None:
            valid_flags.append(False)
            scores.append(0.0)
            continue

        left_len = lefts_s[k]
        right_len = rights_s[k]
        balance = balances_s[k]
        below_len = r["below_len"]

        valid = (
            left_len >= MIN_SIDE_LEN and
            right_len >= MIN_SIDE_LEN and
            below_len >= MIN_TRUNK_BELOW and
            balance >= SIDE_BALANCE_MIN
        )

        score = (
            min(left_len, 90.0) / 90.0 +
            min(right_len, 90.0) / 90.0 +
            balance
        )

        valid_flags.append(valid)
        scores.append(score)

        dbg["samples"].append({
            "pt": r["pt"],
            "left_len": left_len,
            "right_len": right_len,
            "below_len": below_len,
            "balance": balance,
            "valid": valid,
            "score": score,
        })

    best_run_len = 0
    best_run_score = 0.0
    best_run_mid = None

    run_start = None
    run_score_sum = 0.0

    for k, valid in enumerate(valid_flags + [False]):
        if k < len(valid_flags) and valid:
            if run_start is None:
                run_start = k
                run_score_sum = 0.0
            run_score_sum += scores[k]
        else:
            if run_start is not None:
                run_end = k - 1
                run_len = run_end - run_start + 1
                run_avg_score = run_score_sum / max(run_len, 1)
                run_mid = (run_start + run_end) // 2

                if run_len >= MIN_VALID_RUN:
                    if run_len > best_run_len or (run_len == best_run_len and run_avg_score > best_run_score):
                        best_run_len = run_len
                        best_run_score = run_avg_score
                        best_run_mid = run_mid

                run_start = None
                run_score_sum = 0.0

    if best_run_mid is None:
        dbg["reject"] = "no stable run"
        return False, 0.0, None, dbg

    best_pt = raw[best_run_mid]["pt"]
    dbg["best_run_len"] = best_run_len
    dbg["best_run_score"] = best_run_score
    dbg["best_mid"] = best_pt

    return True, float(best_run_score), best_pt, dbg


# ============================================================
# DETECTOR
# ============================================================

class TDetector:
    def __init__(self):
        self.t_state = False
        self.enter_count = 0
        self.exit_count = 0

    def detect(self, frame: np.ndarray) -> TDetectionResult:
        mask, roi_start = _build_mask(frame)
        skel = _skeletonize(mask)
        seed = _find_bottom_seed(skel)

        debug = {
            "seed": seed,
            "trunk_path": [],
            "geom": {},
        }

        valid_now = False
        score_now = 0.0
        point_now = None

        if seed is not None:
            trunk_path = _trace_trunk_from_bottom(skel, seed)
            debug["trunk_path"] = trunk_path

            ok, score, pt, geom_dbg = _evaluate_t_on_trunk(mask, trunk_path)
            debug["geom"] = geom_dbg

            if ok:
                valid_now = True
                score_now = score
                point_now = pt

        # hysteresis
        if not self.t_state:
            if valid_now and score_now >= T_ENTER_SCORE:
                self.enter_count += 1
            else:
                self.enter_count = 0

            if self.enter_count >= T_ENTER_FRAMES:
                self.t_state = True
                self.exit_count = 0
        else:
            if (not valid_now) or (score_now < T_EXIT_SCORE):
                self.exit_count += 1
            else:
                self.exit_count = 0

            if self.exit_count >= T_EXIT_FRAMES:
                self.t_state = False
                self.enter_count = 0

        return TDetectionResult(
            t_detected=self.t_state,
            t_valid_now=valid_now,
            score=score_now,
            point_xy=point_now,
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
        cv2.circle(out, (p[0], p[1] + res.roi_start), 1, (255, 0, 255), -1)

    if res.point_xy is not None:
        x, y = res.point_xy
        color = (0, 0, 255) if res.t_detected else (0, 165, 255)
        cv2.circle(out, (x, y + res.roi_start), 8, color, 2)

        geom = res.debug.get("geom", {})
        best = geom.get("best_mid")
        cv2.putText(
            out,
            f"score={res.score:.2f}",
            (x + 10, y + res.roi_start - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    state = "T DETECTED" if res.t_detected else ("T candidate" if res.t_valid_now else "No T")
    color = (0, 0, 255) if res.t_detected else ((0, 165, 255) if res.t_valid_now else (180, 180, 180))
    cv2.putText(out, state, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    geom = res.debug.get("geom", {})
    best_run_len = geom.get("best_run_len", 0)
    best_run_score = geom.get("best_run_score", 0.0)
    cv2.putText(
        out,
        f"run={best_run_len} score={best_run_score:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    return out