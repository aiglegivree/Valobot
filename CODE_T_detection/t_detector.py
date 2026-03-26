# t_detector.py
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
MORPH_CLOSE_IT = 2
MORPH_OPEN_IT = 1
KERNEL_SIZE = 3

MIN_COMPONENT_AREA = 120

BOTTOM_BAND_FRAC = 0.12
NODE_MERGE_DIST = 10

# seuils relâchés pour mise au point
MIN_BRANCH_LEN = 10
MIN_TRUNK_LEN_BELOW = 10
MAX_TRUNK_ANGLE_DEG = 55
MIN_BRANCH_ANGLE_DEG = 30
MAX_OPPOSITION_ANGLE_DEG = 55

MAX_NODE_HEIGHT_FRAC = 0.88
MIN_NODE_HEIGHT_FRAC = 0.08

T_CONFIRM_FRAMES = 1
T_MEMORY_FRAMES = 2


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
# LOW LEVEL IMAGE OPS
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
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
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

        # priorité à la remontée
        up = [p for p in neigh if p[1] < cur[1]]
        same = [p for p in neigh if p[1] == cur[1]]
        down = [p for p in neigh if p[1] > cur[1]]

        if up:
            up.sort(key=lambda p: abs(p[0] - cur[0]))
            nxt = up[0]
        elif same:
            same.sort(key=lambda p: abs(p[0] - cur[0]))
            nxt = same[0]
        else:
            down.sort(key=lambda p: abs(p[0] - cur[0]))
            nxt = down[0]

        if nxt in path:
            break

        path.append(nxt)
        prev, cur = cur, nxt

    return path


def _branch_pixels(skel: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(skel > 0)
    out = []
    for x, y in zip(xs, ys):
        deg = len(_neighbors8(x, y, skel))
        if deg >= 3:
            out.append((x, y))
    return out


def _merge_close_points(points: list[tuple[int, int]], dist_thr: float) -> list[tuple[int, int]]:
    groups = []

    for p in points:
        added = False
        for g in groups:
            gx = g["sumx"] / g["n"]
            gy = g["sumy"] / g["n"]
            if (p[0] - gx) ** 2 + (p[1] - gy) ** 2 <= dist_thr ** 2:
                g["sumx"] += p[0]
                g["sumy"] += p[1]
                g["n"] += 1
                added = True
                break

        if not added:
            groups.append({"sumx": p[0], "sumy": p[1], "n": 1})

    merged = []
    for g in groups:
        merged.append((int(round(g["sumx"] / g["n"])), int(round(g["sumy"] / g["n"]))))

    return merged


def _closest_point_on_path(path: list[tuple[int, int]], p: tuple[int, int]) -> tuple[int, float]:
    best_i = -1
    best_d2 = 1e18

    for i, q in enumerate(path):
        d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    return best_i, math.sqrt(best_d2)


def _walk_branch(
    skel: np.ndarray,
    start: tuple[int, int],
    prev: tuple[int, int],
    max_steps: int = 300
) -> list[tuple[int, int]]:
    pts = [start]
    cur = start
    last = prev

    for _ in range(max_steps):
        neigh = _neighbors8(cur[0], cur[1], skel)
        neigh = [p for p in neigh if p != last]

        if not neigh:
            break

        if len(neigh) > 1:
            dx0 = cur[0] - last[0]
            dy0 = cur[1] - last[1]
            n0 = math.hypot(dx0, dy0) + 1e-9

            def score(p):
                dx = p[0] - cur[0]
                dy = p[1] - cur[1]
                n = math.hypot(dx, dy) + 1e-9
                # on préfère continuer "droit"
                return -((dx * dx0 + dy * dy0) / (n * n0))

            neigh.sort(key=score)

        nxt = neigh[0]

        if nxt in pts:
            break

        pts.append(nxt)
        last, cur = cur, nxt

    return pts


def _path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0

    s = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        s += math.hypot(dx, dy)

    return s


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> np.ndarray | None:
    v = np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def _dir_from_path(path: list[tuple[int, int]], from_start: bool = True, nseg: int = 8) -> np.ndarray | None:
    if len(path) < 2:
        return None

    if from_start:
        a = path[0]
        b = path[min(len(path) - 1, nseg)]
    else:
        a = path[max(0, len(path) - 1 - nseg)]
        b = path[-1]

    return _dir_between(a, b)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(c))


# ============================================================
# CORE
# ============================================================

def _evaluate_t_candidate(
    skel: np.ndarray,
    trunk_path: list[tuple[int, int]],
    node: tuple[int, int]
) -> tuple[bool, float, dict]:
    h, _ = skel.shape
    dbg = {}

    node_idx, node_dist = _closest_point_on_path(trunk_path, node)
    dbg["node_idx"] = node_idx
    dbg["node_path_dist"] = node_dist

    if node_idx < 0 or node_dist > 6:
        dbg["reject"] = "node too far from trunk"
        return False, 0.0, dbg

    y_frac = node[1] / max(h, 1)
    dbg["node_y_frac"] = y_frac
    if not (MIN_NODE_HEIGHT_FRAC <= y_frac <= MAX_NODE_HEIGHT_FRAC):
        dbg["reject"] = "node bad height"
        return False, 0.0, dbg

    trunk_below = trunk_path[:node_idx + 1]
    trunk_len_below = _path_length(trunk_below)
    dbg["trunk_len_below"] = trunk_len_below
    if trunk_len_below < MIN_TRUNK_LEN_BELOW:
        dbg["reject"] = "trunk too short below"
        return False, 0.0, dbg

    # direction du tronc juste avant le noeud, du bas vers le haut
    if len(trunk_below) >= 2:
        a = trunk_below[max(0, len(trunk_below) - 8)]
        b = trunk_below[-1]
        trunk_dir = _dir_between(a, b)
    else:
        trunk_dir = None

    if trunk_dir is None:
        dbg["reject"] = "invalid trunk dir"
        return False, 0.0, dbg

    vertical_up = np.array([0.0, -1.0], dtype=np.float32)
    trunk_angle_to_vertical = _angle_deg(trunk_dir, vertical_up)
    dbg["trunk_angle_to_vertical"] = trunk_angle_to_vertical

    if trunk_angle_to_vertical > MAX_TRUNK_ANGLE_DEG:
        dbg["reject"] = "trunk not vertical enough"
        return False, 0.0, dbg

    neigh = _neighbors8(node[0], node[1], skel)
    dbg["node_degree"] = len(neigh)

    if len(neigh) < 3:
        dbg["reject"] = "node degree < 3"
        return False, 0.0, dbg

    # voisin du tronc juste avant le noeud
    if node_idx > 0:
        prev_on_trunk = trunk_path[node_idx - 1]
    else:
        prev_on_trunk = trunk_path[0]

    # voisin le plus proche de la direction venant du tronc
    trunk_neighbor = min(
        neigh,
        key=lambda p: (p[0] - prev_on_trunk[0]) ** 2 + (p[1] - prev_on_trunk[1]) ** 2
    )
    dbg["trunk_neighbor"] = trunk_neighbor

    branch_seeds = [p for p in neigh if p != trunk_neighbor]

    branches = []
    for seed in branch_seeds:
        bpath = _walk_branch(skel, start=seed, prev=node)
        full_path = [node] + bpath
        blen = _path_length(full_path)
        bdir = _dir_from_path(full_path, from_start=True)

        if bdir is None:
            continue

        angle_to_trunk = _angle_deg(trunk_dir, bdir)

        cross_z = trunk_dir[0] * bdir[1] - trunk_dir[1] * bdir[0]
        side = 1 if cross_z > 0 else -1

        branches.append({
            "seed": seed,
            "path": bpath,
            "len": blen,
            "dir": bdir,
            "side": side,
            "angle_to_trunk": angle_to_trunk,
        })

    dbg["branches_raw"] = branches
    if len(branches) < 2:
        dbg["reject"] = "not enough branches"
        return False, 0.0, dbg

    good = []
    for b in branches:
        if b["len"] >= MIN_BRANCH_LEN and b["angle_to_trunk"] >= MIN_BRANCH_ANGLE_DEG:
            good.append(b)

    dbg["branches_good"] = good

    if len(good) < 2:
        dbg["reject"] = "not enough good branches"
        return False, 0.0, dbg

    lefts = [b for b in good if b["side"] > 0]
    rights = [b for b in good if b["side"] < 0]

    if not lefts or not rights:
        dbg["reject"] = "missing one side"
        return False, 0.0, dbg

    left = max(lefts, key=lambda b: b["len"])
    right = max(rights, key=lambda b: b["len"])

    opp_angle = _angle_deg(left["dir"], -right["dir"])
    dbg["left_len"] = left["len"]
    dbg["right_len"] = right["len"]
    dbg["left_angle_to_trunk"] = left["angle_to_trunk"]
    dbg["right_angle_to_trunk"] = right["angle_to_trunk"]
    dbg["opp_angle"] = opp_angle
    dbg["left_path"] = left["path"]
    dbg["right_path"] = right["path"]

    if opp_angle > MAX_OPPOSITION_ANGLE_DEG:
        dbg["reject"] = "branches not opposite enough"
        return False, 0.0, dbg

    score = (
        min(left["len"], 40.0) / 40.0
        + min(right["len"], 40.0) / 40.0
        + max(0.0, 1.0 - opp_angle / max(MAX_OPPOSITION_ANGLE_DEG, 1.0))
        + max(0.0, 1.0 - trunk_angle_to_vertical / max(MAX_TRUNK_ANGLE_DEG, 1.0))
    )

    dbg["reject"] = None
    return True, float(score), dbg


# ============================================================
# TEMPORAL FILTER
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
            "nodes": [],
            "best_candidate": None,
        }

        valid_now = False
        best_score = 0.0
        best_node = None

        if seed is not None:
            trunk_path = _trace_trunk_from_bottom(skel, seed)
            debug["trunk_path"] = trunk_path

            raw_nodes = _branch_pixels(skel)
            nodes = _merge_close_points(raw_nodes, NODE_MERGE_DIST)
            debug["nodes"] = nodes

            for node in nodes:
                ok, score, cand_dbg = _evaluate_t_candidate(skel, trunk_path, node)

                if ok and score > best_score:
                    best_score = score
                    best_node = node
                    debug["best_candidate"] = cand_dbg
                    valid_now = True

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
# DEBUG DRAW
# ============================================================

def draw_t_debug(frame: np.ndarray, res: TDetectionResult) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.line(out, (0, res.roi_start), (w - 1, res.roi_start), (255, 255, 0), 1)

    overlay = np.zeros_like(out)
    if h - res.roi_start > 0:
        sk = cv2.cvtColor(res.skeleton, cv2.COLOR_GRAY2BGR)
        overlay[res.roi_start:] = sk
        out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

    seed = res.debug.get("seed")
    if seed is not None:
        cv2.circle(out, (seed[0], seed[1] + res.roi_start), 4, (0, 255, 255), -1)

    trunk = res.debug.get("trunk_path", [])
    for p in trunk:
        cv2.circle(out, (p[0], p[1] + res.roi_start), 1, (0, 255, 0), -1)

    for p in res.debug.get("nodes", []):
        cv2.circle(out, (p[0], p[1] + res.roi_start), 4, (255, 0, 255), 1)

    best = res.debug.get("best_candidate")
    if best is not None:
        for p in best.get("left_path", []):
            cv2.circle(out, (p[0], p[1] + res.roi_start), 1, (255, 0, 0), -1)
        for p in best.get("right_path", []):
            cv2.circle(out, (p[0], p[1] + res.roi_start), 1, (0, 0, 255), -1)

    if res.node_xy is not None:
        x, y = res.node_xy
        color = (0, 0, 255) if res.t_detected else (0, 165, 255)
        cv2.circle(out, (x, y + res.roi_start), 8, color, 2)
        cv2.putText(
            out,
            f"T score={res.score:.2f}",
            (x + 10, y + res.roi_start - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    state = "T DETECTED" if res.t_detected else ("T candidate" if res.t_valid_now else "No T")
    color = (0, 0, 255) if res.t_detected else ((0, 165, 255) if res.t_valid_now else (180, 180, 180))
    cv2.putText(out, state, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    best = res.debug.get("best_candidate")
    if best is not None:
        txt = best.get("reject", None)
        if txt is not None:
            cv2.putText(out, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)

    return out