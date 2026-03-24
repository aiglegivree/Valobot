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

MIN_COMPONENT_AREA = 150

# skeleton graph / T logic
BOTTOM_BAND_FRAC = 0.12          # zone basse pour choisir le point d'entrée
NODE_MERGE_DIST = 10             # fusion des pixels de branchement proches
MIN_BRANCH_LEN = 18              # longueur mini des branches latérales
MIN_TRUNK_LEN_BELOW = 20         # longueur mini du tronc sous le noeud
MAX_TRUNK_ANGLE_DEG = 35         # tronc plutôt vertical localement
MIN_BRANCH_ANGLE_DEG = 45        # branches assez latérales
MAX_OPPOSITION_ANGLE_DEG = 35    # branches approximativement opposées
MAX_NODE_HEIGHT_FRAC = 0.82      # le noeud doit rester dans une zone plausible
MIN_NODE_HEIGHT_FRAC = 0.10

# temporal validation
T_CONFIRM_FRAMES = 3
T_MEMORY_FRAMES = 5


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
    h, w = frame.shape[:2]
    roi_start = int(h * ROI_START_FRAC)
    roi = frame[roi_start:]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    # noir -> blanc dans le masque
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=MORPH_OPEN_IT)

    # garde seulement les grosses composantes
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
# SKELETON GRAPH HELPERS
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

    # pts = [row, col] dans la sous-image
    ys = pts[:, 0] + y0
    xs = pts[:, 1]

    # prendre le pixel le plus bas, puis le plus proche du centre
    max_y = ys.max()
    idxs = np.where(ys == max_y)[0]
    cx = w / 2.0
    best_i = idxs[np.argmin(np.abs(xs[idxs] - cx))]
    return int(xs[best_i]), int(ys[best_i])


def _trace_trunk_from_bottom(skel: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Remonte le squelette depuis le bas.
    À chaque étape, favorise la continuité vers le haut.
    """
    path = [seed]
    prev = None
    cur = seed

    for _ in range(5000):
        neigh = _neighbors8(cur[0], cur[1], skel)
        if prev is not None:
            neigh = [p for p in neigh if p != prev]
        if not neigh:
            break

        # on favorise d'abord les voisins plus hauts, puis faible dérive latérale
        neigh.sort(key=lambda p: (p[1], abs(p[0] - cur[0])))
        nxt = None

        # priorité aux points qui montent
        for p in neigh:
            if p[1] <= cur[1]:
                nxt = p
                break

        if nxt is None:
            nxt = neigh[0]

        if nxt in path:
            break

        path.append(nxt)
        prev, cur = cur, nxt

    return path


def _branch_pixels(skel: np.ndarray) -> list[tuple[int, int]]:
    h, w = skel.shape
    out = []
    ys, xs = np.where(skel > 0)
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
            gx, gy = g["sumx"] / g["n"], g["sumy"] / g["n"]
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
    """
    Suit une branche à partir d'un noeud.
    """
    pts = [start]
    cur = start
    last = prev

    for _ in range(max_steps):
        neigh = _neighbors8(cur[0], cur[1], skel)
        neigh = [p for p in neigh if p != last]
        if not neigh:
            break

        if len(neigh) > 1:
            # on choisit le voisin le plus aligné avec la direction courante
            dx0 = cur[0] - last[0]
            dy0 = cur[1] - last[1]
            n0 = math.hypot(dx0, dy0) + 1e-9

            def score(p):
                dx = p[0] - cur[0]
                dy = p[1] - cur[1]
                n = math.hypot(dx, dy) + 1e-9
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


def _dir_from_path(path: list[tuple[int, int]], from_start: bool = True, nseg: int = 8) -> np.ndarray | None:
    if len(path) < 2:
        return None

    if from_start:
        a = path[0]
        b = path[min(len(path) - 1, nseg)]
    else:
        a = path[max(0, len(path) - 1 - nseg)]
        b = path[-1]

    v = np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(c))


# ============================================================
# CORE T DETECTION
# ============================================================

def _evaluate_t_candidate(
    skel: np.ndarray,
    trunk_path: list[tuple[int, int]],
    node: tuple[int, int]
) -> tuple[bool, float, dict]:
    h, w = skel.shape
    dbg = {}

    node_idx, node_dist = _closest_point_on_path(trunk_path, node)
    dbg["node_path_dist"] = node_dist
    dbg["node_idx"] = node_idx

    if node_idx < 0 or node_dist > 6:
        return False, 0.0, dbg

    y_frac = node[1] / max(h, 1)
    dbg["node_y_frac"] = y_frac
    if not (MIN_NODE_HEIGHT_FRAC <= y_frac <= MAX_NODE_HEIGHT_FRAC):
        return False, 0.0, dbg

    # tronc sous le noeud
    trunk_below = trunk_path[:node_idx + 1]
    trunk_len_below = _path_length(trunk_below)
    dbg["trunk_len_below"] = trunk_len_below
    if trunk_len_below < MIN_TRUNK_LEN_BELOW:
        return False, 0.0, dbg

    trunk_dir = _dir_from_path(trunk_below[::-1], from_start=True)
    if trunk_dir is None:
        return False, 0.0, dbg

    # le tronc doit être plutôt vertical localement
    vertical_up = np.array([0.0, -1.0], dtype=np.float32)
    trunk_angle_to_vertical = _angle_deg(trunk_dir, vertical_up)
    dbg["trunk_angle_to_vertical"] = trunk_angle_to_vertical
    if trunk_angle_to_vertical > MAX_TRUNK_ANGLE_DEG:
        return False, 0.0, dbg

    neigh = _neighbors8(node[0], node[1], skel)
    if len(neigh) < 3:
        return False, 0.0, dbg

    # le voisin le plus proche du tronc aval est la continuité du tronc
    # les autres sont des candidats branches
    path_prev = trunk_path[max(0, node_idx - 1)]
    neigh_sorted = sorted(
        neigh,
        key=lambda p: (p[0] - path_prev[0]) ** 2 + (p[1] - path_prev[1]) ** 2
    )

    trunk_neighbor = neigh_sorted[0]
    branch_seeds = [p for p in neigh if p != trunk_neighbor]

    branches = []
    for seed in branch_seeds:
        b = _walk_branch(skel, start=seed, prev=node)
        blen = _path_length([node] + b)
        bdir = _dir_from_path([node] + b, from_start=True)
        if bdir is None:
            continue

        side = np.sign(np.cross(np.array([trunk_dir[0], trunk_dir[1], 0.0]),
                                np.array([bdir[0], bdir[1], 0.0]))[2])

        angle_to_trunk = _angle_deg(trunk_dir, bdir)

        branches.append({
            "seed": seed,
            "path": b,
            "len": blen,
            "dir": bdir,
            "side": side,
            "angle_to_trunk": angle_to_trunk,
        })

    dbg["n_branches"] = len(branches)
    if len(branches) < 2:
        return False, 0.0, dbg

    # filtre : branches assez longues et assez latérales
    good = []
    for b in branches:
        # proche de perpendiculaire -> angle autour de 90°
        if b["len"] >= MIN_BRANCH_LEN and b["angle_to_trunk"] >= MIN_BRANCH_ANGLE_DEG:
            good.append(b)

    dbg["n_good_branches"] = len(good)
    if len(good) < 2:
        return False, 0.0, dbg

    lefts = [b for b in good if b["side"] > 0]
    rights = [b for b in good if b["side"] < 0]

    if not lefts or not rights:
        return False, 0.0, dbg

    left = max(lefts, key=lambda b: b["len"])
    right = max(rights, key=lambda b: b["len"])

    opp_angle = _angle_deg(left["dir"], -right["dir"])
    dbg["left_len"] = left["len"]
    dbg["right_len"] = right["len"]
    dbg["opp_angle"] = opp_angle
    dbg["left_angle_to_trunk"] = left["angle_to_trunk"]
    dbg["right_angle_to_trunk"] = right["angle_to_trunk"]

    if opp_angle > MAX_OPPOSITION_ANGLE_DEG:
        return False, 0.0, dbg

    # score simple
    score = (
        min(left["len"], 50.0) / 50.0
        + min(right["len"], 50.0) / 50.0
        + max(0.0, 1.0 - opp_angle / max(MAX_OPPOSITION_ANGLE_DEG, 1.0))
        + max(0.0, 1.0 - trunk_angle_to_vertical / max(MAX_TRUNK_ANGLE_DEG, 1.0))
    )

    dbg["left_path"] = left["path"]
    dbg["right_path"] = right["path"]
    dbg["trunk_neighbor"] = trunk_neighbor

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

        # temporal validation
        if valid_now:
            self.confirm_count += 1
            self.memory_count = T_MEMORY_FRAMES
        else:
            self.confirm_count = 0
            self.memory_count = max(0, self.memory_count - 1)

        t_detected = self.confirm_count >= T_CONFIRM_FRAMES or self.memory_count > 0

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

    # ligne ROI
    cv2.line(out, (0, res.roi_start), (w - 1, res.roi_start), (255, 255, 0), 1)

    # masque / squelette remis dans l'image
    skel_bgr = np.zeros_like(out)
    roi_h = h - res.roi_start
    if roi_h > 0:
        sk = cv2.cvtColor(res.skeleton, cv2.COLOR_GRAY2BGR)
        skel_bgr[res.roi_start:] = sk
        out = cv2.addWeighted(out, 1.0, skel_bgr, 0.35, 0)

    # seed
    seed = res.debug.get("seed")
    if seed is not None:
        cv2.circle(out, (seed[0], seed[1] + res.roi_start), 4, (0, 255, 255), -1)

    # trunk
    trunk = res.debug.get("trunk_path", [])
    for p in trunk:
        cv2.circle(out, (p[0], p[1] + res.roi_start), 1, (0, 255, 0), -1)

    # nodes
    for p in res.debug.get("nodes", []):
        cv2.circle(out, (p[0], p[1] + res.roi_start), 4, (255, 0, 255), 1)

    # best node
    if res.node_xy is not None:
        x, y = res.node_xy
        color = (0, 0, 255) if res.t_detected else (0, 165, 255)
        cv2.circle(out, (x, y + res.roi_start), 7, color, 2)
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

    return out