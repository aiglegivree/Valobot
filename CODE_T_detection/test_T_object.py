from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from leds import LED


# Camera
W = 640
H = 480
CAM_ID = 0
USE_PICAMERA2 = True

# Preprocessing
GAUSS_KSIZE = 5
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0
BLACK_H_HIGH = 180
BLACK_S_HIGH = 255
BLACK_V_HIGH = 130
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2

# Detection
ROI_START_FRAC = 0.50
MIN_CONTOUR_AREA = 1800
DEFAULT_MATCH_THRESHOLD = 1.2
MODEL_PATH = Path(__file__).with_name("t_object_model.npz")


class Camera:
    def __init__(self):
        if USE_PICAMERA2:
            from picamera2 import Picamera2

            self.cam = Picamera2()
            self.cam.configure(
                self.cam.create_preview_configuration(
                    main={"size": (W, H), "format": "RGB888"}
                )
            )
            self.cam.start()
            self.type = "picamera2"
        else:
            cap = cv2.VideoCapture(CAM_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            if not cap.isOpened():
                raise RuntimeError("Camera not found")
            self.cam = cap
            self.type = "opencv"

    def read(self):
        if self.type == "picamera2":
            frame = self.cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame

        ret, frame = self.cam.read()
        return ret, frame

    def release(self):
        if self.type == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()


def keep_only_bottom_connected(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    queue: list[tuple[int, int]] = []

    for x in range(w):
        if mask[h - 1, x] == 255:
            visited[h - 1, x] = 1
            queue.append((h - 1, x))

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if visited[ny, nx] == 0 and mask[ny, nx] == 255:
                    visited[ny, nx] = 1
                    queue.append((ny, nx))

    out = np.zeros_like(mask)
    out[visited == 1] = 255
    return out


def compute_mask(frame: np.ndarray) -> tuple[np.ndarray, int]:
    h_frame, _ = frame.shape[:2]
    roi_start = int(h_frame * ROI_START_FRAC)
    roi = frame[roi_start:]

    roi_blur = cv2.GaussianBlur(roi, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = keep_only_bottom_connected(mask)

    return mask, roi_start


def make_t_template(size: int = 160) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (int(0.08 * size), int(0.12 * size)), (int(0.92 * size), int(0.36 * size)), 255, -1)
    cv2.rectangle(img, (int(0.40 * size), int(0.36 * size)), (int(0.60 * size), int(0.95 * size)), 255, -1)
    return img


def compute_distance_maps(binary_template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    template_bool = binary_template > 0
    dist_to_foreground = cv2.distanceTransform((~template_bool).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_background = cv2.distanceTransform(template_bool.astype(np.uint8), cv2.DIST_L2, 3)
    return dist_to_foreground.astype(np.float32), dist_to_background.astype(np.float32)


def load_model() -> dict:
    if MODEL_PATH.exists():
        data = np.load(str(MODEL_PATH))
        return {
            "binary_template": data["binary_template"].astype(np.uint8),
            "dist_fg": data["dist_fg"].astype(np.float32),
            "dist_bg": data["dist_bg"].astype(np.float32),
            "match_threshold": float(data["match_threshold"]),
            "image_size": int(data["image_size"]),
            "source": "trained",
        }

    binary_template = make_t_template()
    dist_fg, dist_bg = compute_distance_maps(binary_template)
    return {
        "binary_template": binary_template,
        "dist_fg": dist_fg,
        "dist_bg": dist_bg,
        "match_threshold": DEFAULT_MATCH_THRESHOLD,
        "image_size": binary_template.shape[0],
        "source": "default",
    }


def normalize_candidate(mask: np.ndarray, contour: np.ndarray, out_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = cv2.boundingRect(contour)
    crop = mask[y:y + h, x:x + w]

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    scale = min((out_size - 8) / max(w, 1), (out_size - 8) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    x0 = (out_size - new_w) // 2
    y0 = (out_size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, (x, y, w, h)


def distance_map_score(candidate_mask: np.ndarray, dist_fg: np.ndarray, dist_bg: np.ndarray) -> float:
    candidate_bool = candidate_mask > 0
    if not np.any(candidate_bool):
        return float("inf")

    fg_error = float(np.mean(dist_fg[candidate_bool])) if np.any(candidate_bool) else 0.0
    bg_mask = ~candidate_bool
    bg_error = float(np.mean(dist_bg[bg_mask])) if np.any(bg_mask) else 0.0
    return 0.5 * fg_error + 0.5 * bg_error


def detect_t_candidates(mask: np.ndarray, model: dict) -> tuple[dict | None, list[dict]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict] = []
    best: dict | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            continue

        norm_mask, bbox = normalize_candidate(mask, contour, out_size=model["image_size"])
        score = distance_map_score(norm_mask, model["dist_fg"], model["dist_bg"])

        candidate = {
            "contour": contour,
            "bbox": bbox,
            "area": area,
            "score": score,
            "is_t": score <= model["match_threshold"],
            "norm_mask": norm_mask,
        }
        candidates.append(candidate)

        if best is None or score < best["score"]:
            best = candidate

    return best, candidates


def draw_candidates(frame: np.ndarray, roi_start: int, best: dict | None, candidates: list[dict], model: dict) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, roi_start), (out.shape[1] - 1, out.shape[0] - 1), (0, 255, 255), 2)

    for candidate in candidates:
        x, y, w, h = candidate["bbox"]
        color = (0, 255, 0) if candidate["is_t"] else (0, 165, 255)
        cv2.rectangle(out, (x, y + roi_start), (x + w, y + h + roi_start), color, 2)

    if best is not None:
        x, y, w, h = best["bbox"]
        cv2.rectangle(out, (x, y + roi_start), (x + w, y + h + roi_start), (255, 0, 0), 3)
        cv2.putText(out, f"score={best['score']:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(
            out,
            f"T={'YES' if best['is_t'] else 'NO'} thr={model['match_threshold']:.3f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if best["is_t"] else (0, 0, 255),
            2,
        )
        cv2.putText(
            out,
            f"model={model['source']}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
    else:
        cv2.putText(out, "No valid object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return out


def build_candidate_view(best: dict | None, model: dict) -> np.ndarray:
    size = model["image_size"]
    binary_template = model["binary_template"]
    dist_vis = cv2.normalize(model["dist_fg"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_bgr = cv2.applyColorMap(dist_vis, cv2.COLORMAP_TURBO)

    view = np.zeros((size + 60, size * 2, 3), dtype=np.uint8)
    view[:size, :size, 1] = binary_template
    view[:size, size:size * 2, :] = dist_bgr
    cv2.putText(view, "Template", (10, size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(view, "Distance map", (size + 10, size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if best is not None:
        candidate_panel = np.zeros((size + 60, size, 3), dtype=np.uint8)
        candidate_panel[:size, :, 2] = best["norm_mask"]
        cv2.putText(candidate_panel, "Best candidate", (10, size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        view = np.hstack([view, candidate_panel])

    return view


def main():
    leds = LED()
    leds.turn_on_all()
    cam = Camera()
    model = load_model()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            mask, roi_start = compute_mask(frame)
            best, candidates = detect_t_candidates(mask, model)

            overlay = draw_candidates(frame, roi_start, best, candidates, model)
            candidate_view = build_candidate_view(best, model)

            cv2.imshow("T object detection", overlay)
            cv2.imshow("T mask", mask)
            cv2.imshow("T object match", candidate_view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
