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
MODEL_PATH = Path(__file__).with_name("t_object_model.npz")
DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_FD_COUNT = 32
DEFAULT_RESAMPLE_POINTS = 256


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


def resample_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        raise ValueError("Contour too short")

    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        raise ValueError("Degenerate contour")

    targets = np.linspace(0.0, total, num_points, endpoint=False)
    out = np.zeros((num_points, 2), dtype=np.float32)
    seg_idx = 0

    for i, t in enumerate(targets):
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < t:
            seg_idx += 1

        p0 = closed[seg_idx]
        p1 = closed[seg_idx + 1]
        seg_len = max(cumulative[seg_idx + 1] - cumulative[seg_idx], 1e-6)
        alpha = (t - cumulative[seg_idx]) / seg_len
        out[i] = (1.0 - alpha) * p0 + alpha * p1

    return out


def fourier_descriptor(contour: np.ndarray, fd_count: int, resample_points: int) -> np.ndarray:
    pts = resample_contour(contour, resample_points)
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - np.mean(z)

    spectrum = np.fft.fft(z)
    magnitudes = np.abs(spectrum)
    descriptor = magnitudes[1:fd_count + 1].astype(np.float32)
    scale = float(descriptor[0]) if descriptor.size > 0 else 1.0
    scale = max(scale, 1e-6)
    descriptor /= scale
    return descriptor


def load_model() -> dict:
    if MODEL_PATH.exists():
        data = np.load(str(MODEL_PATH))
        return {
            "mean_descriptor": data["mean_descriptor"].astype(np.float32),
            "match_threshold": float(data["match_threshold"]),
            "fd_count": int(data["fd_count"]),
            "resample_points": int(data["resample_points"]),
            "source": "trained",
        }

    return {
        "mean_descriptor": None,
        "match_threshold": DEFAULT_MATCH_THRESHOLD,
        "fd_count": DEFAULT_FD_COUNT,
        "resample_points": DEFAULT_RESAMPLE_POINTS,
        "source": "default",
    }


def detect_t_candidates(mask: np.ndarray, model: dict) -> tuple[dict | None, list[dict]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[dict] = []
    best: dict | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            continue

        try:
            descriptor = fourier_descriptor(contour, model["fd_count"], model["resample_points"])
        except ValueError:
            continue

        if model["mean_descriptor"] is None:
            score = float(cv2.matchShapes(contour, contour, cv2.CONTOURS_MATCH_I1, 0.0))
            is_t = False
        else:
            score = float(np.linalg.norm(descriptor - model["mean_descriptor"]))
            is_t = score <= model["match_threshold"]

        x, y, w, h = cv2.boundingRect(contour)
        candidate = {
            "contour": contour,
            "bbox": (x, y, w, h),
            "area": area,
            "score": score,
            "is_t": is_t,
            "descriptor": descriptor,
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
        cv2.putText(out, f"score={best['score']:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(
            out,
            f"T={'YES' if best['is_t'] else 'NO'} thr={model['match_threshold']:.4f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if best["is_t"] else (0, 0, 255),
            2,
        )
        cv2.putText(
            out,
            f"model={model['source']} fd={model['fd_count']}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
    else:
        cv2.putText(out, "No valid object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return out


def build_descriptor_view(best: dict | None, model: dict) -> np.ndarray:
    height = 280
    width = 520
    view = np.zeros((height, width, 3), dtype=np.uint8)

    if model["mean_descriptor"] is not None:
        desc = model["mean_descriptor"]
        xs = np.linspace(20, width - 20, num=desc.size).astype(np.int32)
        max_val = max(float(np.max(desc)), 1e-6)
        ys = (height - 40 - (desc / max_val) * (height - 80)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(view, [pts], False, (0, 255, 255), 2)
        cv2.putText(view, "Learned T descriptor", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if best is not None:
        desc = best["descriptor"]
        xs = np.linspace(20, width - 20, num=desc.size).astype(np.int32)
        max_val = max(float(np.max(desc)), 1e-6)
        ys = (height - 40 - (desc / max_val) * (height - 80)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(view, [pts], False, (0, 0, 255), 2)
        cv2.putText(view, "Best candidate descriptor", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
            descriptor_view = build_descriptor_view(best, model)

            cv2.imshow("T object detection", overlay)
            cv2.imshow("T mask", mask)
            cv2.imshow("T descriptor match", descriptor_view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
