from __future__ import annotations

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
MIN_FILL_RATIO = 0.18
MAX_FILL_RATIO = 0.75
T_MATCH_THRESHOLD = 0.32


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

    bar_y0 = int(0.12 * size)
    bar_y1 = int(0.36 * size)
    stem_x0 = int(0.40 * size)
    stem_x1 = int(0.60 * size)
    stem_y0 = bar_y1
    stem_y1 = int(0.95 * size)

    cv2.rectangle(img, (int(0.08 * size), bar_y0), (int(0.92 * size), bar_y1), 255, -1)
    cv2.rectangle(img, (stem_x0, stem_y0), (stem_x1, stem_y1), 255, -1)
    return img


def normalize_candidate(mask: np.ndarray, contour: np.ndarray, out_size: int = 160) -> tuple[np.ndarray, tuple[int, int, int, int]]:
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


def compute_t_features(norm_mask: np.ndarray) -> tuple[float, float, float]:
    h, w = norm_mask.shape
    white = norm_mask > 0

    top_band = white[: int(0.35 * h), :]
    bottom_band = white[int(0.40 * h):, :]
    center_band = white[:, int(0.38 * w): int(0.62 * w)]

    top_width_ratio = float(np.mean(np.any(top_band, axis=0)))
    center_height_ratio = float(np.mean(np.any(center_band, axis=1)))
    fill_ratio = float(np.count_nonzero(white)) / float(h * w)

    return top_width_ratio, center_height_ratio, fill_ratio


def detect_t_candidates(mask: np.ndarray) -> tuple[dict | None, list[dict]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template = make_t_template()
    candidates: list[dict] = []
    best: dict | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            continue

        norm_mask, bbox = normalize_candidate(mask, contour, out_size=template.shape[0])
        top_width_ratio, center_height_ratio, fill_ratio = compute_t_features(norm_mask)

        if fill_ratio < MIN_FILL_RATIO or fill_ratio > MAX_FILL_RATIO:
            continue

        shape_score = float(cv2.matchShapes(contour, cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], cv2.CONTOURS_MATCH_I1, 0.0))
        template_diff = float(np.mean(cv2.absdiff(norm_mask, template))) / 255.0

        # Lower is better for both shape_score and template_diff.
        total_score = 0.65 * shape_score + 0.35 * template_diff

        candidate = {
            "contour": contour,
            "bbox": bbox,
            "area": area,
            "shape_score": shape_score,
            "template_diff": template_diff,
            "total_score": total_score,
            "top_width_ratio": top_width_ratio,
            "center_height_ratio": center_height_ratio,
            "fill_ratio": fill_ratio,
            "is_t": (
                total_score <= T_MATCH_THRESHOLD and
                top_width_ratio >= 0.55 and
                center_height_ratio >= 0.45
            ),
            "norm_mask": norm_mask,
        }
        candidates.append(candidate)

        if best is None or candidate["total_score"] < best["total_score"]:
            best = candidate

    return best, candidates


def draw_candidates(frame: np.ndarray, roi_start: int, best: dict | None, candidates: list[dict]) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, roi_start), (out.shape[1] - 1, out.shape[0] - 1), (0, 255, 255), 2)

    for candidate in candidates:
        x, y, w, h = candidate["bbox"]
        color = (0, 255, 0) if candidate["is_t"] else (0, 165, 255)
        cv2.rectangle(out, (x, y + roi_start), (x + w, y + h + roi_start), color, 2)

    if best is not None:
        x, y, w, h = best["bbox"]
        cv2.rectangle(out, (x, y + roi_start), (x + w, y + h + roi_start), (255, 0, 0), 3)
        cv2.putText(
            out,
            f"best score={best['total_score']:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            out,
            f"top={best['top_width_ratio']:.2f} center={best['center_height_ratio']:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            out,
            f"T={'YES' if best['is_t'] else 'NO'}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if best["is_t"] else (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            out,
            "No valid object",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return out


def build_candidate_view(best: dict | None) -> np.ndarray:
    view = np.zeros((220, 220, 3), dtype=np.uint8)
    template = make_t_template()
    view[:160, :160, 1] = template
    cv2.putText(view, "Template", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if best is not None:
        candidate = best["norm_mask"]
        view[:160, 60:220, 2] = candidate[:, :160]
        cv2.putText(view, "Best candidate", (60, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return view


def main():
    leds = LED()
    leds.turn_on_all()
    cam = Camera()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            mask, roi_start = compute_mask(frame)
            best, candidates = detect_t_candidates(mask)

            overlay = draw_candidates(frame, roi_start, best, candidates)
            candidate_view = build_candidate_view(best)

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
