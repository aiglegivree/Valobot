from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


TRAIN_ROOT = Path(__file__).with_name("t_training_frames")
MODEL_PATH = Path(__file__).with_name("t_object_model.npz")
MODEL_PREVIEW_PATH = Path(__file__).with_name("t_object_model_preview.png")

IMAGE_SIZE = 160
GAUSS_KSIZE = 5
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0
BLACK_H_HIGH = 180
BLACK_S_HIGH = 255
BLACK_V_HIGH = 130
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2
MIN_CONTOUR_AREA = 1200
FD_COUNT = 32
RESAMPLE_POINTS = 256


def iter_training_images() -> list[Path]:
    if not TRAIN_ROOT.exists():
        return []

    roi_images = sorted(TRAIN_ROOT.glob("session_*/roi/*.png"))
    if roi_images:
        return roi_images

    return sorted(TRAIN_ROOT.glob("**/*.png"))


def preprocess_to_mask(image_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image_bgr, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    return mask


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return None
    return contour


def normalize_contour_mask(mask: np.ndarray, contour: np.ndarray, out_size: int = IMAGE_SIZE) -> np.ndarray:
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
    return canvas


def resample_contour(contour: np.ndarray, num_points: int = RESAMPLE_POINTS) -> np.ndarray:
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


def fourier_descriptor(contour: np.ndarray, fd_count: int = FD_COUNT) -> np.ndarray:
    pts = resample_contour(contour, RESAMPLE_POINTS)
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - np.mean(z)

    spectrum = np.fft.fft(z)
    magnitudes = np.abs(spectrum)

    # ignore DC, keep only magnitude for rotation invariance
    descriptor = magnitudes[1:fd_count + 1].astype(np.float32)
    scale = float(descriptor[0]) if descriptor.size > 0 else 1.0
    scale = max(scale, 1e-6)
    descriptor /= scale
    return descriptor


def build_preview(samples: list[np.ndarray], mean_descriptor: np.ndarray) -> np.ndarray:
    tile = IMAGE_SIZE
    cols = min(4, max(1, len(samples)))
    rows = max(1, int(np.ceil(len(samples) / cols)))
    panel_h = rows * tile

    grid = np.zeros((panel_h, cols * tile, 3), dtype=np.uint8)
    for idx, sample in enumerate(samples[: rows * cols]):
        r = idx // cols
        c = idx % cols
        y0 = r * tile
        x0 = c * tile
        grid[y0:y0 + tile, x0:x0 + tile, 1] = sample

    plot = np.zeros((panel_h, 420, 3), dtype=np.uint8)
    if mean_descriptor.size > 0:
        xs = np.linspace(20, plot.shape[1] - 20, num=mean_descriptor.size).astype(np.int32)
        max_val = max(float(np.max(mean_descriptor)), 1e-6)
        ys = (panel_h - 40 - (mean_descriptor / max_val) * max(panel_h - 80, 1)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(plot, [pts], False, (0, 255, 255), 2)

    cv2.putText(plot, "Mean Fourier descriptor", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    preview = np.hstack([grid, plot])
    cv2.putText(preview, "Samples", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return preview


def main():
    image_paths = iter_training_images()
    if not image_paths:
        raise RuntimeError(
            f"No training images found in {TRAIN_ROOT}. "
            "Capture frames first with capture_t_frames.py."
        )

    normalized_samples: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        mask = preprocess_to_mask(image)
        contour = largest_contour(mask)
        if contour is None:
            continue

        try:
            descriptor = fourier_descriptor(contour, FD_COUNT)
        except ValueError:
            continue

        descriptors.append(descriptor)
        normalized_samples.append(normalize_contour_mask(mask, contour, out_size=IMAGE_SIZE))

    if not descriptors:
        raise RuntimeError("Images found, but none produced a usable T contour.")

    descriptor_stack = np.stack(descriptors, axis=0).astype(np.float32)
    mean_descriptor = np.mean(descriptor_stack, axis=0)
    train_distances = np.linalg.norm(descriptor_stack - mean_descriptor[None, :], axis=1)
    match_threshold = float(np.percentile(train_distances, 95) * 1.20 + 1e-6)

    np.savez(
        MODEL_PATH,
        mean_descriptor=mean_descriptor.astype(np.float32),
        match_threshold=np.float32(match_threshold),
        fd_count=np.int32(FD_COUNT),
        resample_points=np.int32(RESAMPLE_POINTS),
        sample_count=np.int32(len(descriptors)),
    )

    preview = build_preview(normalized_samples, mean_descriptor)
    cv2.imwrite(str(MODEL_PREVIEW_PATH), preview)

    print(f"Used {len(descriptors)} training images.")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved preview to {MODEL_PREVIEW_PATH}")
    print(f"Fourier distance threshold: {match_threshold:.4f}")


if __name__ == "__main__":
    main()
