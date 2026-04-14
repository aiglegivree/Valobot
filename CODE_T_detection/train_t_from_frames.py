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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def compute_distance_maps(binary_template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    template_bool = binary_template > 0
    dist_to_foreground = cv2.distanceTransform((~template_bool).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_background = cv2.distanceTransform(template_bool.astype(np.uint8), cv2.DIST_L2, 3)
    return dist_to_foreground.astype(np.float32), dist_to_background.astype(np.float32)


def distance_map_score(candidate_mask: np.ndarray, dist_fg: np.ndarray, dist_bg: np.ndarray) -> float:
    candidate_bool = candidate_mask > 0
    if not np.any(candidate_bool):
        return float("inf")

    fg_error = float(np.mean(dist_fg[candidate_bool])) if np.any(candidate_bool) else 0.0
    bg_mask = ~candidate_bool
    bg_error = float(np.mean(dist_bg[bg_mask])) if np.any(bg_mask) else 0.0
    return 0.5 * fg_error + 0.5 * bg_error


def build_preview(
    samples: list[np.ndarray],
    prototype_u8: np.ndarray,
    binary_template: np.ndarray,
    dist_fg: np.ndarray,
) -> np.ndarray:
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

    proto_bgr = np.zeros((panel_h, tile, 3), dtype=np.uint8)
    proto_bgr[:tile, :, 2] = prototype_u8

    templ_bgr = np.zeros((panel_h, tile, 3), dtype=np.uint8)
    templ_bgr[:tile, :, 0] = binary_template

    dist_vis = cv2.normalize(dist_fg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_bgr = cv2.applyColorMap(dist_vis, cv2.COLORMAP_TURBO)
    dist_panel = np.zeros((panel_h, tile, 3), dtype=np.uint8)
    dist_panel[:tile, :, :] = dist_bgr

    preview = np.hstack([grid, proto_bgr, templ_bgr, dist_panel])
    cv2.putText(preview, "Samples", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(preview, "Avg prototype", (grid.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(preview, "Binary template", (grid.shape[1] + tile + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(preview, "Distance map", (grid.shape[1] + 2 * tile + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return preview


def main():
    image_paths = iter_training_images()
    if not image_paths:
        raise RuntimeError(
            f"No training images found in {TRAIN_ROOT}. "
            "Capture frames first with capture_t_frames.py."
        )

    normalized_samples: list[np.ndarray] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        mask = preprocess_to_mask(image)
        contour = largest_contour(mask)
        if contour is None:
            continue

        normalized_samples.append(normalize_contour_mask(mask, contour, out_size=IMAGE_SIZE))

    if not normalized_samples:
        raise RuntimeError("Images found, but none produced a usable T mask.")

    sample_stack = np.stack(normalized_samples, axis=0).astype(np.float32) / 255.0
    prototype_float = np.mean(sample_stack, axis=0)
    prototype_u8 = np.clip(np.round(255.0 * prototype_float), 0, 255).astype(np.uint8)
    binary_template = (prototype_float >= 0.50).astype(np.uint8) * 255

    dist_fg, dist_bg = compute_distance_maps(binary_template)
    training_scores = np.array(
        [distance_map_score(sample, dist_fg, dist_bg) for sample in normalized_samples],
        dtype=np.float32,
    )
    match_threshold = float(np.percentile(training_scores, 95) * 1.15)

    np.savez(
        MODEL_PATH,
        prototype=prototype_float.astype(np.float32),
        binary_template=binary_template.astype(np.uint8),
        dist_fg=dist_fg.astype(np.float32),
        dist_bg=dist_bg.astype(np.float32),
        match_threshold=np.float32(match_threshold),
        image_size=np.int32(IMAGE_SIZE),
        sample_count=np.int32(len(normalized_samples)),
    )

    preview = build_preview(normalized_samples, prototype_u8, binary_template, dist_fg)
    cv2.imwrite(str(MODEL_PREVIEW_PATH), preview)

    print(f"Used {len(normalized_samples)} training images.")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved preview to {MODEL_PREVIEW_PATH}")
    print(f"Distance threshold: {match_threshold:.4f}")


if __name__ == "__main__":
    main()
