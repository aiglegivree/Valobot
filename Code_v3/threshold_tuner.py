from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from camera import Camera
from leds import LED
from config import (
    ROI_START_FRAC,
    BOTTOM_SLICE_MAX_DIST,
    BLACK_S_HIGH,
    BLACK_V_HIGH,
    BROWN_H_LOW,
    BROWN_S_LOW,
    BROWN_V_LOW,
    BROWN_H_HIGH,
    BROWN_S_HIGH,
    BROWN_V_HIGH,
    BROWN_MIN_AREA,
    GAUSS_KSIZE,
    KERNEL_SIZE,
    MORPH_CLOSE_IT,
    CORNER_PADDING,
    CORNER_MIN_CONTOUR_AREA,
    CORNER_APPROX_EPSILON_FRAC,
    CORNER_MAX_ANGLE_DEG,
)


WINDOW_CONTROLS = "Threshold Controls"
WINDOW_CAMERA = "Camera"
WINDOW_MASK = "Mask"
WINDOW_SHAPE = "Shape Debug"
WINDOW_BROWN = "Brown Mask"
CONFIG_PATH = Path(__file__).with_name("config.py")


def nothing(_):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 460, 520)

    cv2.createTrackbar("Black S max", WINDOW_CONTROLS, BLACK_S_HIGH, 255, nothing)
    cv2.createTrackbar("Black V max", WINDOW_CONTROLS, BLACK_V_HIGH, 255, nothing)
    cv2.createTrackbar("Brown H low", WINDOW_CONTROLS, BROWN_H_LOW, 180, nothing)
    cv2.createTrackbar("Brown S low", WINDOW_CONTROLS, BROWN_S_LOW, 255, nothing)
    cv2.createTrackbar("Brown V low", WINDOW_CONTROLS, BROWN_V_LOW, 255, nothing)
    cv2.createTrackbar("Brown H high", WINDOW_CONTROLS, BROWN_H_HIGH, 180, nothing)
    cv2.createTrackbar("Brown S high", WINDOW_CONTROLS, BROWN_S_HIGH, 255, nothing)
    cv2.createTrackbar("Brown V high", WINDOW_CONTROLS, BROWN_V_HIGH, 255, nothing)
    cv2.createTrackbar("Brown min area", WINDOW_CONTROLS, min(BROWN_MIN_AREA, 100000), 100000, nothing)
    cv2.createTrackbar("Blur", WINDOW_CONTROLS, min(GAUSS_KSIZE, 15), 15, nothing)
    cv2.createTrackbar("Close kernel", WINDOW_CONTROLS, min(KERNEL_SIZE, 15), 15, nothing)
    cv2.createTrackbar("Close iter", WINDOW_CONTROLS, min(MORPH_CLOSE_IT, 10), 10, nothing)
    cv2.createTrackbar("Corner min area", WINDOW_CONTROLS, min(CORNER_MIN_CONTOUR_AREA, 100000), 100000, nothing)
    cv2.createTrackbar(
        "Corner max angle",
        WINDOW_CONTROLS,
        min(max(int(round(CORNER_MAX_ANGLE_DEG)), 1), 180),
        180,
        nothing,
    )
    cv2.createTrackbar(
        "Approx x100",
        WINDOW_CONTROLS,
        min(max(int(round(CORNER_APPROX_EPSILON_FRAC * 100)), 0), 20),
        20,
        nothing,
    )


def odd_at_least_one(value: int) -> int:
    value = max(1, value)
    if value % 2 == 0:
        value += 1
    return value


def get_values():
    black_s_max = cv2.getTrackbarPos("Black S max", WINDOW_CONTROLS)
    black_v_max = cv2.getTrackbarPos("Black V max", WINDOW_CONTROLS)
    brown_h_low = cv2.getTrackbarPos("Brown H low", WINDOW_CONTROLS)
    brown_s_low = cv2.getTrackbarPos("Brown S low", WINDOW_CONTROLS)
    brown_v_low = cv2.getTrackbarPos("Brown V low", WINDOW_CONTROLS)
    brown_h_high = cv2.getTrackbarPos("Brown H high", WINDOW_CONTROLS)
    brown_s_high = cv2.getTrackbarPos("Brown S high", WINDOW_CONTROLS)
    brown_v_high = cv2.getTrackbarPos("Brown V high", WINDOW_CONTROLS)
    brown_min_area = cv2.getTrackbarPos("Brown min area", WINDOW_CONTROLS)
    blur = odd_at_least_one(cv2.getTrackbarPos("Blur", WINDOW_CONTROLS))
    close_kernel = odd_at_least_one(cv2.getTrackbarPos("Close kernel", WINDOW_CONTROLS))
    close_iter = max(1, cv2.getTrackbarPos("Close iter", WINDOW_CONTROLS))
    min_area = cv2.getTrackbarPos("Corner min area", WINDOW_CONTROLS)
    corner_max_angle_deg = max(1, cv2.getTrackbarPos("Corner max angle", WINDOW_CONTROLS))
    approx_frac = cv2.getTrackbarPos("Approx x100", WINDOW_CONTROLS) / 100.0

    return {
        "black_s_max": black_s_max,
        "black_v_max": black_v_max,
        "brown_h_low": min(brown_h_low, brown_h_high),
        "brown_s_low": min(brown_s_low, brown_s_high),
        "brown_v_low": min(brown_v_low, brown_v_high),
        "brown_h_high": max(brown_h_low, brown_h_high),
        "brown_s_high": max(brown_s_low, brown_s_high),
        "brown_v_high": max(brown_v_low, brown_v_high),
        "brown_min_area": brown_min_area,
        "blur": blur,
        "close_kernel": close_kernel,
        "close_iter": close_iter,
        "min_area": min_area,
        "corner_max_angle_deg": float(corner_max_angle_deg),
        "approx_frac": approx_frac,
    }


def keep_shape_connected_to_bottom_rect(mask: np.ndarray, x_center: int, max_x_dist: int) -> np.ndarray:
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    pad = min(CORNER_PADDING, max(0, h - 1))
    anchor_bottom = max(1, h - pad)
    x0 = max(0, x_center - max_x_dist)
    x1 = min(w, x_center + max_x_dist + 1)
    bottom_band = labels[max(0, anchor_bottom - 2):anchor_bottom, x0:x1]
    candidate_labels = np.unique(bottom_band)
    candidate_labels = candidate_labels[candidate_labels != 0]

    if candidate_labels.size == 0:
        return np.zeros_like(mask)

    best_label = None
    best_area = -1
    for lab in candidate_labels:
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = lab

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def apply_black_border(mask: np.ndarray) -> np.ndarray:
    if CORNER_PADDING <= 0:
        return mask.copy()

    bordered = mask.copy()
    pad = min(CORNER_PADDING, bordered.shape[0] // 2, bordered.shape[1] // 2)
    if pad <= 0:
        return bordered

    bordered[:pad, :] = 0
    bordered[-pad:, :] = 0
    bordered[:, :pad] = 0
    bordered[:, -pad:] = 0
    return bordered


def compute_shape_mask(frame: np.ndarray, values: dict) -> tuple[np.ndarray, int]:
    h_frame, _ = frame.shape[:2]
    roi_start = int(h_frame * ROI_START_FRAC)
    roi = frame[roi_start:]

    roi_blur = cv2.GaussianBlur(roi, (values["blur"], values["blur"]), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    low = np.array([0, 0, 0], dtype=np.uint8)
    high = np.array([180, values["black_s_max"], values["black_v_max"]], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)

    kernel = np.ones((values["close_kernel"], values["close_kernel"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=values["close_iter"])
    mask = apply_black_border(mask)
    mask = keep_shape_connected_to_bottom_rect(
        mask,
        x_center=mask.shape[1] // 2,
        max_x_dist=BOTTOM_SLICE_MAX_DIST,
    )
    mask = apply_black_border(mask)

    return mask, roi_start


def compute_brown_mask(frame: np.ndarray, values: dict) -> tuple[np.ndarray, int]:
    h_frame, _ = frame.shape[:2]
    roi_start = int(h_frame * ROI_START_FRAC)
    roi = frame[roi_start:]

    roi_blur = cv2.GaussianBlur(roi, (values["blur"], values["blur"]), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    low = np.array(
        [values["brown_h_low"], values["brown_s_low"], values["brown_v_low"]],
        dtype=np.uint8,
    )
    high = np.array(
        [values["brown_h_high"], values["brown_s_high"], values["brown_v_high"]],
        dtype=np.uint8,
    )

    mask = cv2.inRange(hsv, low, high)
    kernel = np.ones((values["close_kernel"], values["close_kernel"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=values["close_iter"])
    return mask, roi_start


def apply_min_area_filter(
    mask: np.ndarray,
    contour: np.ndarray | None,
    area: float,
    min_area: int,
) -> np.ndarray:
    if contour is None:
        return mask
    if area >= min_area:
        return mask
    return np.zeros_like(mask)


def is_near_frame_edge(x: int, y: int, width: int, height: int, margin: int) -> bool:
    return (
        x <= margin or
        y <= margin or
        x >= width - 1 - margin or
        y >= height - 1 - margin
    )


def count_side_border_corners(
    corners: list[tuple[int, int]],
    width: int,
    margin: int,
) -> tuple[int, int]:
    left_count = 0
    right_count = 0

    for x, _ in corners:
        if x <= margin:
            left_count += 1
        if x >= width - 1 - margin:
            right_count += 1

    return left_count, right_count


def extract_corners(mask: np.ndarray, values: dict) -> tuple[list[tuple[int, int]], np.ndarray | None, float]:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return [], None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < values["min_area"]:
        return [], contour, area

    perimeter = cv2.arcLength(contour, True)
    epsilon = values["approx_frac"] * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    corners = []
    approx_points = approx.reshape(-1, 2)
    if len(approx_points) < 3:
        return [], contour, area

    edge_margin = max(1, 2 * CORNER_PADDING)
    mask_h, mask_w = mask.shape

    for i, point in enumerate(approx_points):
        px = int(point[0])
        py = int(point[1])
        x = px - 1
        y = py - 1

        if is_near_frame_edge(x, y, mask_w, mask_h, margin=edge_margin):
            corners.append((x, y))
            continue

        prev_point = approx_points[(i - 1) % len(approx_points)].astype(np.float32)
        curr_point = point.astype(np.float32)
        next_point = approx_points[(i + 1) % len(approx_points)].astype(np.float32)

        v1 = prev_point - curr_point
        v2 = next_point - curr_point
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = float(np.dot(v1, v2) / (norm1 * norm2))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if angle_deg > values["corner_max_angle_deg"]:
            continue

        corners.append((x, y))

    return corners, contour, area


def draw_shape_debug(mask: np.ndarray, contour: np.ndarray | None, corners: list[tuple[int, int]]) -> np.ndarray:
    disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if contour is not None and len(contour) > 0:
        shifted = contour - np.array([[[1, 1]]], dtype=contour.dtype)
        cv2.drawContours(disp, [shifted], -1, (0, 255, 0), 2)

    for x, y in corners:
        if 0 <= x < disp.shape[1] and 0 <= y < disp.shape[0]:
            cv2.circle(disp, (x, y), 5, (255, 0, 0), -1)

    return disp


def extract_all_brown_regions(mask: np.ndarray, min_area: float) -> tuple[float, np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, np.zeros_like(mask)

    area = 0.0
    out = np.zeros_like(mask)
    kept_contours = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area:
            continue
        area += contour_area
        kept_contours.append(contour)

    if kept_contours:
        cv2.drawContours(out, kept_contours, -1, 255, thickness=cv2.FILLED)

    return area, out


def replace_config_value(text: str, key: str, value: str) -> str:
    marker = f"{key} = "
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(marker):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{key} = {value}"
            return "\n".join(lines) + "\n"
    raise ValueError(f"Key {key} not found in config.py")


def save_values_to_config(values: dict):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    updates = {
        "BLACK_S_HIGH": str(values["black_s_max"]),
        "BLACK_V_HIGH": str(values["black_v_max"]),
        "BROWN_H_LOW": str(values["brown_h_low"]),
        "BROWN_S_LOW": str(values["brown_s_low"]),
        "BROWN_V_LOW": str(values["brown_v_low"]),
        "BROWN_H_HIGH": str(values["brown_h_high"]),
        "BROWN_S_HIGH": str(values["brown_s_high"]),
        "BROWN_V_HIGH": str(values["brown_v_high"]),
        "BROWN_MIN_AREA": str(values["brown_min_area"]),
        "GAUSS_KSIZE": str(values["blur"]),
        "KERNEL_SIZE": str(values["close_kernel"]),
        "MORPH_CLOSE_IT": str(values["close_iter"]),
        "CORNER_MIN_CONTOUR_AREA": str(values["min_area"]),
        "CORNER_MAX_ANGLE_DEG": f"{values['corner_max_angle_deg']:.1f}",
        "CORNER_APPROX_EPSILON_FRAC": f"{values['approx_frac']:.2f}",
    }

    for key, value in updates.items():
        text = replace_config_value(text, key, value)

    CONFIG_PATH.write_text(text, encoding="utf-8")


def main():
    leds = LED()
    leds.turn_on_all()
    leds_on = True
    cam = Camera()

    create_trackbars()

    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_SHAPE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_BROWN, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            values = get_values()
            mask, roi_start = compute_shape_mask(frame, values)
            brown_mask, _ = compute_brown_mask(frame, values)
            corners, contour, area = extract_corners(mask, values)
            brown_area, brown_filtered = extract_all_brown_regions(
                brown_mask,
                values["brown_min_area"],
            )
            filtered_mask = apply_min_area_filter(mask, contour, area, values["min_area"])
            corner_count = len(corners)
            edge_margin = max(1, 2 * CORNER_PADDING)
            left_border_corners, right_border_corners = count_side_border_corners(
                corners,
                filtered_mask.shape[1],
                edge_margin,
            )
            brown_detected = brown_area >= values["brown_min_area"]

            camera_view = frame.copy()
            cv2.rectangle(
                camera_view,
                (0, roi_start),
                (camera_view.shape[1] - 1, camera_view.shape[0] - 1),
                (0, 255, 255),
                2,
            )

            shape_view = draw_shape_debug(filtered_mask, contour, corners)

            cv2.putText(
                shape_view,
                f"Corners: {corner_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                shape_view,
                f"Sharp turn: {'YES' if corner_count == 6 else 'NO'}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if corner_count == 6 else (0, 0, 255),
                2,
            )
            cv2.putText(
                shape_view,
                f"Brown: {'YES' if brown_detected else 'NO'}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if brown_detected else (0, 0, 255),
                2,
            )
            cv2.putText(
                shape_view,
                f"LEDs: {'ON' if leds_on else 'OFF'} | l toggle | s save config",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                shape_view,
                f"Corner min area: {values['min_area']}",
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                shape_view,
                f"Corner max angle: {int(values['corner_max_angle_deg'])}",
                (10, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                shape_view,
                f"Area: {int(area)}",
                (10, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                shape_view,
                f"Brown area: {int(brown_area)} / {values['brown_min_area']}",
                (10, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_CAMERA, camera_view)
            cv2.imshow(WINDOW_MASK, filtered_mask)
            cv2.imshow(WINDOW_SHAPE, shape_view)
            cv2.imshow(WINDOW_BROWN, brown_filtered)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("l"):
                leds_on = not leds_on
                if leds_on:
                    leds.turn_on_all()
                else:
                    leds.turn_off_all()
            if key == ord("s"):
                save_values_to_config(values)

    finally:
        try:
            leds.turn_off_all()
        except Exception:
            pass

        try:
            cam.release()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
