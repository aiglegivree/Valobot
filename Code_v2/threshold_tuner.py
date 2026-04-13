from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from camera import Camera
from leds import LED
from config import ROI_START_FRAC, BOTTOM_SLICE_MAX_DIST


WINDOW_CONTROLS = "Threshold Controls"
WINDOW_CAMERA = "Camera"
WINDOW_MASK = "Mask"
WINDOW_SHAPE = "Shape Debug"
CONFIG_PATH = Path(__file__).with_name("config.py")


def nothing(_):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 460, 520)

    cv2.createTrackbar("Black S max", WINDOW_CONTROLS, 255, 255, nothing)
    cv2.createTrackbar("Black V max", WINDOW_CONTROLS, 130, 255, nothing)
    cv2.createTrackbar("Blur", WINDOW_CONTROLS, 5, 15, nothing)
    cv2.createTrackbar("Close kernel", WINDOW_CONTROLS, 5, 15, nothing)
    cv2.createTrackbar("Close iter", WINDOW_CONTROLS, 2, 10, nothing)
    cv2.createTrackbar("Min area", WINDOW_CONTROLS, 1200, 10000, nothing)
    cv2.createTrackbar("Approx x100", WINDOW_CONTROLS, 2, 20, nothing)


def odd_at_least_one(value: int) -> int:
    value = max(1, value)
    if value % 2 == 0:
        value += 1
    return value


def get_values():
    black_s_max = cv2.getTrackbarPos("Black S max", WINDOW_CONTROLS)
    black_v_max = cv2.getTrackbarPos("Black V max", WINDOW_CONTROLS)
    blur = odd_at_least_one(cv2.getTrackbarPos("Blur", WINDOW_CONTROLS))
    close_kernel = odd_at_least_one(cv2.getTrackbarPos("Close kernel", WINDOW_CONTROLS))
    close_iter = max(1, cv2.getTrackbarPos("Close iter", WINDOW_CONTROLS))
    min_area = cv2.getTrackbarPos("Min area", WINDOW_CONTROLS)
    approx_frac = cv2.getTrackbarPos("Approx x100", WINDOW_CONTROLS) / 100.0

    return {
        "black_s_max": black_s_max,
        "black_v_max": black_v_max,
        "blur": blur,
        "close_kernel": close_kernel,
        "close_iter": close_iter,
        "min_area": min_area,
        "approx_frac": approx_frac,
    }


def keep_only_bottom_connected_in_rect(mask: np.ndarray, x_center: int, max_x_dist: int) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    queue: list[tuple[int, int]] = []

    x0 = max(0, x_center - max_x_dist)
    x1 = min(w, x_center + max_x_dist + 1)

    for x in range(x0, x1):
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
    mask = keep_only_bottom_connected_in_rect(
        mask,
        x_center=mask.shape[1] // 2,
        max_x_dist=BOTTOM_SLICE_MAX_DIST,
    )

    return mask, roi_start


def extract_corners(mask: np.ndarray, values: dict) -> tuple[list[tuple[int, int]], np.ndarray | None]:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return [], None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < values["min_area"]:
        return [], contour

    perimeter = cv2.arcLength(contour, True)
    epsilon = values["approx_frac"] * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    corners = []
    for point in approx.reshape(-1, 2):
        corners.append((int(point[0]) - 1, int(point[1]) - 1))

    return corners, contour


def draw_shape_debug(mask: np.ndarray, contour: np.ndarray | None, corners: list[tuple[int, int]]) -> np.ndarray:
    disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if contour is not None and len(contour) > 0:
        shifted = contour - np.array([[[1, 1]]], dtype=contour.dtype)
        cv2.drawContours(disp, [shifted], -1, (0, 255, 0), 2)

    for x, y in corners:
        if 0 <= x < disp.shape[1] and 0 <= y < disp.shape[0]:
            cv2.circle(disp, (x, y), 5, (255, 0, 0), -1)

    return disp


def print_values(values: dict):
    print("Suggested config values:")
    print(f"BLACK_S_HIGH = {values['black_s_max']}")
    print(f"BLACK_V_HIGH = {values['black_v_max']}")
    print(f"GAUSS_KSIZE = {values['blur']}")
    print(f"KERNEL_SIZE = {values['close_kernel']}")
    print(f"MORPH_CLOSE_IT = {values['close_iter']}")
    print(f"CORNER_MIN_CONTOUR_AREA = {values['min_area']}")
    print(f"CORNER_APPROX_EPSILON_FRAC = {values['approx_frac']:.2f}")


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
        "GAUSS_KSIZE": str(values["blur"]),
        "KERNEL_SIZE": str(values["close_kernel"]),
        "MORPH_CLOSE_IT": str(values["close_iter"]),
        "CORNER_MIN_CONTOUR_AREA": str(values["min_area"]),
        "CORNER_APPROX_EPSILON_FRAC": f"{values['approx_frac']:.2f}",
    }

    for key, value in updates.items():
        text = replace_config_value(text, key, value)

    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(f"Saved current values to {CONFIG_PATH}")


def main():
    leds = LED()
    leds.turn_on_all()
    leds_on = True
    cam = Camera()

    create_trackbars()

    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_SHAPE, cv2.WINDOW_NORMAL)

    last_values = None

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            values = get_values()
            mask, roi_start = compute_shape_mask(frame, values)
            corners, contour = extract_corners(mask, values)
            corner_count = len(corners)

            camera_view = frame.copy()
            cv2.rectangle(
                camera_view,
                (0, roi_start),
                (camera_view.shape[1] - 1, camera_view.shape[0] - 1),
                (0, 255, 255),
                2,
            )

            shape_view = draw_shape_debug(mask, contour, corners)

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
                f"T: {'YES' if corner_count == 8 else 'NO'}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if corner_count == 8 else (0, 0, 255),
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

            cv2.imshow(WINDOW_CAMERA, camera_view)
            cv2.imshow(WINDOW_MASK, mask)
            cv2.imshow(WINDOW_SHAPE, shape_view)

            current_values = tuple(values.items())
            if current_values != last_values:
                print_values(values)
                last_values = current_values

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("p"):
                print_values(values)
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
