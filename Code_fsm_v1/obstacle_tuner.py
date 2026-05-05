from __future__ import annotations

import math

import cv2
import numpy as np

from camera import Camera
from obstacle_vision_config import load_obstacle_settings, save_obstacle_settings


WINDOW_CONTROLS = "Obstacle Controls"
WINDOW_CAMERA = "Obstacle Camera"
WINDOW_EDGES = "Obstacle Edges"
WINDOW_DEBUG = "Obstacle Debug"

FRONT_BOTH_COLOR = (255, 0, 0)
FRONT_RIGHT_ONLY_COLOR = (255, 0, 255)
FRONT_LEFT_ONLY_COLOR = (255, 255, 0)


def nothing(_):
    pass


def create_trackbars(settings: dict[str, int]) -> None:
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 520, 520)

    cv2.createTrackbar("Canny low", WINDOW_CONTROLS, settings["canny_low"], 255, nothing)
    cv2.createTrackbar("Canny high", WINDOW_CONTROLS, settings["canny_high"], 255, nothing)
    cv2.createTrackbar("Contour min len", WINDOW_CONTROLS, min(settings["contour_min_length"], 500), 500, nothing)
    cv2.createTrackbar("Hough threshold", WINDOW_CONTROLS, min(settings["hough_threshold"], 300), 300, nothing)

    cv2.createTrackbar("Left angle low", WINDOW_CONTROLS, settings["left_angle_low"], 180, nothing)
    cv2.createTrackbar("Left angle high", WINDOW_CONTROLS, settings["left_angle_high"], 180, nothing)
    cv2.createTrackbar("Right angle low", WINDOW_CONTROLS, settings["right_angle_low"], 180, nothing)
    cv2.createTrackbar("Right angle high", WINDOW_CONTROLS, settings["right_angle_high"], 180, nothing)
    cv2.createTrackbar("Front angle low", WINDOW_CONTROLS, settings["infront_angle_low"], 180, nothing)
    cv2.createTrackbar("Front angle high", WINDOW_CONTROLS, settings["infront_angle_high"], 180, nothing)


def get_values() -> dict[str, int]:
    values = {
        "canny_low": cv2.getTrackbarPos("Canny low", WINDOW_CONTROLS),
        "canny_high": cv2.getTrackbarPos("Canny high", WINDOW_CONTROLS),
        "contour_min_length": cv2.getTrackbarPos("Contour min len", WINDOW_CONTROLS),
        "hough_threshold": cv2.getTrackbarPos("Hough threshold", WINDOW_CONTROLS),
        "left_angle_low": cv2.getTrackbarPos("Left angle low", WINDOW_CONTROLS),
        "left_angle_high": cv2.getTrackbarPos("Left angle high", WINDOW_CONTROLS),
        "right_angle_low": cv2.getTrackbarPos("Right angle low", WINDOW_CONTROLS),
        "right_angle_high": cv2.getTrackbarPos("Right angle high", WINDOW_CONTROLS),
        "infront_angle_low": cv2.getTrackbarPos("Front angle low", WINDOW_CONTROLS),
        "infront_angle_high": cv2.getTrackbarPos("Front angle high", WINDOW_CONTROLS),
    }

    if values["canny_high"] < values["canny_low"]:
        values["canny_high"] = values["canny_low"]

    for prefix in ("left", "right", "infront"):
        low_key = f"{prefix}_angle_low"
        high_key = f"{prefix}_angle_high"
        if values[high_key] < values[low_key]:
            values[high_key] = values[low_key]

    return values


def hough_line_endpoints(rho: float, theta: float, length: int = 1000) -> tuple[tuple[int, int], tuple[int, int]]:
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + length * (-b))
    y1 = int(y0 + length * a)
    x2 = int(x0 - length * (-b))
    y2 = int(y0 - length * a)
    return (x1, y1), (x2, y2)


def hough_y_at_x(line: tuple[float, float], x: float) -> float | None:
    rho, theta = line
    a = math.cos(theta)
    b = math.sin(theta)
    if abs(b) < 1e-6:
        return None
    return float((rho - a * x) / b)


def front_line_side_presence(
    edges: np.ndarray,
    line: tuple[float, float] | None,
    *,
    samples: int = 80,
    band_radius: int = 2,
    min_hits: int = 4,
) -> str:
    if line is None:
        return "NONE"

    height, width = edges.shape[:2]
    if width <= 0 or height <= 0:
        return "NONE"

    left_hits = 0
    right_hits = 0
    sample_count = max(samples, 2)
    for sample_index in range(sample_count):
        x = int(round((sample_index / (sample_count - 1)) * (width - 1)))
        y_value = hough_y_at_x(line, x)
        if y_value is None:
            continue
        y = int(round(y_value))
        if not (0 <= y < height):
            continue

        x0 = max(0, x - band_radius)
        x1 = min(width, x + band_radius + 1)
        y0 = max(0, y - band_radius)
        y1 = min(height, y + band_radius + 1)
        if np.any(edges[y0:y1, x0:x1] > 0):
            if x < (width / 2.0):
                left_hits += 1
            else:
                right_hits += 1

    left_present = left_hits >= min_hits
    right_present = right_hits >= min_hits
    if left_present and right_present:
        return "BOTH"
    if right_present:
        return "RIGHT_ONLY"
    if left_present:
        return "LEFT_ONLY"
    return "NONE"


def front_line_color(side_presence: str) -> tuple[int, int, int]:
    if side_presence == "RIGHT_ONLY":
        return FRONT_RIGHT_ONLY_COLOR
    if side_presence == "LEFT_ONLY":
        return FRONT_LEFT_ONLY_COLOR
    return FRONT_BOTH_COLOR


def merge_front_side_presence(current: str, new: str) -> str:
    if new == "NONE":
        return current
    if current == "NONE":
        return new
    if current == new:
        return current
    return "BOTH"


def classify_line(theta_deg: float, values: dict[str, int]) -> str | None:
    if values["left_angle_low"] <= theta_deg <= values["left_angle_high"]:
        return "LEFT"
    if values["right_angle_low"] <= theta_deg <= values["right_angle_high"]:
        return "RIGHT"
    if values["infront_angle_low"] <= theta_deg <= values["infront_angle_high"]:
        return "FRONT"
    return None


def detect_obstacle_lines(frame: np.ndarray, values: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, values["canny_low"], values["canny_high"])

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered = np.zeros_like(edges)
    kept_contours = 0
    for contour in contours:
        if cv2.arcLength(contour, closed=False) >= float(values["contour_min_length"]):
            cv2.drawContours(filtered, [contour], -1, 255, 1)
            kept_contours += 1

    raw_lines = cv2.HoughLines(filtered, 1, np.pi / 180, values["hough_threshold"])
    debug = frame.copy()
    counts: dict[str, int | str] = {"LEFT": 0, "RIGHT": 0, "FRONT": 0, "FRONT_SIDE": "NONE"}

    if raw_lines is not None:
        for line in raw_lines:
            rho, theta = line[0]
            angle_deg = math.degrees(theta)
            classified = classify_line(angle_deg, values)
            if classified is None:
                continue
            label = classified
            color = (255, 255, 255)
            if label == "LEFT":
                color = (0, 255, 0)
            elif label == "RIGHT":
                color = (0, 0, 255)
            else:
                front_side = front_line_side_presence(filtered, (rho, theta))
                color = front_line_color(front_side)
                counts["FRONT_SIDE"] = merge_front_side_presence(str(counts["FRONT_SIDE"]), front_side)
            pt1, pt2 = hough_line_endpoints(rho, theta)
            cv2.line(debug, pt1, pt2, color, 2)
            counts[label] = int(counts[label]) + 1

    counts["CONTOURS"] = kept_contours
    counts["LINES"] = 0 if raw_lines is None else int(len(raw_lines))
    return filtered, debug, counts


def main() -> None:
    cam = Camera()
    settings = load_obstacle_settings()

    create_trackbars(settings)
    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_EDGES, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            values = get_values()
            edges, debug_view, counts = detect_obstacle_lines(frame, values)

            cv2.putText(debug_view, f"Canny: {values['canny_low']} / {values['canny_high']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(debug_view, f"Contour min len: {values['contour_min_length']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(debug_view, f"Hough threshold: {values['hough_threshold']}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(debug_view, f"Lines L/R/F: {counts['LEFT']}/{counts['RIGHT']}/{counts['FRONT']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(debug_view, f"Contours kept: {counts['CONTOURS']} | Raw lines: {counts['LINES']}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(debug_view, f"Left angles: {values['left_angle_low']}-{values['left_angle_high']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(debug_view, f"Right angles: {values['right_angle_low']}-{values['right_angle_high']}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
            front_color = front_line_color(str(counts["FRONT_SIDE"]))
            cv2.putText(debug_view, f"Front angles: {values['infront_angle_low']}-{values['infront_angle_high']}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, front_color, 1)
            cv2.putText(debug_view, f"Front side: {counts['FRONT_SIDE']}", (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, front_color, 1)
            cv2.putText(debug_view, "s save config | q or esc quit", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow(WINDOW_CAMERA, frame)
            cv2.imshow(WINDOW_EDGES, edges)
            cv2.imshow(WINDOW_DEBUG, debug_view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("s"):
                save_obstacle_settings(values)
                print("Saved obstacle tuner settings to Code_fsm_v1/obstacle_vision_config.py")

    finally:
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
