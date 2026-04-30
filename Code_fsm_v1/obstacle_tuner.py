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


def classify_line(theta_deg: float, values: dict[str, int]) -> tuple[str, tuple[int, int, int]] | None:
    if values["left_angle_low"] <= theta_deg <= values["left_angle_high"]:
        return "LEFT", (0, 255, 0)
    if values["right_angle_low"] <= theta_deg <= values["right_angle_high"]:
        return "RIGHT", (0, 0, 255)
    if values["infront_angle_low"] <= theta_deg <= values["infront_angle_high"]:
        return "FRONT", (255, 0, 0)
    return None


def detect_obstacle_lines(frame: np.ndarray, values: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
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
    counts = {"LEFT": 0, "RIGHT": 0, "FRONT": 0}

    if raw_lines is not None:
        for line in raw_lines:
            rho, theta = line[0]
            angle_deg = math.degrees(theta)
            classified = classify_line(angle_deg, values)
            if classified is None:
                continue
            label, color = classified
            pt1, pt2 = hough_line_endpoints(rho, theta)
            cv2.line(debug, pt1, pt2, color, 2)
            counts[label] += 1

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
            cv2.putText(debug_view, f"Front angles: {values['infront_angle_low']}-{values['infront_angle_high']}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1)
            cv2.putText(debug_view, "s save config | q or esc quit", (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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
