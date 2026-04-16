from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import sys
import time


WINDOW_NAME = "Obstacle Canny Test"
CONTROL_WINDOW_NAME = "Obstacle Controls"

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_V2_DIR = ROOT_DIR / "Code_v2"
if str(CODE_V2_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_V2_DIR))

from config import W, H, CAM_ID, USE_PICAMERA2  # noqa: E402
from leds import LED  # noqa: E402
from obstacle_vision_config import (  # noqa: E402
    CONFIG_PATH,
    load_obstacle_settings,
    save_obstacle_settings,
)


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
                raise RuntimeError("Could not open camera")
            self.cam = cap
            self.type = "opencv"

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.type == "picamera2":
            return True, self.cam.capture_array()
        return self.cam.read()

    def release(self) -> None:
        if self.type == "opencv":
            self.cam.release()


def _noop(_: int) -> None:
    pass


def load_settings() -> dict[str, int]:
    return load_obstacle_settings()


def save_settings(settings: dict[str, int]) -> None:
    save_obstacle_settings(settings)


def create_controls(settings: dict[str, int]) -> None:
    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW_NAME, 520, 420)
    cv2.createTrackbar("Canny low", CONTROL_WINDOW_NAME, settings["canny_low"], 255, _noop)
    cv2.createTrackbar("Canny high", CONTROL_WINDOW_NAME, settings["canny_high"], 255, _noop)
    cv2.createTrackbar(
        "Contour min len",
        CONTROL_WINDOW_NAME,
        settings["contour_min_length"],
        500,
        _noop,
    )
    cv2.createTrackbar("Hough thresh", CONTROL_WINDOW_NAME, settings["hough_threshold"], 300, _noop)
    cv2.createTrackbar("Left low", CONTROL_WINDOW_NAME, settings["left_angle_low"], 180, _noop)
    cv2.createTrackbar("Left high", CONTROL_WINDOW_NAME, settings["left_angle_high"], 180, _noop)
    cv2.createTrackbar("Right low", CONTROL_WINDOW_NAME, settings["right_angle_low"], 180, _noop)
    cv2.createTrackbar("Right high", CONTROL_WINDOW_NAME, settings["right_angle_high"], 180, _noop)
    cv2.createTrackbar("Front low", CONTROL_WINDOW_NAME, settings["infront_angle_low"], 180, _noop)
    cv2.createTrackbar("Front high", CONTROL_WINDOW_NAME, settings["infront_angle_high"], 180, _noop)


def get_controls() -> dict[str, int]:
    canny_low = cv2.getTrackbarPos("Canny low", CONTROL_WINDOW_NAME)
    canny_high = cv2.getTrackbarPos("Canny high", CONTROL_WINDOW_NAME)
    contour_min_length = cv2.getTrackbarPos("Contour min len", CONTROL_WINDOW_NAME)
    hough_threshold = max(1, cv2.getTrackbarPos("Hough thresh", CONTROL_WINDOW_NAME))
    left_angle_low = cv2.getTrackbarPos("Left low", CONTROL_WINDOW_NAME)
    left_angle_high = cv2.getTrackbarPos("Left high", CONTROL_WINDOW_NAME)
    right_angle_low = cv2.getTrackbarPos("Right low", CONTROL_WINDOW_NAME)
    right_angle_high = cv2.getTrackbarPos("Right high", CONTROL_WINDOW_NAME)
    infront_angle_low = cv2.getTrackbarPos("Front low", CONTROL_WINDOW_NAME)
    infront_angle_high = cv2.getTrackbarPos("Front high", CONTROL_WINDOW_NAME)

    if canny_high < canny_low:
        canny_high = canny_low
        cv2.setTrackbarPos("Canny high", CONTROL_WINDOW_NAME, canny_high)

    for low_name, high_name, low_value, high_value in [
        ("Left low", "Left high", left_angle_low, left_angle_high),
        ("Right low", "Right high", right_angle_low, right_angle_high),
        ("Front low", "Front high", infront_angle_low, infront_angle_high),
    ]:
        if high_value < low_value:
            high_value = low_value
            cv2.setTrackbarPos(high_name, CONTROL_WINDOW_NAME, high_value)
        if low_name == "Left low":
            left_angle_high = high_value
        elif low_name == "Right low":
            right_angle_high = high_value
        else:
            infront_angle_high = high_value

    return {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "contour_min_length": contour_min_length,
        "hough_threshold": hough_threshold,
        "left_angle_low": left_angle_low,
        "left_angle_high": left_angle_high,
        "right_angle_low": right_angle_low,
        "right_angle_high": right_angle_high,
        "infront_angle_low": infront_angle_low,
        "infront_angle_high": infront_angle_high,
    }


def detect_edges(frame: np.ndarray, canny_low: int, canny_high: int, contour_min_length: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map = cv2.Canny(blur, canny_low, canny_high)

    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered = np.zeros_like(edge_map)
    for contour in contours:
        if cv2.arcLength(contour, closed=False) >= float(contour_min_length):
            cv2.drawContours(filtered, [contour], -1, 255, 1)
    return filtered


def hough_line_endpoints(rho: float, theta: float, length: int = 1000) -> tuple[tuple[int, int], tuple[int, int]]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + length * (-b))
    y1 = int(y0 + length * a)
    x2 = int(x0 - length * (-b))
    y2 = int(y0 - length * a)
    return (x1, y1), (x2, y2)


def detect_hough_lines(edge_map: np.ndarray, hough_threshold: int) -> np.ndarray | None:
    return cv2.HoughLines(edge_map, rho=1, theta=np.pi / 180, threshold=hough_threshold)


def median_line(lines: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not lines:
        return None
    rhos = np.array([line[0] for line in lines], dtype=np.float32)
    thetas = np.array([line[1] for line in lines], dtype=np.float32)
    return float(np.median(rhos)), float(np.median(thetas))


def classify_lines(lines: np.ndarray | None, controls: dict[str, int]) -> dict[str, list[tuple[float, float]]]:
    categories: dict[str, list[tuple[float, float]]] = {
        "right": [],
        "left": [],
        "infront": [],
    }
    if lines is None:
        return categories

    for line in lines:
        rho, theta = line[0]
        angle_deg = int(round(np.degrees(theta)))
        if controls["right_angle_low"] <= angle_deg <= controls["right_angle_high"]:
            categories["right"].append((rho, theta))
        elif controls["left_angle_low"] <= angle_deg <= controls["left_angle_high"]:
            categories["left"].append((rho, theta))
        elif controls["infront_angle_low"] <= angle_deg <= controls["infront_angle_high"]:
            categories["infront"].append((rho, theta))
    return categories


def category_medians(categories: dict[str, list[tuple[float, float]]]) -> dict[str, tuple[float, float] | None]:
    return {name: median_line(lines) for name, lines in categories.items()}


def draw_result(
    frame: np.ndarray,
    edge_map: np.ndarray,
    controls: dict[str, int],
    categories: dict[str, list[tuple[float, float]]],
    medians: dict[str, tuple[float, float] | None],
    leds_on: bool,
) -> np.ndarray:
    overlay = frame.copy()
    for name, color in [
        ("right", (0, 0, 255)),
        ("left", (0, 255, 0)),
        ("infront", (255, 0, 0)),
    ]:
        line = medians[name]
        if line is None:
            continue
        rho, theta = line
        pt1, pt2 = hough_line_endpoints(rho, theta)
        cv2.line(overlay, pt1, pt2, color, 3)

    edge_vis = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((overlay, edge_vis))

    cv2.putText(
        combined,
        f"Canny low/high: {controls['canny_low']}/{controls['canny_high']}",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Contour min len: {controls['contour_min_length']}",
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Hough threshold: {controls['hough_threshold']}",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"LEDs: {'ON' if leds_on else 'OFF'}  Toggle: l  Save: s",
        (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Right {controls['right_angle_low']}-{controls['right_angle_high']}: {len(categories['right'])}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Left {controls['left_angle_low']}-{controls['left_angle_high']}: {len(categories['left'])}",
        (10, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        combined,
        (
            f"Infront {controls['infront_angle_low']}-"
            f"{controls['infront_angle_high']}: {len(categories['infront'])}"
        ),
        (10, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 0),
        1,
    )
    cv2.putText(
        combined,
        (
            f"Edges: {int(np.count_nonzero(edge_map))}  "
            f"Classified lines: {sum(len(items) for items in categories.values())}"
        ),
        (10, 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    return combined


def main() -> None:
    cam = Camera()
    leds = LED()
    leds_on = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    settings = load_settings()
    create_controls(settings)
    save_message_until = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            controls = get_controls()
            edges = detect_edges(
                frame,
                controls["canny_low"],
                controls["canny_high"],
                controls["contour_min_length"],
            )
            lines = detect_hough_lines(
                edges,
                controls["hough_threshold"],
            )
            categories = classify_lines(lines, controls)
            medians = category_medians(categories)
            result = draw_result(frame, edges, controls, categories, medians, leds_on)
            if time.monotonic() < save_message_until:
                cv2.putText(
                    result,
                    f"Saved to {CONFIG_PATH.name}",
                    (10, 168),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow(WINDOW_NAME, result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_settings(controls)
                save_message_until = time.monotonic() + 1.5
            if key == ord("l"):
                leds_on = not leds_on
                if leds_on:
                    leds.turn_on_all()
                else:
                    leds.turn_off_all()
            if key == 27 or key == ord("q"):
                break
    finally:
        try:
            leds.turn_off_all()
        except Exception:
            pass
        try:
            leds.cleanup()
        except Exception:
            pass
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
