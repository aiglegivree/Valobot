from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import sys


WINDOW_NAME = "Obstacle Canny Test"
CONTROL_WINDOW_NAME = "Obstacle Controls"

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_V2_DIR = ROOT_DIR / "Code_v2"
if str(CODE_V2_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_V2_DIR))

from config import W, H, CAM_ID, USE_PICAMERA2  # noqa: E402


DEFAULT_CANNY_LOW = 68
DEFAULT_CANNY_HIGH = 135
DEFAULT_CONTOUR_MIN_LENGTH = 85
DEFAULT_HOUGH_THRESHOLD = 40
DEFAULT_HOUGH_MIN_LINE_LENGTH = 50
DEFAULT_HOUGH_MAX_LINE_GAP = 20


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


def create_controls() -> None:
    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW_NAME, 420, 260)
    cv2.createTrackbar("Canny low", CONTROL_WINDOW_NAME, DEFAULT_CANNY_LOW, 255, _noop)
    cv2.createTrackbar("Canny high", CONTROL_WINDOW_NAME, DEFAULT_CANNY_HIGH, 255, _noop)
    cv2.createTrackbar(
        "Contour min len",
        CONTROL_WINDOW_NAME,
        DEFAULT_CONTOUR_MIN_LENGTH,
        500,
        _noop,
    )
    cv2.createTrackbar("Hough thresh", CONTROL_WINDOW_NAME, DEFAULT_HOUGH_THRESHOLD, 300, _noop)
    cv2.createTrackbar(
        "Hough min len",
        CONTROL_WINDOW_NAME,
        DEFAULT_HOUGH_MIN_LINE_LENGTH,
        500,
        _noop,
    )
    cv2.createTrackbar(
        "Hough max gap",
        CONTROL_WINDOW_NAME,
        DEFAULT_HOUGH_MAX_LINE_GAP,
        200,
        _noop,
    )


def get_controls() -> dict[str, int]:
    canny_low = cv2.getTrackbarPos("Canny low", CONTROL_WINDOW_NAME)
    canny_high = cv2.getTrackbarPos("Canny high", CONTROL_WINDOW_NAME)
    contour_min_length = cv2.getTrackbarPos("Contour min len", CONTROL_WINDOW_NAME)
    hough_threshold = max(1, cv2.getTrackbarPos("Hough thresh", CONTROL_WINDOW_NAME))
    hough_min_line_length = cv2.getTrackbarPos("Hough min len", CONTROL_WINDOW_NAME)
    hough_max_line_gap = cv2.getTrackbarPos("Hough max gap", CONTROL_WINDOW_NAME)

    if canny_high < canny_low:
        canny_high = canny_low
        cv2.setTrackbarPos("Canny high", CONTROL_WINDOW_NAME, canny_high)

    return {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "contour_min_length": contour_min_length,
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "hough_max_line_gap": hough_max_line_gap,
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


def detect_hough_lines(
    edge_map: np.ndarray,
    hough_threshold: int,
    hough_min_line_length: int,
    hough_max_line_gap: int,
) -> np.ndarray | None:
    return cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length,
        maxLineGap=hough_max_line_gap,
    )


def draw_result(frame: np.ndarray, edge_map: np.ndarray, controls: dict[str, int], lines: np.ndarray | None) -> np.ndarray:
    overlay = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

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
        (
            f"Hough thr/min/gap: {controls['hough_threshold']}/"
            f"{controls['hough_min_line_length']}/{controls['hough_max_line_gap']}"
        ),
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Edges: {int(np.count_nonzero(edge_map))}  Lines: {0 if lines is None else len(lines)}",
        (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    return combined


def main() -> None:
    cam = Camera()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    create_controls()

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
                controls["hough_min_line_length"],
                controls["hough_max_line_gap"],
            )
            result = draw_result(frame, edges, controls, lines)
            cv2.imshow(WINDOW_NAME, result)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
