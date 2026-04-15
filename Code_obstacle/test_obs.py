from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import sys


WINDOW_NAME = "Obstacle Canny Test"

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_V2_DIR = ROOT_DIR / "Code_v2"
if str(CODE_V2_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_V2_DIR))

from config import W, H, CAM_ID, USE_PICAMERA2  # noqa: E402


CANNY_LOW = 68
CANNY_HIGH = 135
MIN_LENGTH = 85


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


def detect_edges(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered = np.zeros_like(edge_map)
    for contour in contours:
        if cv2.arcLength(contour, closed=False) >= float(MIN_LENGTH):
            cv2.drawContours(filtered, [contour], -1, 255, 1)
    return filtered


def draw_result(frame: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    edge_vis = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, edge_vis))

    cv2.putText(
        combined,
        f"Canny low/high: {CANNY_LOW}/{CANNY_HIGH}",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Min length: {MIN_LENGTH}",
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        f"Edge pixels: {int(np.count_nonzero(edge_map))}",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    return combined


def main() -> None:
    cam = Camera()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            edges = detect_edges(frame)
            result = draw_result(frame, edges)
            cv2.imshow(WINDOW_NAME, result)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
