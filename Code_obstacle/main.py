from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import sys


WINDOW_NAME = "Obstacle Edge Direction"
CONTROLS_WINDOW = "Detection Controls"

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_V2_DIR = ROOT_DIR / "Code_v2"
if str(CODE_V2_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_V2_DIR))

from config import W, H, CAM_ID, USE_PICAMERA2  # noqa: E402


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
            frame = self.cam.capture_array()
            return True, frame

        ret, frame = self.cam.read()
        return ret, frame

    def release(self) -> None:
        if self.type == "opencv":
            self.cam.release()


def nothing(_: int) -> None:
    pass


def create_trackbars() -> None:
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 320, 120)
    cv2.createTrackbar("Canny low", CONTROLS_WINDOW, 50, 255, nothing)
    cv2.createTrackbar("Canny high", CONTROLS_WINDOW, 135, 255, nothing)


def get_detection_params() -> tuple[int, int]:
    canny_low = cv2.getTrackbarPos("Canny low", CONTROLS_WINDOW)
    canny_high = cv2.getTrackbarPos("Canny high", CONTROLS_WINDOW)

    canny_low = max(1, canny_low)
    canny_high = max(canny_low + 1, canny_high)
    return canny_low, canny_high


def detect_edges(
    frame: np.ndarray,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map = cv2.Canny(blur, canny_low, canny_high)
    return edge_map


def draw_result(
    frame: np.ndarray,
    edge_map: np.ndarray,
    params: tuple[int, int],
) -> np.ndarray:
    display = frame.copy()

    edge_vis = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((display, edge_vis))

    edge_count = int(np.count_nonzero(edge_map))
    cv2.putText(
        combined,
        f"Edge pixels: {edge_count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    cv2.putText(
        combined,
        "Mode: Canny only",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    canny_low, canny_high = params
    cv2.putText(
        combined,
        f"Canny: {canny_low}/{canny_high}",
        (10, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )

    cv2.putText(
        combined,
        "Frame",
        (10, combined.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        "Canny edges",
        (frame.shape[1] + 10, combined.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    return combined


def main() -> None:
    cam = Camera()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    create_trackbars()

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            params = get_detection_params()
            edge_map = detect_edges(frame, *params)
            result = draw_result(frame, edge_map, params)

            cv2.imshow(WINDOW_NAME, result)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
