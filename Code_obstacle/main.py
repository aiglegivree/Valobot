from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import sys


WINDOW_NAME = "Obstacle Edge Direction"

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


def detect_edges(frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map = cv2.Canny(blur, 60, 150)

    raw_lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=25,
        maxLineGap=15,
    )

    if raw_lines is None:
        return edge_map, []

    lines = [line[0] for line in raw_lines]
    return edge_map, lines


def line_direction_vector(line: np.ndarray) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return np.zeros(2, dtype=np.float32), 0.0

    # Undirected line orientation: force the vector to point to the right
    # so opposite representations of the same edge do not cancel out.
    if dx < 0 or (abs(dx) < 1e-6 and dy < 0):
        dx = -dx
        dy = -dy

    return np.array([dx / length, dy / length], dtype=np.float32), length


def line_left_x(line: np.ndarray) -> float:
    x1, _, x2, _ = map(float, line)
    return min(x1, x2)


def select_leftmost_edge(lines: list[np.ndarray]) -> np.ndarray | None:
    if not lines:
        return None
    return min(lines, key=line_left_x)


def draw_result(
    frame: np.ndarray,
    edge_map: np.ndarray,
    lines: list[np.ndarray],
    best_line: np.ndarray | None,
) -> np.ndarray:
    display = frame.copy()

    for line in lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(display, (x1, y1), (x2, y2), (80, 80, 80), 1)

    best_angle_deg: float | None = None
    if best_line is not None:
        x1, y1, x2, y2 = map(int, best_line)
        cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.circle(display, (x1, y1), 4, (0, 255, 255), -1)
        cv2.circle(display, (x2, y2), 4, (0, 255, 255), -1)

        best_vec, _ = line_direction_vector(best_line)
        best_angle_deg = float(np.degrees(np.arctan2(best_vec[1], best_vec[0])))

    edge_vis = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((display, edge_vis))

    cv2.putText(
        combined,
        f"Edges: {len(lines)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    angle_text = "Best angle: none" if best_angle_deg is None else f"Best angle: {best_angle_deg:.1f} deg"
    cv2.putText(
        combined,
        angle_text,
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    cv2.putText(
        combined,
        "Frame + best edge",
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

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            edge_map, lines = detect_edges(frame)
            best_line = select_leftmost_edge(lines)
            result = draw_result(frame, edge_map, lines, best_line)

            cv2.imshow(WINDOW_NAME, result)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
