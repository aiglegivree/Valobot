from __future__ import annotations

import cv2
import numpy as np


CAM_ID = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
WINDOW_NAME = "Obstacle Edge Detector"


def line_length(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line
    return float(np.hypot(x2 - x1, y2 - y1))


def line_angle_deg(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def angle_difference_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def midpoint_distance(line_a: np.ndarray, line_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = line_a
    bx1, by1, bx2, by2 = line_b
    a_mid = np.array([(ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5], dtype=np.float32)
    b_mid = np.array([(bx1 + bx2) * 0.5, (by1 + by2) * 0.5], dtype=np.float32)
    return float(np.linalg.norm(a_mid - b_mid))


def detect_main_edges(frame: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20,
    )

    if raw_lines is None:
        return []

    candidates = [line[0] for line in raw_lines]
    candidates.sort(key=line_length, reverse=True)

    selected: list[np.ndarray] = []
    for candidate in candidates:
        if not selected:
            selected.append(candidate)
            continue

        too_similar = False
        for chosen in selected:
            angle_gap = angle_difference_deg(
                line_angle_deg(candidate),
                line_angle_deg(chosen),
            )
            mid_gap = midpoint_distance(candidate, chosen)
            if angle_gap < 10.0 and mid_gap < 40.0:
                too_similar = True
                break

        if not too_similar:
            selected.append(candidate)

        if len(selected) == 2:
            break

    return selected


def draw_lines(frame: np.ndarray, lines: list[np.ndarray]) -> np.ndarray:
    out = frame.copy()
    colors = [(0, 255, 0), (0, 165, 255)]

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = map(int, line)
        color = colors[i % len(colors)]
        cv2.line(out, (x1, y1), (x2, y2), color, 3)
        cv2.circle(out, (x1, y1), 4, color, -1)
        cv2.circle(out, (x2, y2), 4, color, -1)
        cv2.putText(
            out,
            f"Edge {i + 1}",
            (x1 + 5, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    cv2.putText(
        out,
        f"Detected edges: {len(lines)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    return out


def main() -> None:
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            lines = detect_main_edges(frame)
            display = draw_lines(frame, lines)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
