#!/usr/bin/env python3
"""
Corridor following on Raspberry Pi 4 using a single camera (monocular),
based on line detection (Canny + HoughLinesP) and a simple steering command.

- Reads frames from Pi camera (libcamera) if available, else from /dev/video0.
- Detects left/right wall lines in a ROI (lower half of image).
- Computes a normalized lateral error (center offset) and suggests steering.

Dependencies:
  sudo apt-get install -y python3-opencv
Optional (recommended for Pi camera via libcamera):
  sudo apt-get install -y python3-picamera2

Run:
  python3 corridor_follow.py

Notes:
- This outputs steering values; connect it to your motor controller separately.
"""

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np


# -----------------------------
# Capture backend (PiCamera2 or OpenCV VideoCapture)
# -----------------------------
class FrameSource:
    def read(self) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class OpenCVCapture(FrameSource):
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(device_index)
        # Best effort settings (may be ignored by some cameras)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class PiCamera2Capture(FrameSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        # Import locally to avoid hard dependency
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)

    def read(self):
        frame = self.picam2.capture_array()  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def release(self):
        self.picam2.stop()


def make_source(width: int, height: int, fps: int) -> FrameSource:
    # Prefer PiCamera2 if available
    try:
        import picamera2  # noqa: F401
        return PiCamera2Capture(width=width, height=height, fps=fps)
    except Exception:
        return OpenCVCapture(device_index=0, width=width, height=height, fps=fps)


# -----------------------------
# Corridor following logic
# -----------------------------
@dataclass
class FollowResult:
    # error in [-1, 1], positive means corridor center is to the right -> steer right
    center_error: float
    # recommended steering in [-1, 1]
    steer: float
    # diagnostics
    left_line: Optional[Tuple[int, int, int, int]] = None
    right_line: Optional[Tuple[int, int, int, int]] = None
    vanishing_point: Optional[Tuple[int, int]] = None


def roi_lower_half(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return ROI image and its (x0, y0) offset in original image."""
    h, w = img.shape[:2]
    y0 = int(h * 0.45)
    return img[y0:h, 0:w], (0, y0)


def detect_lines(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # Slight blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)

    # Morphological close to connect gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return edges


def line_slope_intercept(x1, y1, x2, y2) -> Optional[Tuple[float, float]]:
    dx = (x2 - x1)
    dy = (y2 - y1)
    if abs(dx) < 1e-6:
        return None
    m = dy / dx
    b = y1 - m * x1
    return m, b


def pick_representative_line(lines: List[Tuple[int, int, int, int]], prefer_left: bool) -> Optional[Tuple[int, int, int, int]]:
    """
    Pick the best line among candidates by length and position.
    prefer_left=True picks negative slope lines; prefer_left=False picks positive slope lines.
    """
    if not lines:
        return None

    best = None
    best_score = -1.0

    for (x1, y1, x2, y2) in lines:
        li = line_slope_intercept(x1, y1, x2, y2)
        if li is None:
            continue
        m, _b = li

        # Corridor walls: left wall tends to have negative slope in image, right wall positive
        if prefer_left and m >= -0.2:
            continue
        if (not prefer_left) and m <= 0.2:
            continue

        length = math.hypot(x2 - x1, y2 - y1)

        # Score: length + bias toward being near corresponding side
        midx = 0.5 * (x1 + x2)
        # prefer line more to the left or right
        side_bias = -midx if prefer_left else midx
        score = length + 0.001 * side_bias

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    return best


def extend_line_to_roi_bounds(x1, y1, x2, y2, roi_w, roi_h) -> Optional[Tuple[int, int, int, int]]:
    """Extend a line segment to intersect ROI top and bottom for stable visualization/geometry."""
    li = line_slope_intercept(x1, y1, x2, y2)
    if li is None:
        return None
    m, b = li

    # y = m x + b -> x = (y - b)/m
    y_top = 0
    y_bot = roi_h - 1
    x_top = int((y_top - b) / m)
    x_bot = int((y_bot - b) / m)

    # Clamp to ROI width for drawing; if far outside, still clamp.
    x_top = max(0, min(roi_w - 1, x_top))
    x_bot = max(0, min(roi_w - 1, x_bot))
    return (x_top, y_top, x_bot, y_bot)


def intersect_lines(l1: Tuple[int, int, int, int], l2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
    """Intersection of two infinite lines in ROI coordinates."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    return (px, py)


def corridor_follow(frame_bgr: np.ndarray, kp: float = 1.2, deadband: float = 0.03) -> FollowResult:
    h, w = frame_bgr.shape[:2]
    roi, (x0, y0) = roi_lower_half(frame_bgr)
    roi_h, roi_w = roi.shape[:2]

    edges = detect_lines(roi)

    # Hough line detection
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=int(roi_w * 0.15),
        maxLineGap=15,
    )

    left_candidates = []
    right_candidates = []
    if raw is not None:
        for seg in raw[:, 0, :]:
            x1, y1, x2, y2 = map(int, seg)
            li = line_slope_intercept(x1, y1, x2, y2)
            if li is None:
                continue
            m, _b = li
            if m < 0:
                left_candidates.append((x1, y1, x2, y2))
            else:
                right_candidates.append((x1, y1, x2, y2))

    left = pick_representative_line(left_candidates, prefer_left=True)
    right = pick_representative_line(right_candidates, prefer_left=False)

    left_ext = extend_line_to_roi_bounds(*left, roi_w, roi_h) if left else None
    right_ext = extend_line_to_roi_bounds(*right, roi_w, roi_h) if right else None

    # Estimate corridor center at bottom of ROI
    y_ref = roi_h - 1
    x_center_est = roi_w / 2.0

    def x_at_y(line_ext, y):
        x1, y1, x2, y2 = line_ext
        # avoid division by 0 by using param form
        if abs(y2 - y1) < 1e-6:
            return None
        t = (y - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        return x

    xl = x_at_y(left_ext, y_ref) if left_ext else None
    xr = x_at_y(right_ext, y_ref) if right_ext else None

    if xl is not None and xr is not None and xr - xl > 20:
        x_center_est = 0.5 * (xl + xr)
    elif xl is not None:
        # if only left wall: assume corridor width roughly half image -> infer center
        x_center_est = min(roi_w - 1, xl + roi_w * 0.35)
    elif xr is not None:
        x_center_est = max(0, xr - roi_w * 0.35)

    # Normalized error: + means target center is to the right of image center
    err_px = x_center_est - (roi_w / 2.0)
    center_error = float(err_px / (roi_w / 2.0))
    # Deadband + proportional steering
    if abs(center_error) < deadband:
        steer = 0.0
    else:
        steer = float(np.clip(kp * center_error, -1.0, 1.0))

    vp = None
    if left_ext and right_ext:
        p = intersect_lines(left_ext, right_ext)
        if p is not None:
            vp = (int(p[0] + x0), int(p[1] + y0))

    # Convert ROI line coords back to full-image coords for drawing
    def to_full(line_ext):
        x1, y1, x2, y2 = line_ext
        return (x1 + x0, y1 + y0, x2 + x0, y2 + y0)

    return FollowResult(
        center_error=center_error,
        steer=steer,
        left_line=to_full(left_ext) if left_ext else None,
        right_line=to_full(right_ext) if right_ext else None,
        vanishing_point=vp,
    )


# -----------------------------
# Demo loop (visualization + printing steering)
# -----------------------------
def main():
    width, height, fps = 640, 480, 30
    src = make_source(width, height, fps)

    print("Press 'q' to quit.")
    last_t = time.time()
    ema_fps = 0.0

    try:
        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                continue

            res = corridor_follow(frame, kp=1.2, deadband=0.03)

            # FPS estimate
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                ema_fps = 0.9 * ema_fps + 0.1 * inst if ema_fps > 0 else inst

            # Draw diagnostics
            vis = frame.copy()
            if res.left_line:
                cv2.line(vis, res.left_line[:2], res.left_line[2:], (0, 255, 0), 2)
            if res.right_line:
                cv2.line(vis, res.right_line[:2], res.right_line[2:], (0, 255, 0), 2)
            if res.vanishing_point:
                cv2.circle(vis, res.vanishing_point, 6, (255, 0, 0), -1)

            h, w = vis.shape[:2]
            # show target center estimate as vertical line
            # reconstruct last used ROI center estimate indirectly:
            # draw error as arrow around image center
            cx = w // 2
            cy = int(h * 0.8)
            arrow_len = int(res.steer * 150)
            cv2.arrowedLine(vis, (cx, cy), (cx + arrow_len, cy), (0, 0, 255), 3, tipLength=0.2)

            cv2.putText(
                vis,
                f"steer={res.steer:+.2f}  err={res.center_error:+.2f}  fps~{ema_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Print steering (for your motor controller)
            # Convention: steer in [-1,1]. You map it to differential speeds.
            print(f"{res.steer:+.3f}")

            cv2.imshow("corridor_follow", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        src.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
