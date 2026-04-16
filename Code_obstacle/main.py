from __future__ import annotations

import cv2
import math
import numpy as np
import time
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_V2_DIR = ROOT_DIR / "Code_v2"
if str(CODE_V2_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_V2_DIR))

from camera import Camera  # noqa: E402
from motors import MotorDriver  # noqa: E402
from obstacle_vision_config import load_obstacle_settings  # noqa: E402


class CannyCorridorNavigator:
    def __init__(
        self,
        alpha: float = 0.25,
        stop_y_threshold: int = 150,
        candidate_miss_threshold: int = 5,
        target_half_width_ratio: float = 0.22,
        target_hold_frames: int = 6,
        recovery_stable_frames: int = 2,
        debug_angles: bool = True,
    ):
        self.cap = Camera()
        self.vision_settings = load_obstacle_settings()
        self.alpha = alpha
        self.stop_y_threshold = stop_y_threshold
        self.candidate_miss_threshold = candidate_miss_threshold
        self.target_half_width_ratio = target_half_width_ratio
        self.target_hold_frames = target_hold_frames
        self.recovery_stable_frames = recovery_stable_frames
        self.debug_angles = debug_angles

        self.horizontal_smooth = None
        self.left_smooth = None
        self.right_smooth = None

        self.horizontal_misses = 0
        self.left_misses = 0
        self.right_misses = 0

        self.left_stable_frames = 0
        self.right_stable_frames = 0

        self.turn_candidate = None
        self.candidate_locked = False
        self.last_turn_candidate = "RIGHT"

        self.state = "FORWARD"
        self.target_x = None
        self.target_visible = False
        self.target_mode = "NONE"
        self.intersection = None
        self.detected_angles = {"horizontal": None, "left": None, "right": None}
        self.last_target_x = None
        self.target_hold_count = 0

        self.frame = None
        self.original = None
        self.edges = None
        self.H = None
        self.W = None

    @staticmethod
    def median_line(lines):
        if not lines:
            return None
        rhos = np.array([line[0] for line in lines], dtype=np.float32)
        thetas = np.array([line[1] for line in lines], dtype=np.float32)
        return float(np.median(rhos)), float(np.median(thetas))

    @staticmethod
    def smooth_line(previous, current, alpha=0.25):
        if current is None:
            return previous
        if previous is None:
            return current
        prev_rho, prev_theta = previous
        curr_rho, curr_theta = current
        rho = alpha * curr_rho + (1.0 - alpha) * prev_rho
        theta = alpha * curr_theta + (1.0 - alpha) * prev_theta
        return rho, theta

    @staticmethod
    def hough_line_endpoints(rho, theta, length=1000):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * (a))
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * (a))
        return (x1, y1), (x2, y2)

    @staticmethod
    def intersect_hough_lines(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        a2 = np.cos(theta2)
        b2 = np.sin(theta2)
        A = np.array([[a1, b1], [a2, b2]], dtype=np.float32)
        B = np.array([rho1, rho2], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None
        x, y = np.linalg.solve(A, B)
        return float(x), float(y)

    @staticmethod
    def hough_y_at_x(line, x):
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        if abs(b) < 1e-6:
            return None
        return float((rho - a * x) / b)

    @staticmethod
    def hough_x_at_y(line, y):
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        if abs(a) < 1e-6:
            return None
        return float((rho - b * y) / a)

    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_map = cv2.Canny(
            blur,
            self.vision_settings["canny_low"],
            self.vision_settings["canny_high"],
        )

        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        filtered = np.zeros_like(edge_map)
        for contour in contours:
            if cv2.arcLength(contour, closed=False) >= float(self.vision_settings["contour_min_length"]):
                cv2.drawContours(filtered, [contour], -1, 255, 1)
        return filtered

    def detect_hough_lines(self, edges):
        return cv2.HoughLines(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.vision_settings["hough_threshold"],
        )

    @staticmethod
    def classify_lines(lines):
        horizontal_lines = []
        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                deg = math.degrees(theta)
                if 85 <= deg <= 95:
                    horizontal_lines.append((rho, theta))
                elif 30 < deg < 80:
                    left_lines.append((rho, theta))
                elif 100 < deg < 150:
                    right_lines.append((rho, theta))

        return horizontal_lines, left_lines, right_lines

    def update_smoothed_lines(self, horizontal_rep, left_rep, right_rep):
        if horizontal_rep is not None:
            self.horizontal_smooth = self.smooth_line(self.horizontal_smooth, horizontal_rep, self.alpha)
            self.horizontal_misses = 0
        else:
            self.horizontal_misses += 1

        if left_rep is not None:
            self.left_smooth = self.smooth_line(self.left_smooth, left_rep, self.alpha)
            self.left_misses = 0
            self.left_stable_frames += 1
        else:
            self.left_misses += 1
            self.left_stable_frames = 0

        if right_rep is not None:
            self.right_smooth = self.smooth_line(self.right_smooth, right_rep, self.alpha)
            self.right_misses = 0
            self.right_stable_frames += 1
        else:
            self.right_misses += 1
            self.right_stable_frames = 0

    def update_turn_candidate(self):
        if self.state != "FORWARD" or self.candidate_locked:
            return
        if self.turn_candidate is None:
            if self.left_misses >= self.candidate_miss_threshold and self.right_misses < self.candidate_miss_threshold:
                self.turn_candidate = "LEFT"
            elif self.right_misses >= self.candidate_miss_threshold and self.left_misses < self.candidate_miss_threshold:
                self.turn_candidate = "RIGHT"

    def update_state(self, update=True):
        if not update:
            return
        if self.state == "FORWARD":
            if self.horizontal_smooth is not None:
                y_center = self.hough_y_at_x(self.horizontal_smooth, self.W / 2)
                if y_center is not None and y_center > self.stop_y_threshold:
                    self.state = "TURN"
                    if self.turn_candidate is None and self.last_turn_candidate is not None:
                        self.turn_candidate = self.last_turn_candidate
                    self.candidate_locked = True
                    self.last_turn_candidate = self.turn_candidate
        elif self.state == "TURN":
            recovered_corridor = (
                self.left_stable_frames >= self.recovery_stable_frames
                or self.right_stable_frames >= self.recovery_stable_frames
            )
            if recovered_corridor:
                self.state = "FORWARD"
                self.candidate_locked = False
                self.turn_candidate = None

    def update_target(self):
        self.target_x = None
        self.target_visible = False
        self.target_mode = "NONE"
        self.intersection = None
        target_y = self.H - 12
        half_width_px = int(self.W * self.target_half_width_ratio)

        left_x = self.hough_x_at_y(self.left_smooth, target_y) if self.left_smooth is not None else None
        right_x = self.hough_x_at_y(self.right_smooth, target_y) if self.right_smooth is not None else None

        if left_x is not None and right_x is not None:
            intersection = self.intersect_hough_lines(self.left_smooth, self.right_smooth)
            if intersection is not None:
                ix, iy = intersection
                self.intersection = (ix, iy)
            self.target_x = int(max(0, min(self.W - 1, round((left_x + right_x) / 2.0))))
            self.target_visible = True
            self.target_mode = "BOTH"
            self.last_target_x = self.target_x
            self.target_hold_count = self.target_hold_frames
            return

        if left_x is not None:
            self.target_x = int(max(0, min(self.W - 1, round(left_x + half_width_px))))
            self.target_visible = True
            self.target_mode = "LEFT_ONLY"
            self.last_target_x = self.target_x
            self.target_hold_count = self.target_hold_frames
            return

        if right_x is not None:
            self.target_x = int(max(0, min(self.W - 1, round(right_x - half_width_px))))
            self.target_visible = True
            self.target_mode = "RIGHT_ONLY"
            self.last_target_x = self.target_x
            self.target_hold_count = self.target_hold_frames
            return

        if self.last_target_x is not None and self.target_hold_count > 0:
            self.target_x = self.last_target_x
            self.target_visible = True
            self.target_mode = "HOLD"
            self.target_hold_count -= 1

    def draw_hough_line(self, image, rho, theta, color, thickness=2):
        pt1, pt2 = self.hough_line_endpoints(rho, theta)
        cv2.line(image, pt1, pt2, color, thickness)

    def draw_overlay(self, image):
        cv2.line(image, (0, self.stop_y_threshold), (self.W - 1, self.stop_y_threshold), (0, 255, 255), 1)

        if self.horizontal_smooth is not None:
            self.draw_hough_line(image, *self.horizontal_smooth, (0, 255, 255), 2)
        if self.left_smooth is not None:
            self.draw_hough_line(image, *self.left_smooth, (0, 0, 255), 2)
        if self.right_smooth is not None:
            self.draw_hough_line(image, *self.right_smooth, (0, 255, 0), 2)

        if self.intersection is not None:
            ix, iy = self.intersection
            cv2.circle(image, (int(round(ix)), int(round(iy))), 5, (255, 255, 0), -1)

        if self.target_visible and self.target_x is not None:
            cv2.circle(image, (self.target_x, self.H - 1), 6, (255, 0, 255), -1)

        cv2.putText(image, f"State: {self.state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Turn: {self.turn_candidate}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Target: {self.target_mode}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(
            image,
            (
                f"Canny {self.vision_settings['canny_low']}/{self.vision_settings['canny_high']} "
                f"L{self.vision_settings['contour_min_length']} H{self.vision_settings['hough_threshold']}"
            ),
            (10, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        if self.debug_angles:
            y = 94
            for key, color in [("horizontal", (0, 255, 255)), ("left", (0, 0, 255)), ("right", (0, 255, 0))]:
                angle = self.detected_angles[key]
                if angle is not None:
                    cv2.putText(image, f"{key}: {angle:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    y += 18
        return image

    def read_frame(self):
        return self.cap.read()

    def process_frame(self, frame, update=True, draw=True):
        self.frame = frame
        self.original = frame.copy()
        self.H, self.W, _ = frame.shape

        self.edges = self.detect_edges(frame)
        lines = self.detect_hough_lines(self.edges)
        horizontal_lines, left_lines, right_lines = self.classify_lines(lines)

        horizontal_rep = self.median_line(horizontal_lines)
        left_rep = self.median_line(left_lines)
        right_rep = self.median_line(right_lines)

        self.detected_angles["horizontal"] = math.degrees(horizontal_rep[1]) if horizontal_rep else None
        self.detected_angles["left"] = math.degrees(left_rep[1]) if left_rep else None
        self.detected_angles["right"] = math.degrees(right_rep[1]) if right_rep else None

        self.update_smoothed_lines(horizontal_rep, left_rep, right_rep)
        self.update_turn_candidate()
        self.update_state(update=update)
        self.update_target()

        vis = None
        if draw:
            vis = self.draw_overlay(self.original.copy())
        return vis

    def step(self, draw=True, update=True):
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return False, None
        vis = self.process_frame(frame, update=update, draw=draw)
        return True, vis

    def get_control_data(self):
        return {
            "state": self.state,
            "turn_candidate": self.turn_candidate,
            "target_x": self.target_x,
            "target_visible": self.target_visible,
            "target_mode": self.target_mode,
            "frame_width": self.W,
        }

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


def clamp(value, low, high):
    return max(low, min(high, value))


def execute_turn(navigator, motors, direction, dead_zone_seconds, max_seconds, turn_speed):
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            return False

        motors.set_speeds(direction * turn_speed, -direction * turn_speed)
        allow_update = elapsed >= dead_zone_seconds
        ret, vis = navigator.step(draw=True, update=allow_update)
        if vis is not None:
            cv2.imshow("Obstacle Navigation", vis)
            cv2.imshow("Obstacle Edges", navigator.edges)
        cv2.waitKey(1)
        if allow_update and navigator.get_control_data()["state"] == "FORWARD":
            return True


def main():
    navigator = CannyCorridorNavigator()
    motors = MotorDriver()

    base_speed = 0.22
    k_p = 0.18
    max_correction = 0.18
    lost_target_speed = 0.16
    error_deadband_px = 8
    search_correction = 0.08

    turn_speed = 0.22
    dead_zone_seconds = 0.6
    max_seconds_90 = 2.0
    max_seconds_180 = 4.0

    try:
        while True:
            ret, vis = navigator.step(draw=True)
            if not ret:
                break

            data = navigator.get_control_data()
            state = data["state"]

            if state == "TURN":
                motors.stop()
                direction = 1 if data["turn_candidate"] == "RIGHT" else -1
                success = execute_turn(
                    navigator,
                    motors,
                    direction=direction,
                    dead_zone_seconds=dead_zone_seconds,
                    max_seconds=max_seconds_90,
                    turn_speed=turn_speed,
                )
                if not success:
                    success = execute_turn(
                        navigator,
                        motors,
                        direction=-direction,
                        dead_zone_seconds=max_seconds_90 + dead_zone_seconds,
                        max_seconds=max_seconds_180,
                        turn_speed=turn_speed,
                    )
                if not success:
                    motors.stop()
                    break
                continue

            target_x = data["target_x"]
            target_visible = data["target_visible"]
            target_mode = data["target_mode"]
            frame_width = data["frame_width"]

            if target_visible and frame_width is not None:
                center_x = frame_width / 2.0
                error_pixels = target_x - center_x
                if abs(error_pixels) < error_deadband_px:
                    error_pixels = 0.0
                error_norm = error_pixels / max(center_x, 1.0)
                correction = clamp(k_p * error_norm, -max_correction, max_correction)
                left_speed = clamp(base_speed + correction, -1.0, 1.0)
                right_speed = clamp(base_speed - correction, -1.0, 1.0)
                motors.set_speeds(left_speed, right_speed)
            else:
                if navigator.turn_candidate == "LEFT":
                    motors.set_speeds(
                        clamp(lost_target_speed - search_correction, -1.0, 1.0),
                        clamp(lost_target_speed + search_correction, -1.0, 1.0),
                    )
                elif navigator.turn_candidate == "RIGHT":
                    motors.set_speeds(
                        clamp(lost_target_speed + search_correction, -1.0, 1.0),
                        clamp(lost_target_speed - search_correction, -1.0, 1.0),
                    )
                else:
                    motors.set_speeds(lost_target_speed, lost_target_speed)

            if vis is not None:
                cv2.imshow("Obstacle Navigation", vis)
            if navigator.edges is not None:
                cv2.imshow("Obstacle Edges", navigator.edges)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        try:
            motors.cleanup()
        except Exception:
            pass
        navigator.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
