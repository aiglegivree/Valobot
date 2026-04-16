from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_V2_DIR = ROOT_DIR / "Code_v2"
if str(CODE_V2_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_V2_DIR))

from camera import Camera  # noqa: E402
from config import W, H  # noqa: E402
from leds import LED  # noqa: E402
from motors import MotorDriver  # noqa: E402
from vision import detect_line  # noqa: E402

from corridor_config import (  # noqa: E402
    AURA_FARM_DELAY,
    BLIND_FORWARD_SPEED,
    BLUE_ALIGN_GAIN,
    BLUE_ALIGN_SIGN,
    CONTINUE_SEC,
    DASHED_GAP_PX,
    DASHED_SEGMENT_PX,
    DEFAULT_TURN_DIRECTION,
    FORWARD_BASE_SPEED,
    FORWARD_LEFT_SCALE,
    FORWARD_RIGHT_SCALE,
    MAX_STEER_DELTA,
    MEDIAN_HISTORY_FRAMES,
    RECORDINGS_DIR,
    RECORD_RUN_FPS,
    RECORD_RUN_VIDEO,
    SHOW_DEBUG,
    SIDE_BLUE_INTERSECTION_GAIN,
    SIDE_BLUE_TARGET_FRAC,
    SINGLE_WALL_SEARCH_DELTA,
    STEER_DEADBAND_PX,
    STEER_SIGN,
    TURN_BLIND_SEC,
    TURN_CANDIDATE_MISS_FRAMES,
    TURN_DURATION_SEC,
    TURN_HEIGHT_FRAC,
    TURN_REACQUIRE_TOP_FRAC,
    TURN_RETRY_BLIND_EXTRA_SEC,
    TURN_RETRY_DURATION_MULTIPLIER,
    TURN_SIGN,
    TURN_SPEED,
    USE_LEDS,
    VANISHING_STEER_GAIN,
)
from obstacle_vision_config import load_obstacle_settings  # noqa: E402


WINDOW_NAME = "Corridor Navigation"
EDGE_WINDOW_NAME = "Corridor Edges"
WINDOW_SCALE = 2

RIGHT_COLOR = (0, 0, 255)
LEFT_COLOR = (0, 255, 0)
INFRONT_COLOR = (255, 0, 0)
TARGET_COLOR = (255, 0, 255)
MEMORY_COLOR = (255, 255, 255)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def intersect_hough_lines(line1: tuple[float, float], line2: tuple[float, float]) -> tuple[float, float] | None:
    rho1, theta1 = line1
    rho2, theta2 = line2
    a1 = math.cos(theta1)
    b1 = math.sin(theta1)
    a2 = math.cos(theta2)
    b2 = math.sin(theta2)
    A = np.array([[a1, b1], [a2, b2]], dtype=np.float32)
    B = np.array([rho1, rho2], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    x, y = np.linalg.solve(A, B)
    return float(x), float(y)


def line_tilt_px(line: tuple[float, float], width: int) -> float | None:
    y_left = hough_y_at_x(line, 0)
    y_right = hough_y_at_x(line, width - 1)
    if y_left is None or y_right is None:
        return None
    return float(y_right - y_left)


def median_line(lines: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not lines:
        return None
    rhos = np.array([line[0] for line in lines], dtype=np.float32)
    thetas = np.array([line[1] for line in lines], dtype=np.float32)
    return float(np.median(rhos)), float(np.median(thetas))


def stable_line_from_history(history: deque[tuple[float, float] | None]) -> tuple[float, float] | None:
    valid = [line for line in history if line is not None]
    return median_line(valid)


def draw_dashed_line(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return

    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    offset = 0.0
    while offset < length:
        start_x = int(round(x1 + dx * offset))
        start_y = int(round(y1 + dy * offset))
        end_offset = min(offset + DASHED_SEGMENT_PX, length)
        end_x = int(round(x1 + dx * end_offset))
        end_y = int(round(y1 + dy * end_offset))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
        offset += DASHED_SEGMENT_PX + DASHED_GAP_PX


class CorridorNavigator:
    def __init__(self):
        self.cap = Camera()
        self.vision_settings = load_obstacle_settings()
        self.line_history = {
            "left": deque(maxlen=MEDIAN_HISTORY_FRAMES),
            "right": deque(maxlen=MEDIAN_HISTORY_FRAMES),
            "infront": deque(maxlen=MEDIAN_HISTORY_FRAMES),
        }
        self.side_memory_line: dict[str, tuple[float, float] | None] = {
            "left": None,
            "right": None,
        }
        self.side_memory_time = {
            "left": 0.0,
            "right": 0.0,
        }
        self.side_miss_counts = {
            "left": 0,
            "right": 0,
        }
        self.turn_direction = DEFAULT_TURN_DIRECTION
        self.frame: np.ndarray | None = None
        self.edges: np.ndarray | None = None
        self.current_lines: dict[str, tuple[float, float] | None] = {}
        self.stable_lines: dict[str, tuple[float, float] | None] = {}

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def reset_for_forward(self) -> None:
        self.side_miss_counts["left"] = 0
        self.side_miss_counts["right"] = 0
        self.turn_direction = DEFAULT_TURN_DIRECTION

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

    def detect_hough_lines(self, edges: np.ndarray) -> np.ndarray | None:
        return cv2.HoughLines(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.vision_settings["hough_threshold"],
        )

    def classify_lines(self, lines: np.ndarray | None) -> dict[str, list[tuple[float, float]]]:
        categories = {
            "right": [],
            "left": [],
            "infront": [],
        }
        if lines is None:
            return categories

        for line in lines:
            rho, theta = line[0]
            angle_deg = math.degrees(theta)
            if self.vision_settings["right_angle_low"] <= angle_deg <= self.vision_settings["right_angle_high"]:
                categories["right"].append((rho, theta))
            elif self.vision_settings["left_angle_low"] <= angle_deg <= self.vision_settings["left_angle_high"]:
                categories["left"].append((rho, theta))
            elif self.vision_settings["infront_angle_low"] <= angle_deg <= self.vision_settings["infront_angle_high"]:
                categories["infront"].append((rho, theta))
        return categories

    def update_histories(self, current_lines: dict[str, tuple[float, float] | None]) -> dict[str, tuple[float, float] | None]:
        stable_lines: dict[str, tuple[float, float] | None] = {}
        for key, line in current_lines.items():
            self.line_history[key].append(line)
            stable_lines[key] = stable_line_from_history(self.line_history[key])
        return stable_lines

    def step(self, now: float) -> tuple[bool, dict[str, object] | None]:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        self.frame = frame
        self.edges = self.detect_edges(frame)
        raw_lines = self.detect_hough_lines(self.edges)
        categories = self.classify_lines(raw_lines)
        current_lines = {key: median_line(values) for key, values in categories.items()}
        stable_lines = self.update_histories(current_lines)
        t_detected = bool(detect_line(frame, angle_deg=20).t_detected)

        self.current_lines = current_lines
        self.stable_lines = stable_lines

        effective_lines: dict[str, tuple[float, float] | None] = {
            "left": None,
            "right": None,
        }
        dashed_side = {
            "left": False,
            "right": False,
        }

        for side in ("left", "right"):
            current_line = current_lines[side]
            stable_line = stable_lines[side]
            if current_line is not None and stable_line is not None:
                self.side_memory_line[side] = stable_line
                self.side_memory_time[side] = now
                self.side_miss_counts[side] = 0
                effective_lines[side] = stable_line
            else:
                self.side_miss_counts[side] += 1
                memory_line = self.side_memory_line[side]
                if memory_line is not None and (now - self.side_memory_time[side]) <= CONTINUE_SEC:
                    effective_lines[side] = memory_line
                    dashed_side[side] = True

        left_seen_now = current_lines["left"] is not None
        right_seen_now = current_lines["right"] is not None
        if left_seen_now and not right_seen_now:
            self.turn_direction = "RIGHT"
        elif right_seen_now and not left_seen_now:
            self.turn_direction = "LEFT"
        elif left_seen_now and right_seen_now:
            self.turn_direction = "RIGHT"

        vanishing_point = None
        if effective_lines["left"] is not None and effective_lines["right"] is not None:
            vanishing_point = intersect_hough_lines(effective_lines["left"], effective_lines["right"])

        blue_line = stable_lines["infront"]
        blue_y_center = hough_y_at_x(blue_line, W / 2.0) if blue_line is not None else None
        blue_tilt = line_tilt_px(blue_line, W) if blue_line is not None else None
        turn_trigger_y = H * (1.0 - TURN_HEIGHT_FRAC)
        blue_top_limit = H * TURN_REACQUIRE_TOP_FRAC

        result = {
            "frame": frame,
            "edges": self.edges,
            "current_lines": current_lines,
            "stable_lines": stable_lines,
            "effective_lines": effective_lines,
            "dashed_side": dashed_side,
            "vanishing_point": vanishing_point,
            "blue_line": blue_line,
            "blue_y_center": blue_y_center,
            "blue_tilt": blue_tilt,
            "left_blue_intersection": (
                intersect_hough_lines(stable_lines["left"], blue_line)
                if stable_lines["left"] is not None and blue_line is not None
                else None
            ),
            "right_blue_intersection": (
                intersect_hough_lines(stable_lines["right"], blue_line)
                if stable_lines["right"] is not None and blue_line is not None
                else None
            ),
            "turn_trigger": blue_y_center is not None and blue_y_center >= turn_trigger_y,
            "blue_top_visible": blue_y_center is not None and blue_y_center <= blue_top_limit,
            "current_side_count": int(current_lines["left"] is not None) + int(current_lines["right"] is not None),
            "turn_direction": self.turn_direction,
            "t_detected": t_detected,
        }
        return True, result


def draw_line(image: np.ndarray, line: tuple[float, float], color: tuple[int, int, int], dashed: bool = False) -> None:
    pt1, pt2 = hough_line_endpoints(*line)
    if dashed:
        draw_dashed_line(image, pt1, pt2, color, 2)
    else:
        cv2.line(image, pt1, pt2, color, 2)


def draw_debug(
    frame: np.ndarray,
    vision: dict[str, object],
    state: str,
    turn_phase: int,
    turn_direction: str | None,
    command_left: float,
    command_right: float,
) -> np.ndarray:
    overlay = frame.copy()

    stable_lines = vision["stable_lines"]
    effective_lines = vision["effective_lines"]
    dashed_side = vision["dashed_side"]

    right_line = stable_lines["right"]
    left_line = stable_lines["left"]
    blue_line = stable_lines["infront"]

    if right_line is not None:
        draw_line(overlay, right_line, RIGHT_COLOR, dashed=bool(dashed_side["right"]))
    elif effective_lines["right"] is not None:
        draw_line(overlay, effective_lines["right"], MEMORY_COLOR, dashed=True)

    if left_line is not None:
        draw_line(overlay, left_line, LEFT_COLOR, dashed=bool(dashed_side["left"]))
    elif effective_lines["left"] is not None:
        draw_line(overlay, effective_lines["left"], MEMORY_COLOR, dashed=True)

    if blue_line is not None:
        draw_line(overlay, blue_line, INFRONT_COLOR, dashed=False)

    if vision["vanishing_point"] is not None:
        ix, iy = vision["vanishing_point"]
        cv2.circle(overlay, (int(round(ix)), int(round(iy))), 5, TARGET_COLOR, -1)

    for key, color in [("left_blue_intersection", LEFT_COLOR), ("right_blue_intersection", RIGHT_COLOR)]:
        point = vision[key]
        if point is None:
            continue
        ix, iy = point
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(overlay, (int(round(ix)), int(round(iy))), 5, color, -1)

    blue_y_center = vision["blue_y_center"]
    if blue_y_center is not None:
        cv2.circle(overlay, (W // 2, int(round(blue_y_center))), 5, INFRONT_COLOR, -1)

    turn_trigger_y = int(round(H * (1.0 - TURN_HEIGHT_FRAC)))
    cv2.line(overlay, (0, turn_trigger_y), (W - 1, turn_trigger_y), INFRONT_COLOR, 1)

    top_reacquire_y = int(round(H * TURN_REACQUIRE_TOP_FRAC))
    cv2.line(overlay, (0, top_reacquire_y), (W - 1, top_reacquire_y), (255, 255, 0), 1)

    cv2.putText(overlay, f"State: {state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        overlay,
        f"Turn direction var: {vision['turn_direction']}",
        (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        overlay,
        f"Turn phase: {turn_phase} dir: {turn_direction}",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        overlay,
        (
            f"Canny {load_obstacle_settings()['canny_low']}/{load_obstacle_settings()['canny_high']} "
            f"H {load_obstacle_settings()['hough_threshold']}"
        ),
        (10, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        overlay,
        f"Sides now: {vision['current_side_count']}  blue_y: {vision['blue_y_center']}",
        (10, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        overlay,
        f"Cmd L/R: {command_left:.2f} / {command_right:.2f}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        overlay,
        f"T detected: {'YES' if vision['t_detected'] else 'NO'}",
        (10, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0) if vision["t_detected"] else (0, 0, 255),
        1,
    )
    return overlay


def forward_wheels(correction: float) -> tuple[float, float]:
    base_left = FORWARD_BASE_SPEED * FORWARD_LEFT_SCALE
    base_right = FORWARD_BASE_SPEED * FORWARD_RIGHT_SCALE
    left = clamp(base_left + correction, -1.0, 1.0)
    right = clamp(base_right - correction, -1.0, 1.0)
    return left, right


def single_wall_search_wheels(visible_side: str) -> tuple[float, float]:
    if visible_side == "left":
        return forward_wheels(-SINGLE_WALL_SEARCH_DELTA)
    if visible_side == "right":
        return forward_wheels(SINGLE_WALL_SEARCH_DELTA)
    return forward_wheels(0.0)


def blind_forward_wheels() -> tuple[float, float]:
    return (
        clamp(BLIND_FORWARD_SPEED, -1.0, 1.0),
        clamp(BLIND_FORWARD_SPEED, -1.0, 1.0),
    )


def fail_stop(motors: MotorDriver, leds: LED, aura_started: bool) -> bool:
    motors.stop()
    if not USE_LEDS:
        return aura_started
    if not aura_started:
        leds.aura_farm_start()
        return True
    leds.aura_farm_step(delay=AURA_FARM_DELAY)
    return aura_started


def build_recording_frame(debug_frame: np.ndarray, edge_frame: np.ndarray | None) -> np.ndarray:
    if edge_frame is None:
        edge_vis = np.zeros_like(debug_frame)
    elif len(edge_frame.shape) == 2:
        edge_vis = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)
    else:
        edge_vis = edge_frame

    target_height = debug_frame.shape[0]
    target_width = debug_frame.shape[1]
    edge_resized = cv2.resize(edge_vis, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    combined = np.hstack((debug_frame, edge_resized))
    cv2.putText(combined, "Navigation", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(
        combined,
        "Edges",
        (target_width + 15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    return combined


def create_run_video_writer(frame_size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not RECORD_RUN_VIDEO:
        return None

    recordings_dir = Path(__file__).resolve().parent / RECORDINGS_DIR
    recordings_dir.mkdir(parents=True, exist_ok=True)

    run_index = 1
    while True:
        output_path = recordings_dir / f"run_{run_index:03d}.mp4"
        if not output_path.exists():
            break
        run_index += 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, RECORD_RUN_FPS, frame_size)
    if not writer.isOpened():
        return None

    print(f"Recording run video to: {output_path}")
    return writer


def setup_windows() -> None:
    display_width = W * WINDOW_SCALE
    display_height = H * WINDOW_SCALE

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(EDGE_WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)
    cv2.resizeWindow(EDGE_WINDOW_NAME, display_width, display_height)

    cv2.moveWindow(WINDOW_NAME, 40, 40)
    cv2.moveWindow(EDGE_WINDOW_NAME, 60 + display_width, 40)


def main() -> None:
    navigator = CorridorNavigator()
    motors = MotorDriver()
    leds = LED() if USE_LEDS else None
    video_writer: cv2.VideoWriter | None = None

    if SHOW_DEBUG:
        setup_windows()

    state = "FORWARD"
    turn_phase = 0
    turn_direction = DEFAULT_TURN_DIRECTION
    turn_direction_sign = 1
    turn_started_at = 0.0
    blind_until = 0.0
    turn_deadline = 0.0
    aura_started = False
    command_left = 0.0
    command_right = 0.0

    try:
        while True:
            now = time.monotonic()
            ret, vision = navigator.step(now)
            if not ret or vision is None:
                continue

            frame = vision["frame"]

            if bool(vision["t_detected"]):
                state = "FAIL"
                command_left = 0.0
                command_right = 0.0
                aura_started = fail_stop(motors, leds, aura_started)

            elif state == "FORWARD":
                if leds is not None:
                    leds.turn_on_all()
                aura_started = False

                if bool(vision["turn_trigger"]):
                    state = "TURN"
                    turn_phase = 1
                    turn_direction = str(vision["turn_direction"])
                    turn_direction_sign = TURN_SIGN * (1 if turn_direction == "RIGHT" else -1)
                    turn_started_at = now
                    blind_until = now + TURN_BLIND_SEC
                    turn_deadline = now + TURN_DURATION_SEC
                    command_left = turn_direction_sign * TURN_SPEED
                    command_right = -turn_direction_sign * TURN_SPEED
                    motors.set_speeds(command_left, command_right)
                else:
                    correction = None
                    visible_side = None
                    if vision["vanishing_point"] is not None:
                        error_pixels = float(vision["vanishing_point"][0]) - (W / 2.0)
                        if abs(error_pixels) < STEER_DEADBAND_PX:
                            error_pixels = 0.0
                        error_norm = error_pixels / max(W / 2.0, 1.0)
                        correction = clamp(
                            STEER_SIGN * VANISHING_STEER_GAIN * error_norm,
                            -MAX_STEER_DELTA,
                            MAX_STEER_DELTA,
                        )
                    elif vision["blue_line"] is not None and int(vision["current_side_count"]) <= 1:
                        side_intersection = None
                        target_x = None
                        if vision["current_lines"]["left"] is not None:
                            side_intersection = vision["left_blue_intersection"]
                            target_x = W * SIDE_BLUE_TARGET_FRAC
                        elif vision["current_lines"]["right"] is not None:
                            side_intersection = vision["right_blue_intersection"]
                            target_x = W * (1.0 - SIDE_BLUE_TARGET_FRAC)

                        if side_intersection is not None and target_x is not None:
                            error_pixels = float(side_intersection[0]) - target_x
                            if abs(error_pixels) < STEER_DEADBAND_PX:
                                error_pixels = 0.0
                            error_norm = error_pixels / max(W / 2.0, 1.0)
                            correction = clamp(
                                STEER_SIGN * SIDE_BLUE_INTERSECTION_GAIN * error_norm,
                                -MAX_STEER_DELTA,
                                MAX_STEER_DELTA,
                            )
                        else:
                            blue_tilt = vision["blue_tilt"]
                            if blue_tilt is not None:
                                tilt_norm = float(blue_tilt) / max(H, 1.0)
                                correction = clamp(
                                    BLUE_ALIGN_SIGN * BLUE_ALIGN_GAIN * tilt_norm,
                                    -MAX_STEER_DELTA,
                                    MAX_STEER_DELTA,
                                )
                    elif vision["current_lines"]["left"] is not None and vision["blue_line"] is None:
                        visible_side = "left"
                    elif vision["current_lines"]["right"] is not None and vision["blue_line"] is None:
                        visible_side = "right"

                    if correction is None:
                        if visible_side is not None:
                            command_left, command_right = single_wall_search_wheels(visible_side)
                            motors.set_speeds(command_left, command_right)
                        else:
                            command_left, command_right = blind_forward_wheels()
                            motors.set_speeds(command_left, command_right)
                    else:
                        command_left, command_right = forward_wheels(correction)
                        motors.set_speeds(command_left, command_right)

            elif state == "TURN":
                if leds is not None:
                    leds.turn_on_all()
                aura_started = False
                command_left = turn_direction_sign * TURN_SPEED
                command_right = -turn_direction_sign * TURN_SPEED
                motors.set_speeds(command_left, command_right)

                if now >= blind_until and bool(vision["blue_top_visible"]):
                    state = "FORWARD"
                    navigator.reset_for_forward()
                    command_left = 0.0
                    command_right = 0.0
                    motors.stop()
                elif now >= turn_deadline:
                    if turn_phase == 1:
                        turn_phase = 2
                        turn_direction = "LEFT" if turn_direction == "RIGHT" else "RIGHT"
                        turn_direction_sign = TURN_SIGN * (1 if turn_direction == "RIGHT" else -1)
                        turn_started_at = now
                        blind_until = now + TURN_BLIND_SEC + TURN_RETRY_BLIND_EXTRA_SEC
                        turn_deadline = now + (TURN_DURATION_SEC * TURN_RETRY_DURATION_MULTIPLIER)
                    else:
                        state = "FAIL"
                        navigator.reset_for_forward()
                        command_left = 0.0
                        command_right = 0.0
                        motors.stop()

            else:
                command_left = 0.0
                command_right = 0.0
                aura_started = fail_stop(motors, leds, aura_started)

            debug = draw_debug(frame, vision, state, turn_phase, turn_direction, command_left, command_right)
            recording_frame = build_recording_frame(debug, navigator.edges)

            if RECORD_RUN_VIDEO:
                if video_writer is None:
                    frame_size = (recording_frame.shape[1], recording_frame.shape[0])
                    video_writer = create_run_video_writer(frame_size)
                if video_writer is not None:
                    video_writer.write(recording_frame)

            if SHOW_DEBUG:
                cv2.imshow(WINDOW_NAME, debug)
                if navigator.edges is not None:
                    cv2.imshow(EDGE_WINDOW_NAME, navigator.edges)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

    finally:
        try:
            motors.stop()
        except Exception:
            pass

        try:
            if leds is not None:
                leds.aura_farm_stop()
        except Exception:
            pass

        try:
            navigator.release()
        except Exception:
            pass

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
