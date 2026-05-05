from __future__ import annotations

from collections import deque
from pathlib import Path
import atexit
import math
import signal
import time

import cv2
import numpy as np

from camera import Camera  # noqa: E402
from config import (  # noqa: E402
    BAND_THICKNESS,
    BLUE_HIGH_CORRIDOR,
    BLUE_LOW_CORRIDOR,
    LOCK_FRAMES_REQUIRED,
    MAG_CAPACITY,
    MIN_TARGET_AREA_CORRIDOR,
    MIN_TARGET_WIDTH_CORRIDOR,
    NOMBRE_DE_CIBLES,
    POST_SHOT_CONFIRM_FRAMES,
    POST_SHOT_SETTLE_TIME,
    TARGET_MATCH_DIST_PX,
    TARGET_LOCK_MISS_FRAMES,
    CENTER_TOL,
    W,
    H,
)
from controller import Controller  # noqa: E402
from leds import LED  # noqa: E402
from music import MusicPlayer  # noqa: E402
from motors import MotorDriver  # noqa: E402
from tir_auto import (  # noqa: E402
    Shooter,
    aim_to_target,
    detect_blue_targets,
    match_target_by_position,
    pick_target_by_shot_order,
)
from vision import detect_line, draw_debug_slices, draw_mask_slices  # noqa: E402

from corridor_config import (  # noqa: E402
    AURA_FARM_DELAY,
    BLIND_FORWARD_SPEED,
    BLUE_ALIGN_GAIN,
    BLUE_ALIGN_SIGN,
    CONTINUE_SEC,
    DEFAULT_TURN_DIRECTION,
    FORWARD_BASE_SPEED,
    FORWARD_LEFT_SCALE,
    FORWARD_RIGHT_SCALE,
    HORIZONTAL_LINE_HEIGHT_JUMP_PX,
    LINE_CHANGING_RIGHT_TURN_SEC,
    LINE_CHANGING_WARNING_SEC,
    MAX_STEER_DELTA,
    MEDIAN_HISTORY_FRAMES,
    NO_LINE_SEARCH_TURN_SEC,
    NO_LINE_SEARCH_TURN_SPEED,
    SIDE_BLUE_INTERSECTION_GAIN,
    SIDE_BLUE_TARGET_FRAC,
    SINGLE_WALL_SEARCH_DELTA,
    STEER_DEADBAND_PX,
    STEER_SIGN,
    TURN_BLIND_SEC,
    TURN_DURATION_SEC,
    TURN_HEIGHT_FRAC,
    TURN_REACQUIRE_TOP_FRAC,
    TURN_RETRY_BLIND_EXTRA_SEC,
    TURN_RETRY_DURATION_MULTIPLIER,
    TURN_SIGN,
    TURN_SPEED,
    VANISHING_STEER_GAIN,
)
from fsm_config import (  # noqa: E402
    BLUE_DETECT_CONFIRM_FRAMES,
    BLUE_TO_T_RIGHT_TURN_SPEED,
    FLYWHEEL_SPINUP_TIME,
    FRONT_SIDE_CONFIRM_FRAMES,
    FIRST_T_SIDE_LINES_TOP_FRAC,
    FIRST_T_STRAIGHT_SEC,
    FIRST_T_STRAIGHT_SPEED,
    HIT_AREA_RATIO_THRESHOLD,
    POST_SHOOT_BACK_SEC,
    POST_SHOOT_BACK_SPEED,
    POST_SHOOT_DRIVE_SEC,
    POST_SHOOT_DRIVE_SPEED,
    PREFER_LINE_WHEN_VISIBLE,
    RECORDINGS_DIR,
    RECORD_RUN_FPS,
    RECORD_RUN_VIDEO,
    RETURN_T_IGNORE_SECONDS,
    SHOOT_SEARCH_BASE_SEC,
    SHOOT_SEARCH_SEC_INCREMENT,
    SHOOT_SEARCH_TURN_SPEED,
    SHOOT_STUCK_BOOST_SEC,
    SHOOT_STUCK_BOOST_SPEED,
    SHOOT_STUCK_TARGET_DELTA_PX,
    SHOOT_STUCK_TIMEOUT_SEC,
    SHOW_DEBUG,
    T_LIM_SPEED,
    T_LIM_TIME,
    USE_CORRIDOR_LEDS,
    USE_LEDS,
    USE_MUSIC,
    MUSIC_BACKEND,
    MUSIC_DEBUG,
    MUSIC_SINK,
    USE_SHOOTING_LEDS,
    WINDOW_SCALE,
)
from obstacle_vision_config import load_obstacle_settings  # noqa: E402


MAIN_WINDOW = "FSM Navigation"
AUX_WINDOW = "FSM Aux"

STATE_RELOAD = "RELOAD"
STATE_LINE_FOLLOW = "LINE_FOLLOW"
STATE_POST_T_STRAIGHT = "POST_T_STRAIGHT"
STATE_CORRIDOR_FOLLOW = "CORRIDOR_FOLLOW"
STATE_BLUE_TO_T = "BLUE_TO_T"
STATE_LINE_TO_SECOND_T = "LINE_TO_SECOND_T"
STATE_POST_SECOND_T_STRAIGHT = "POST_SECOND_T_STRAIGHT"
STATE_SHOOTING = "SHOOTING"
STATE_POST_SHOOT_DRIVE = "POST_SHOOT_DRIVE"
STATE_AURA = "AURA"

MODE_LINE = "LINE"
MODE_CORRIDOR = "CORRIDOR"

POST_SHOOT_PHASE_BACK_UP = "BACK_UP"
POST_SHOOT_PHASE_ALIGN = "ALIGN"
POST_SHOOT_PHASE_DRIVE_FORWARD = "DRIVE_FORWARD"

RIGHT_COLOR = (0, 0, 255)
LEFT_COLOR = (0, 255, 0)
INFRONT_COLOR = (255, 0, 0)
INFRONT_RIGHT_ONLY_COLOR = (255, 0, 255)
INFRONT_LEFT_ONLY_COLOR = (255, 255, 0)
TARGET_COLOR = (255, 0, 255)
MEMORY_COLOR = (255, 255, 255)
TURN_TRIGGER_GUIDE_COLOR = (0, 255, 255)
TURN_REACQUIRE_GUIDE_COLOR = (255, 255, 0)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def draw_low_speed_debug(image: np.ndarray, controller, origin: tuple[int, int] = (10, 95)) -> None:
    if not controller.low_speed_latched:
        return

    x, y = origin
    label = "LOW SPEED SWITCHED ON" if controller.low_speed_just_latched else "LOW SPEED ACTIVE"
    cv2.rectangle(image, (x - 4, y - 18), (x + 210, y + 8), (0, 0, 0), -1)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)


def set_leds_for_line(leds: LED | None) -> None:
    if leds is not None:
        leds.turn_on_all()


def set_leds_for_corridor(leds: LED | None) -> None:
    if leds is None:
        return
    if USE_CORRIDOR_LEDS:
        leds.turn_on_all()
    else:
        leds.turn_off_all()


def set_leds_for_shooting(leds: LED | None) -> None:
    if leds is None:
        return
    if USE_SHOOTING_LEDS:
        leds.turn_on_all()
    else:
        leds.turn_off_all()


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


def line_intersects_top_region(line: tuple[float, float], width: int, top_limit: int) -> bool:
    if top_limit <= 0:
        return False

    sample_xs = (0.0, (width - 1) / 2.0, float(width - 1))
    valid_ys = [hough_y_at_x(line, x) for x in sample_xs]
    valid_ys = [y for y in valid_ys if y is not None]
    if not valid_ys:
        return False

    if any(0.0 <= y <= float(top_limit) for y in valid_ys):
        return True

    min_y = min(valid_ys)
    max_y = max(valid_ys)
    return min_y <= float(top_limit) <= max_y


def front_line_side_presence(
    edges: np.ndarray,
    line: tuple[float, float] | None,
    *,
    samples: int = 80,
    band_radius: int = 2,
    min_hits: int = 4,
) -> str:
    if line is None:
        return "NONE"

    height, width = edges.shape[:2]
    if width <= 0 or height <= 0:
        return "NONE"
    left_limit = width / 3.0
    right_limit = (2.0 * width) / 3.0

    left_hits = 0
    center_hits = 0
    right_hits = 0
    sample_count = max(samples, 2)
    for sample_index in range(sample_count):
        x = int(round((sample_index / (sample_count - 1)) * (width - 1)))
        y_value = hough_y_at_x(line, x)
        if y_value is None:
            continue
        y = int(round(y_value))
        if not (0 <= y < height):
            continue

        x0 = max(0, x - band_radius)
        x1 = min(width, x + band_radius + 1)
        y0 = max(0, y - band_radius)
        y1 = min(height, y + band_radius + 1)
        if np.any(edges[y0:y1, x0:x1] > 0):
            if x <= left_limit:
                left_hits += 1
            elif x >= right_limit:
                right_hits += 1
            else:
                center_hits += 1

    left_present = left_hits >= min_hits
    center_present = center_hits >= min_hits
    right_present = right_hits >= min_hits
    if right_present and not left_present and not center_present:
        return "RIGHT_ONLY"
    if left_present and not right_present and not center_present:
        return "LEFT_ONLY"
    if left_present or center_present or right_present:
        return "BOTH"
    return "NONE"


def front_line_color(side_presence: str) -> tuple[int, int, int]:
    if side_presence == "RIGHT_ONLY":
        return INFRONT_RIGHT_ONLY_COLOR
    if side_presence == "LEFT_ONLY":
        return INFRONT_LEFT_ONLY_COLOR
    return INFRONT_COLOR


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
    segment = 12.0
    gap = 8.0
    offset = 0.0
    while offset < length:
        start_x = int(round(x1 + dx * offset))
        start_y = int(round(y1 + dy * offset))
        end_offset = min(offset + segment, length)
        end_x = int(round(x1 + dx * end_offset))
        end_y = int(round(y1 + dy * end_offset))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
        offset += segment + gap


def draw_line(image: np.ndarray, line: tuple[float, float], color: tuple[int, int, int], dashed: bool = False) -> None:
    pt1, pt2 = hough_line_endpoints(*line)
    if dashed:
        draw_dashed_line(image, pt1, pt2, color, 2)
    else:
        cv2.line(image, pt1, pt2, color, 2)


class CorridorVision:
    def __init__(self):
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
        self.turn_direction = DEFAULT_TURN_DIRECTION
        self.last_blue_y_center: float | None = None
        self.line_changing_warning_until = 0.0
        self.front_side_candidate = "NONE"
        self.front_side_candidate_frames = 0

    def reset_for_forward(self) -> None:
        self.turn_direction = DEFAULT_TURN_DIRECTION

    def confirmed_front_side(self, side_presence: str, line_present: bool) -> str:
        if side_presence in ("RIGHT_ONLY", "LEFT_ONLY"):
            if side_presence == self.front_side_candidate:
                self.front_side_candidate_frames += 1
            else:
                self.front_side_candidate = side_presence
                self.front_side_candidate_frames = 1

            if self.front_side_candidate_frames >= FRONT_SIDE_CONFIRM_FRAMES:
                return side_presence
            return "BOTH" if line_present else "NONE"

        self.front_side_candidate = side_presence
        self.front_side_candidate_frames = 0
        return side_presence

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
        categories = {"right": [], "left": [], "infront": []}
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

    def process(
        self,
        frame: np.ndarray,
        now: float,
        *,
        use_lateral_lines: bool = True,
        lateral_top_frac: float | None = None,
        use_front_line: bool = True,
    ) -> dict[str, object]:
        edges = self.detect_edges(frame)
        raw_lines = self.detect_hough_lines(edges)
        categories = self.classify_lines(raw_lines)
        if not use_lateral_lines:
            categories["left"] = []
            categories["right"] = []
        elif lateral_top_frac is not None:
            top_limit = int(round(H * max(0.0, min(1.0, lateral_top_frac))))
            categories["left"] = [
                line for line in categories["left"] if line_intersects_top_region(line, W, top_limit)
            ]
            categories["right"] = [
                line for line in categories["right"] if line_intersects_top_region(line, W, top_limit)
            ]
        if not use_front_line:
            categories["infront"] = []
        current_lines = {key: median_line(values) for key, values in categories.items()}
        stable_lines: dict[str, tuple[float, float] | None] = {}
        for key, line in current_lines.items():
            self.line_history[key].append(line)
            stable_lines[key] = stable_line_from_history(self.line_history[key])

        effective_lines: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
        dashed_side = {"left": False, "right": False}
        for side in ("left", "right"):
            current_line = current_lines[side]
            stable_line = stable_lines[side]
            if current_line is not None and stable_line is not None:
                self.side_memory_line[side] = stable_line
                self.side_memory_time[side] = now
                effective_lines[side] = stable_line
            else:
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
        raw_blue_line_side = front_line_side_presence(edges, blue_line)
        blue_line_side = self.confirmed_front_side(raw_blue_line_side, blue_line is not None)
        blue_y_center = hough_y_at_x(blue_line, W / 2.0) if blue_line is not None else None
        blue_tilt = line_tilt_px(blue_line, W) if blue_line is not None else None
        line_changing_detected = False
        if blue_y_center is not None and self.last_blue_y_center is not None:
            if abs(float(blue_y_center) - self.last_blue_y_center) >= HORIZONTAL_LINE_HEIGHT_JUMP_PX:
                self.line_changing_warning_until = now + LINE_CHANGING_WARNING_SEC
                line_changing_detected = True
        if blue_y_center is not None:
            self.last_blue_y_center = float(blue_y_center)
        else:
            self.last_blue_y_center = None

        return {
            "edges": edges,
            "current_lines": current_lines,
            "stable_lines": stable_lines,
            "effective_lines": effective_lines,
            "dashed_side": dashed_side,
            "vanishing_point": vanishing_point,
            "blue_line": blue_line,
            "blue_line_side": blue_line_side,
            "blue_line_color": front_line_color(blue_line_side),
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
            "turn_trigger": (
                blue_y_center is not None and blue_y_center >= (H * (1.0 - TURN_HEIGHT_FRAC))
            ),
            "blue_top_visible": (
                blue_y_center is not None and blue_y_center <= (H * TURN_REACQUIRE_TOP_FRAC)
            ),
            "current_side_count": int(current_lines["left"] is not None) + int(current_lines["right"] is not None),
            "turn_direction": self.turn_direction,
            "line_changing_detected": line_changing_detected,
            "line_changing_warning_active": now < self.line_changing_warning_until,
        }


class CorridorController:
    def __init__(self):
        self.vision = CorridorVision()
        self.state = "FORWARD"
        self.turn_phase = 0
        self.turn_direction = DEFAULT_TURN_DIRECTION
        self.command_left = 0.0
        self.command_right = 0.0
        self.blind_until = 0.0
        self.turn_deadline = 0.0
        self.search_turn_direction = "RIGHT"
        self.search_turn_until = 0.0
        self.line_change_turn_until = 0.0

    def reset(self) -> None:
        self.state = "FORWARD"
        self.turn_phase = 0
        self.turn_direction = DEFAULT_TURN_DIRECTION
        self.command_left = 0.0
        self.command_right = 0.0
        self.blind_until = 0.0
        self.turn_deadline = 0.0
        self.search_turn_direction = "RIGHT"
        self.search_turn_until = 0.0
        self.line_change_turn_until = 0.0
        self.vision.reset_for_forward()

    def forward_wheels(self, correction: float) -> tuple[float, float]:
        base_left = FORWARD_BASE_SPEED * FORWARD_LEFT_SCALE
        base_right = FORWARD_BASE_SPEED * FORWARD_RIGHT_SCALE
        left = clamp(base_left + correction, -1.0, 1.0)
        right = clamp(base_right - correction, -1.0, 1.0)
        return left, right

    def blind_forward_wheels(self) -> tuple[float, float]:
        speed = clamp(BLIND_FORWARD_SPEED, -1.0, 1.0)
        return speed, speed

    def no_line_search_wheels(self) -> tuple[float, float]:
        turn_sign = TURN_SIGN * (1 if self.search_turn_direction == "RIGHT" else -1)
        speed = clamp(NO_LINE_SEARCH_TURN_SPEED, 0.0, 1.0)
        return turn_sign * speed, -turn_sign * speed

    def single_wall_search_wheels(self, visible_side: str) -> tuple[float, float]:
        if visible_side == "left":
            return self.forward_wheels(SINGLE_WALL_SEARCH_DELTA)
        if visible_side == "right":
            return self.forward_wheels(-SINGLE_WALL_SEARCH_DELTA)
        return self.forward_wheels(0.0)

    def update(
        self,
        frame: np.ndarray,
        now: float,
        *,
        lateral_top_frac: float | None = None,
        use_front_line: bool = True,
    ) -> tuple[float, float, np.ndarray, np.ndarray, str]:
        vision = self.vision.process(
            frame,
            now,
            use_lateral_lines=(self.state != "TURN"),
            lateral_top_frac=lateral_top_frac,
            use_front_line=use_front_line,
        )
        _, blue_targets = detect_blue_targets(
            frame,
            blue_low=BLUE_LOW_CORRIDOR,
            blue_high=BLUE_HIGH_CORRIDOR,
            min_target_area=MIN_TARGET_AREA_CORRIDOR,
            min_target_width=MIN_TARGET_WIDTH_CORRIDOR,
        )
        blue_targets_seen = bool(blue_targets)
        if blue_targets_seen:
            self.line_change_turn_until = 0.0
        current_any_line = (
            vision["current_lines"]["left"] is not None
            or vision["current_lines"]["right"] is not None
            or vision["current_lines"]["infront"] is not None
        )

        if self.state == "FORWARD":
            if bool(vision["line_changing_detected"]) and not blue_targets_seen:
                self.line_change_turn_until = now + LINE_CHANGING_RIGHT_TURN_SEC

            if self.line_change_turn_until > now and not blue_targets_seen:
                turn_sign = TURN_SIGN * 1
                self.command_left = turn_sign * TURN_SPEED
                self.command_right = -turn_sign * TURN_SPEED
            elif bool(vision["turn_trigger"]):
                self.state = "TURN"
                self.turn_phase = 1
                self.turn_direction = str(vision["turn_direction"])
                self.vision.reset_for_forward()
                turn_sign = TURN_SIGN * (1 if self.turn_direction == "RIGHT" else -1)
                self.command_left = turn_sign * TURN_SPEED
                self.command_right = -turn_sign * TURN_SPEED
                self.blind_until = now + TURN_BLIND_SEC
                self.turn_deadline = now + TURN_DURATION_SEC
            elif not current_any_line:
                self.state = "SEARCH_LINES"
                self.search_turn_direction = "RIGHT"
                self.search_turn_until = now + NO_LINE_SEARCH_TURN_SEC
                self.command_left, self.command_right = self.no_line_search_wheels()
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
                    elif int(vision["current_side_count"]) == 0 and vision["blue_line_side"] in ("RIGHT_ONLY", "LEFT_ONLY"):
                        direction = 1.0 if vision["blue_line_side"] == "RIGHT_ONLY" else -1.0
                        correction = clamp(direction * MAX_STEER_DELTA, -MAX_STEER_DELTA, MAX_STEER_DELTA)
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

                if correction is not None:
                    self.command_left, self.command_right = self.forward_wheels(correction)
                elif visible_side is not None:
                    self.command_left, self.command_right = self.single_wall_search_wheels(visible_side)
                else:
                    self.command_left, self.command_right = self.blind_forward_wheels()

        elif self.state == "TURN":
            turn_sign = TURN_SIGN * (1 if self.turn_direction == "RIGHT" else -1)
            self.command_left = turn_sign * TURN_SPEED
            self.command_right = -turn_sign * TURN_SPEED

            if now >= self.blind_until and bool(vision["blue_top_visible"]):
                self.state = "FORWARD"
                self.turn_phase = 0
                self.command_left = 0.0
                self.command_right = 0.0
                self.vision.reset_for_forward()
            elif now >= self.turn_deadline:
                if self.turn_phase == 1:
                    self.turn_phase = 2
                    self.turn_direction = "LEFT" if self.turn_direction == "RIGHT" else "RIGHT"
                    turn_sign = TURN_SIGN * (1 if self.turn_direction == "RIGHT" else -1)
                    self.command_left = turn_sign * TURN_SPEED
                    self.command_right = -turn_sign * TURN_SPEED
                    self.blind_until = now + TURN_BLIND_SEC + TURN_RETRY_BLIND_EXTRA_SEC
                    self.turn_deadline = now + (TURN_DURATION_SEC * TURN_RETRY_DURATION_MULTIPLIER)
                else:
                    self.turn_phase = 2
                    self.turn_direction = "LEFT" if self.turn_direction == "RIGHT" else "RIGHT"
                    turn_sign = TURN_SIGN * (1 if self.turn_direction == "RIGHT" else -1)
                    self.command_left = turn_sign * TURN_SPEED
                    self.command_right = -turn_sign * TURN_SPEED
                    self.blind_until = now + TURN_BLIND_SEC + TURN_RETRY_BLIND_EXTRA_SEC
                    self.turn_deadline = now + (TURN_DURATION_SEC * TURN_RETRY_DURATION_MULTIPLIER)
        elif self.state == "SEARCH_LINES":
            if current_any_line:
                self.state = "FORWARD"
                self.command_left = 0.0
                self.command_right = 0.0
                self.search_turn_until = 0.0
            else:
                if now >= self.search_turn_until:
                    self.search_turn_direction = "LEFT" if self.search_turn_direction == "RIGHT" else "RIGHT"
                    self.search_turn_until = now + NO_LINE_SEARCH_TURN_SEC
                self.command_left, self.command_right = self.no_line_search_wheels()
        else:
            self.command_left = 0.0
            self.command_right = 0.0

        debug = self.draw_debug(frame, vision)
        aux = cv2.cvtColor(vision["edges"], cv2.COLOR_GRAY2BGR)
        return self.command_left, self.command_right, debug, aux, self.state

    def draw_debug(self, frame: np.ndarray, vision: dict[str, object]) -> np.ndarray:
        overlay = frame.copy()
        stable_lines = vision["stable_lines"]
        effective_lines = vision["effective_lines"]
        dashed_side = vision["dashed_side"]
        turn_trigger_y = int(round(H * (1.0 - TURN_HEIGHT_FRAC)))
        turn_reacquire_y = int(round(H * TURN_REACQUIRE_TOP_FRAC))

        if stable_lines["right"] is not None:
            draw_line(overlay, stable_lines["right"], RIGHT_COLOR, dashed=bool(dashed_side["right"]))
        elif effective_lines["right"] is not None:
            draw_line(overlay, effective_lines["right"], MEMORY_COLOR, dashed=True)

        if stable_lines["left"] is not None:
            draw_line(overlay, stable_lines["left"], LEFT_COLOR, dashed=bool(dashed_side["left"]))
        elif effective_lines["left"] is not None:
            draw_line(overlay, effective_lines["left"], MEMORY_COLOR, dashed=True)

        if stable_lines["infront"] is not None:
            draw_line(overlay, stable_lines["infront"], vision["blue_line_color"], dashed=False)

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

        cv2.line(overlay, (0, turn_trigger_y), (W - 1, turn_trigger_y), TURN_TRIGGER_GUIDE_COLOR, 1)
        cv2.line(overlay, (0, turn_reacquire_y), (W - 1, turn_reacquire_y), TURN_REACQUIRE_GUIDE_COLOR, 1)
        cv2.putText(
            overlay,
            "Turn trigger line",
            (10, max(12, turn_trigger_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            TURN_TRIGGER_GUIDE_COLOR,
            1,
        )
        cv2.putText(
            overlay,
            "Turn end line",
            (10, max(12, turn_reacquire_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            TURN_REACQUIRE_GUIDE_COLOR,
            1,
        )

        cv2.putText(overlay, f"FSM mode: {MODE_CORRIDOR}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"Corridor state: {self.state}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"Turn direction var: {vision['turn_direction']}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, f"Cmd L/R: {self.command_left:.2f} / {self.command_right:.2f}", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if bool(vision.get("line_changing_warning_active", False)):
            cv2.putText(
                overlay,
                "LINE CHANGING",
                (18, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                4,
            )
        return overlay


def setup_windows() -> None:
    display_width = W * WINDOW_SCALE
    display_height = H * WINDOW_SCALE
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(AUX_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, display_width, display_height)
    cv2.resizeWindow(AUX_WINDOW, display_width, display_height)
    cv2.moveWindow(MAIN_WINDOW, 40, 40)
    cv2.moveWindow(AUX_WINDOW, 60 + display_width, 40)


def build_recording_frame(main_frame: np.ndarray, aux_frame: np.ndarray | None) -> np.ndarray:
    if aux_frame is None:
        aux_vis = np.zeros_like(main_frame)
    elif len(aux_frame.shape) == 2:
        aux_vis = cv2.cvtColor(aux_frame, cv2.COLOR_GRAY2BGR)
    else:
        aux_vis = aux_frame

    target_height = main_frame.shape[0]
    target_width = main_frame.shape[1]
    aux_resized = cv2.resize(aux_vis, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    combined = np.hstack((main_frame, aux_resized))
    cv2.putText(combined, "Main", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(combined, "Aux", (target_width + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return combined


def build_t_blue_debug(line_res, blue_mask: np.ndarray | None) -> np.ndarray:
    t_vis = draw_mask_slices(line_res)
    t_vis = cv2.resize(t_vis, (W, H), interpolation=cv2.INTER_NEAREST)

    if blue_mask is None:
        blue_vis = np.zeros((H, W, 3), dtype=np.uint8)
    elif len(blue_mask.shape) == 2:
        blue_vis = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    else:
        blue_vis = blue_mask.copy()

    blue_vis = cv2.resize(blue_vis, (W, H), interpolation=cv2.INTER_NEAREST)
    combined = np.hstack((t_vis, blue_vis))
    cv2.putText(combined, "T mask (ROI only)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(combined, "Blue mask (full frame)", (W + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
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


def fail_stop(motors: MotorDriver, leds: LED | None, aura_started: bool) -> bool:
    motors.stop()
    if not USE_LEDS or leds is None:
        return aura_started
    if not aura_started:
        leds.aura_farm_start()
        return True
    leds.aura_farm_step(delay=AURA_FARM_DELAY)
    return aura_started


def main() -> None:
    cam = Camera()
    line_controller = Controller()
    second_line_controller = Controller(
        base_speed=0.25,
        turn_speed=0.25,
        blind_speed=0.20,
        search_speed=0.20,
        steer_gain=0.10,
    )
    post_shoot_line_controller = Controller(
        base_speed=0.22,
        turn_speed=0.20,
        blind_speed=0.18,
        search_speed=0.18,
        steer_gain=0.10,
    )
    corridor_controller = CorridorController()
    motors = MotorDriver()
    leds = LED() if USE_LEDS else None
    music_player = MusicPlayer(
        enabled=USE_MUSIC,
        backend=MUSIC_BACKEND,
        sink=MUSIC_SINK,
        debug=MUSIC_DEBUG,
    )
    shooter = Shooter()
    video_writer: cv2.VideoWriter | None = None

    def _reload_ammo_on_exit(*_args) -> None:
        try:
            shooter.cleanup()
        except Exception:
            pass

    atexit.register(_reload_ammo_on_exit)
    signal.signal(signal.SIGINT, _reload_ammo_on_exit)
    signal.signal(signal.SIGTERM, _reload_ammo_on_exit)

    shooter.reset_ammo()
    fsm_state = STATE_RELOAD
    aura_started = False
    post_t_until = 0.0
    post_second_t_until = 0.0
    post_shoot_back_until = 0.0
    post_shoot_until = 0.0
    post_shoot_phase = POST_SHOOT_PHASE_BACK_UP

    lock_frames = 0
    last_target_center = None
    locked_target_ref = None
    locked_target_miss_count = 0
    pending_shot = False
    pending_target_before_shot = None
    confirm_delay_frames = 0
    targets_hit = 0
    blue_detect_frames = 0
    post_shoot_turn_sign = 1
    aim_stuck_since = 0.0
    aim_last_motion_center: tuple[int, int] | None = None
    aim_boost_until = 0.0
    shoot_search_direction = 1
    shoot_search_step = 0
    shoot_search_until = 0.0

    if SHOW_DEBUG:
        setup_windows()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            now = time.monotonic()
            line_res = detect_line(frame, angle_deg=20)

            main_vis = frame.copy()
            aux_vis = np.zeros((H, W, 3), dtype=np.uint8)

            if fsm_state == STATE_RELOAD:
                motors.stop()
                shooter.spin_flywheel(False)
                set_leds_for_line(leds)
                main_vis = frame.copy()
                cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, "Press R to start run", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"AMMO: {shooter.ammo_remaining()}/{MAG_CAPACITY}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                aux_vis = np.zeros((H, W, 3), dtype=np.uint8)

            elif fsm_state == STATE_LINE_FOLLOW:
                left, right, _, _, line_mode = line_controller.update(line_res)
                if line_mode == "T_STOP":
                    corridor_controller.reset()
                    motors.set_speeds(FIRST_T_STRAIGHT_SPEED, FIRST_T_STRAIGHT_SPEED)
                    post_t_until = now + FIRST_T_STRAIGHT_SEC
                    fsm_state = STATE_POST_T_STRAIGHT
                elif line_res.t_detected:
                    motors.set_speeds(FIRST_T_STRAIGHT_SPEED, FIRST_T_STRAIGHT_SPEED)
                    set_leds_for_line(leds)
                    aura_started = False
                    main_vis = draw_debug_slices(frame, line_res, BAND_THICKNESS)
                    aux_vis = draw_mask_slices(line_res)
                    cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Nav mode: {MODE_LINE}", (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, "Line mode: T_CONFIRM_STRAIGHT", (10, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    draw_low_speed_debug(main_vis, line_controller)
                    if line_controller.low_speed_just_latched:
                        print("LOW SPEED SWITCHED ON")
                else:
                    motors.set_speeds(left, right)
                    set_leds_for_line(leds)
                    aura_started = False
                    main_vis = draw_debug_slices(frame, line_res, BAND_THICKNESS)
                    aux_vis = draw_mask_slices(line_res)
                    cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Nav mode: {MODE_LINE}", (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Line mode: {line_mode}", (10, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    draw_low_speed_debug(main_vis, line_controller)
                    if line_controller.low_speed_just_latched:
                        print("LOW SPEED SWITCHED ON")

            elif fsm_state == STATE_POST_T_STRAIGHT:
                left, right, main_vis, aux_vis, corridor_state = corridor_controller.update(
                    frame,
                    now,
                    lateral_top_frac=FIRST_T_SIDE_LINES_TOP_FRAC,
                    use_front_line=False,
                )
                motors.set_speeds(left, right)
                set_leds_for_corridor(leds)
                remaining = max(0.0, post_t_until - now)
                cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"Top-third side-line mode: {remaining:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(main_vis, f"Corridor substate: {corridor_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if now >= post_t_until:
                    music_player.play("couloir.wav")
                    fsm_state = STATE_CORRIDOR_FOLLOW

            elif fsm_state == STATE_CORRIDOR_FOLLOW:
                blue_mask, blue_targets = detect_blue_targets(
                    frame,
                    blue_low=BLUE_LOW_CORRIDOR,
                    blue_high=BLUE_HIGH_CORRIDOR,
                    min_target_area=MIN_TARGET_AREA_CORRIDOR,
                    min_target_width=MIN_TARGET_WIDTH_CORRIDOR,
                )
                if blue_targets:
                    blue_detect_frames += 1
                    if blue_detect_frames >= BLUE_DETECT_CONFIRM_FRAMES:
                        motors.stop()
                        fsm_state = STATE_BLUE_TO_T
                        main_vis = frame.copy()
                        for target in blue_targets:
                            x, y, w, h = target["bbox"]
                            cv2.rectangle(main_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.circle(main_vis, (target["cx"], target["cy"]), 4, (0, 0, 255), -1)
                        cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        cv2.putText(main_vis, "Blue detected: turning right to find T", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        aux_vis = build_t_blue_debug(line_res, blue_mask)
                    else:
                        left, right, main_vis, aux_vis, corridor_state = corridor_controller.update(frame, now)
                        motors.set_speeds(left, right)
                        set_leds_for_corridor(leds)
                        aura_started = False
                        aux_vis = build_t_blue_debug(line_res, blue_mask)
                        cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                        cv2.putText(main_vis, f"Nav mode: {MODE_CORRIDOR}", (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                        cv2.putText(main_vis, f"Blue targets: {len(blue_targets)}", (10, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                        cv2.putText(main_vis, f"Blue confirm: {blue_detect_frames}/{BLUE_DETECT_CONFIRM_FRAMES}", (10, 199), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                else:
                    blue_detect_frames = 0
                    left, right, main_vis, aux_vis, corridor_state = corridor_controller.update(frame, now)
                    motors.set_speeds(left, right)
                    set_leds_for_corridor(leds)
                    aura_started = False
                    aux_vis = build_t_blue_debug(line_res, blue_mask)
                    cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Nav mode: {MODE_CORRIDOR}", (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Blue targets: {len(blue_targets)}", (10, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Blue confirm: {blue_detect_frames}/{BLUE_DETECT_CONFIRM_FRAMES}", (10, 199), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            elif fsm_state == STATE_BLUE_TO_T:
                blue_mask, blue_targets = detect_blue_targets(
                    frame,
                    blue_low=BLUE_LOW_CORRIDOR,
                    blue_high=BLUE_HIGH_CORRIDOR,
                    min_target_area=MIN_TARGET_AREA_CORRIDOR,
                    min_target_width=MIN_TARGET_WIDTH_CORRIDOR,
                )
                main_vis = frame.copy()
                for target in blue_targets:
                    x, y, w, h = target["bbox"]
                    cv2.rectangle(main_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(main_vis, (target["cx"], target["cy"]), 4, (0, 0, 255), -1)
                cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                action_label = "Action: right turn while blue is visible" if blue_targets else "Action: left turn until blue or line is found"
                cv2.putText(main_vis, action_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(main_vis, f"Blue targets: {len(blue_targets)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(main_vis, f"Line seen this frame: {'YES' if line_res.any_seen else 'NO'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                aux_vis = build_t_blue_debug(line_res, blue_mask)

                if line_res.any_seen:
                    motors.stop()
                    set_leds_for_line(leds)
                    second_line_controller.reset()
                    fsm_state = STATE_LINE_TO_SECOND_T
                else:
                    turn_sign = TURN_SIGN * (1.0 if blue_targets else -1.0)
                    motors.set_speeds(
                        turn_sign * BLUE_TO_T_RIGHT_TURN_SPEED,
                        -turn_sign * BLUE_TO_T_RIGHT_TURN_SPEED,
                    )
                    set_leds_for_corridor(leds)
                    aura_started = False

            elif fsm_state == STATE_LINE_TO_SECOND_T:
                blue_mask, blue_targets = detect_blue_targets(
                    frame,
                    blue_low=BLUE_LOW_CORRIDOR,
                    blue_high=BLUE_HIGH_CORRIDOR,
                    min_target_area=MIN_TARGET_AREA_CORRIDOR,
                    min_target_width=MIN_TARGET_WIDTH_CORRIDOR,
                )
                left, right, _, _, line_mode = second_line_controller.update(line_res)
                post_shoot_turn_sign = second_line_controller.last_turn_sign
                if second_line_controller.t_confirmed:
                    motors.set_speeds(T_LIM_SPEED, T_LIM_SPEED)
                    post_second_t_until = now + T_LIM_TIME
                    fsm_state = STATE_POST_SECOND_T_STRAIGHT
                elif line_res.t_detected:
                    motors.set_speeds(T_LIM_SPEED, T_LIM_SPEED)
                    set_leds_for_line(leds)
                    aura_started = False
                    main_vis = draw_debug_slices(frame, line_res, BAND_THICKNESS)
                    cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Nav mode: {MODE_LINE}", (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, "Line mode: T_CONFIRM_STRAIGHT", (10, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Blue targets: {len(blue_targets)}", (10, 199), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    draw_low_speed_debug(main_vis, second_line_controller)
                    if second_line_controller.low_speed_just_latched:
                        print("LOW SPEED SWITCHED ON")
                    aux_vis = build_t_blue_debug(line_res, blue_mask)
                else:
                    motors.set_speeds(left, right)
                    set_leds_for_line(leds)
                    aura_started = False
                    main_vis = draw_debug_slices(frame, line_res, BAND_THICKNESS)
                    cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Nav mode: {MODE_LINE}", (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Line mode: {line_mode}", (10, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.putText(main_vis, f"Blue targets: {len(blue_targets)}", (10, 199), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    draw_low_speed_debug(main_vis, second_line_controller)
                    if second_line_controller.low_speed_just_latched:
                        print("LOW SPEED SWITCHED ON")
                    aux_vis = build_t_blue_debug(line_res, blue_mask)

            elif fsm_state == STATE_POST_SECOND_T_STRAIGHT:
                motors.set_speeds(T_LIM_SPEED, T_LIM_SPEED)
                set_leds_for_line(leds)
                main_vis = frame.copy()
                remaining = max(0.0, post_second_t_until - now)
                cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"Straight before shooting: {remaining:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                aux_vis = build_t_blue_debug(line_res, None)
                if now >= post_second_t_until:
                    motors.stop()
                    music_player.play("tir.wav")
                    shooter.spin_flywheel(True)
                    time.sleep(FLYWHEEL_SPINUP_TIME)
                    fsm_state = STATE_SHOOTING
                    lock_frames = 0
                    last_target_center = None
                    locked_target_ref = None
                    locked_target_miss_count = 0
                    pending_shot = False
                    pending_target_before_shot = None
                    confirm_delay_frames = 0
                    aim_stuck_since = 0.0
                    aim_last_motion_center = None
                    aim_boost_until = 0.0
                    shoot_search_direction = 1
                    shoot_search_step = 0
                    shoot_search_until = 0.0

            elif fsm_state == STATE_SHOOTING:
                set_leds_for_shooting(leds)
                blue_mask, targets = detect_blue_targets(frame)

                if pending_shot:
                    aim_stuck_since = 0.0
                    aim_last_motion_center = None
                    aim_boost_until = 0.0
                    shoot_search_until = 0.0
                    if confirm_delay_frames > 0:
                        confirm_delay_frames -= 1
                        motors.set_speeds(0, 0)
                    else:
                        old = pending_target_before_shot
                        matched, dist = match_target_by_position(
                            targets,
                            old["cx"],
                            old["cy"],
                            TARGET_MATCH_DIST_PX,
                        )
                        hit_confirmed = False
                        if matched is None or dist > TARGET_MATCH_DIST_PX:
                            hit_confirmed = True
                        elif matched["area"] < HIT_AREA_RATIO_THRESHOLD * old["area"]:
                            hit_confirmed = True
                        if hit_confirmed:
                            targets_hit += 1
                            last_target_center = None
                            locked_target_ref = None
                            locked_target_miss_count = 0
                        elif matched is not None:
                            last_target_center = (matched["cx"], matched["cy"])
                            locked_target_ref = {
                                "cx": matched["cx"],
                                "cy": matched["cy"],
                                "area": matched["area"],
                            }
                            locked_target_miss_count = 0
                        pending_shot = False
                        pending_target_before_shot = None
                        lock_frames = 0
                else:
                    target = None
                    if locked_target_ref is not None:
                        target, dist = match_target_by_position(
                            targets,
                            locked_target_ref["cx"],
                            locked_target_ref["cy"],
                            TARGET_MATCH_DIST_PX * 3,
                        )
                        if target is None or dist > TARGET_MATCH_DIST_PX * 3:
                            target = None
                            locked_target_miss_count += 1
                            if locked_target_miss_count >= TARGET_LOCK_MISS_FRAMES:
                                locked_target_ref = None
                                locked_target_miss_count = 0
                        else:
                            locked_target_miss_count = 0
                            locked_target_ref = {
                                "cx": target["cx"],
                                "cy": target["cy"],
                                "area": target["area"],
                            }
                            shoot_search_until = 0.0

                    if target is None:
                        if locked_target_ref is not None:
                            lock_frames = 0
                            aim_stuck_since = 0.0
                            aim_last_motion_center = None
                            aim_boost_until = 0.0
                        else:
                            target = pick_target_by_shot_order(targets, targets_hit)
                            if target is not None:
                                locked_target_ref = {
                                    "cx": target["cx"],
                                    "cy": target["cy"],
                                    "area": target["area"],
                                }
                                locked_target_miss_count = 0
                                shoot_search_until = 0.0

                    if target is None:
                        lock_frames = 0
                        last_target_center = None
                        aim_stuck_since = 0.0
                        aim_last_motion_center = None
                        aim_boost_until = 0.0
                        if now >= shoot_search_until:
                            search_duration = SHOOT_SEARCH_BASE_SEC + (shoot_search_step * SHOOT_SEARCH_SEC_INCREMENT)
                            shoot_search_direction = 1 if (shoot_search_step % 2) == 0 else -1
                            shoot_search_until = now + search_duration
                            shoot_search_step += 1
                        turn_sign = 1.0 if shoot_search_direction > 0 else -1.0
                        motors.set_speeds(
                            turn_sign * SHOOT_SEARCH_TURN_SPEED,
                            -turn_sign * SHOOT_SEARCH_TURN_SPEED,
                        )
                    else:
                        shoot_search_until = 0.0
                        shoot_search_direction = 1
                        shoot_search_step = 0
                        last_target_center = (target["cx"], target["cy"])
                        locked, err = aim_to_target(motors, target)
                        current_center = (target["cx"], target["cy"])
                        should_move = not locked

                        if should_move:
                            if aim_last_motion_center is None:
                                aim_last_motion_center = current_center
                                aim_stuck_since = now
                            else:
                                delta_x = abs(current_center[0] - aim_last_motion_center[0])
                                delta_y = abs(current_center[1] - aim_last_motion_center[1])
                                if delta_x > SHOOT_STUCK_TARGET_DELTA_PX or delta_y > SHOOT_STUCK_TARGET_DELTA_PX:
                                    aim_last_motion_center = current_center
                                    aim_stuck_since = now
                                elif aim_stuck_since <= 0.0:
                                    aim_stuck_since = now

                            if now >= aim_boost_until and aim_stuck_since > 0.0 and (now - aim_stuck_since) >= SHOOT_STUCK_TIMEOUT_SEC:
                                boost_sign = 1.0 if err > 0 else -1.0
                                motors.set_speeds(
                                    boost_sign * SHOOT_STUCK_BOOST_SPEED,
                                    -boost_sign * SHOOT_STUCK_BOOST_SPEED,
                                )
                                aim_boost_until = now + SHOOT_STUCK_BOOST_SEC
                                aim_stuck_since = now

                            if now < aim_boost_until:
                                boost_sign = 1.0 if err > 0 else -1.0
                                motors.set_speeds(
                                    boost_sign * SHOOT_STUCK_BOOST_SPEED,
                                    -boost_sign * SHOOT_STUCK_BOOST_SPEED,
                                )
                        else:
                            aim_stuck_since = 0.0
                            aim_last_motion_center = current_center
                            aim_boost_until = 0.0

                        lock_frames = lock_frames + 1 if locked else 0
                        if lock_frames >= LOCK_FRAMES_REQUIRED:
                            fired = shooter.fire_once()
                            lock_frames = 0
                            if fired:
                                pending_shot = True
                                pending_target_before_shot = {
                                    "cx": target["cx"],
                                    "cy": target["cy"],
                                    "area": target["area"],
                                }
                                confirm_delay_frames = POST_SHOT_CONFIRM_FRAMES
                                aim_stuck_since = 0.0
                                aim_last_motion_center = None
                                aim_boost_until = 0.0
                                time.sleep(POST_SHOT_SETTLE_TIME)

                mission_done = (targets_hit >= NOMBRE_DE_CIBLES) or shooter.is_empty()
                if mission_done and not pending_shot:
                    motors.set_speeds(0, 0)
                    shooter.spin_flywheel(False)
                    set_leds_for_line(leds)
                    time.sleep(0.2)
                    post_shoot_back_until = now + POST_SHOOT_BACK_SEC
                    post_shoot_phase = POST_SHOOT_PHASE_BACK_UP
                    post_shoot_line_controller.reset()
                    post_shoot_line_controller.last_turn_sign = post_shoot_turn_sign
                    post_shoot_line_controller.last_green_turn_sign = post_shoot_turn_sign
                    fsm_state = STATE_POST_SHOOT_DRIVE

                main_vis = frame.copy()
                for t in targets:
                    x, y, w, h = t["bbox"]
                    cv2.rectangle(main_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(main_vis, (t["cx"], t["cy"]), 4, (0, 0, 255), -1)
                cv2.line(main_vis, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
                cv2.line(main_vis, (W // 2 - CENTER_TOL, 0), (W // 2 - CENTER_TOL, H), (0, 255, 255), 1)
                cv2.line(main_vis, (W // 2 + CENTER_TOL, 0), (W // 2 + CENTER_TOL, H), (0, 255, 255), 1)
                cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"HITS: {targets_hit}/{NOMBRE_DE_CIBLES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"AMMO: {shooter.ammo_remaining()}/{MAG_CAPACITY}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"PENDING: {pending_shot}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"AIM BOOST: {'ON' if now < aim_boost_until else 'OFF'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                if not pending_shot and not targets:
                    direction_label = "RIGHT" if shoot_search_direction > 0 else "LEFT"
                    remaining = max(0.0, shoot_search_until - now)
                    cv2.putText(main_vis, f"SEARCH: {direction_label} {remaining:.2f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                aux_vis = blue_mask

            elif fsm_state == STATE_POST_SHOOT_DRIVE:
                set_leds_for_line(leds)
                main_vis = draw_debug_slices(frame, line_res, BAND_THICKNESS)
                direction_label = "RIGHT" if post_shoot_turn_sign > 0 else "LEFT"
                action_label = ""
                status_label = ""

                if post_shoot_phase == POST_SHOOT_PHASE_BACK_UP:
                    motors.set_speeds(POST_SHOOT_BACK_SPEED, POST_SHOOT_BACK_SPEED)
                    remaining = max(0.0, post_shoot_back_until - now)
                    action_label = "Action: backing up before line reacquire"
                    status_label = f"Reverse for: {remaining:.2f}s"
                    aux_vis = draw_mask_slices(line_res)
                    if now >= post_shoot_back_until:
                        motors.stop()
                        post_shoot_phase = POST_SHOOT_PHASE_ALIGN
                        post_shoot_line_controller.reset()
                        post_shoot_line_controller.last_turn_sign = post_shoot_turn_sign
                        post_shoot_line_controller.last_green_turn_sign = post_shoot_turn_sign
                elif post_shoot_phase == POST_SHOOT_PHASE_ALIGN:
                    left, right, _, _, line_mode = post_shoot_line_controller.update(line_res)
                    motors.set_speeds(left, right)
                    action_label = "Action: line follow until T is detected"
                    status_label = f"Line mode: {line_mode} | T: {'YES' if line_res.t_detected else 'NO'}"
                    aux_vis = draw_mask_slices(line_res)
                    if line_res.t_detected:
                        motors.set_speeds(POST_SHOOT_DRIVE_SPEED, POST_SHOOT_DRIVE_SPEED)
                        post_shoot_until = now + POST_SHOOT_DRIVE_SEC
                        post_shoot_phase = POST_SHOOT_PHASE_DRIVE_FORWARD
                else:
                    motors.set_speeds(POST_SHOOT_DRIVE_SPEED, POST_SHOOT_DRIVE_SPEED)
                    remaining = max(0.0, post_shoot_until - now)
                    action_label = "Action: straight drive"
                    status_label = f"Drive after shooting: {remaining:.2f}s"
                    aux_vis = draw_mask_slices(line_res)
                    if now >= post_shoot_until:
                        motors.stop()
                        fsm_state = STATE_AURA

                cv2.putText(main_vis, f"FSM: {fsm_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(main_vis, f"Post-shoot phase: {post_shoot_phase}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(main_vis, status_label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(main_vis, f"Remembered line direction: {direction_label}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(main_vis, action_label, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                if post_shoot_phase == POST_SHOOT_PHASE_ALIGN:
                    draw_low_speed_debug(main_vis, post_shoot_line_controller)
                    if post_shoot_line_controller.low_speed_just_latched:
                        print("LOW SPEED SWITCHED ON")

            elif fsm_state == STATE_AURA:
                aura_started = fail_stop(motors, leds, aura_started)
                main_vis = frame.copy()
                cv2.putText(main_vis, "FSM: AURA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(main_vis, f"HITS: {targets_hit}/{NOMBRE_DE_CIBLES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                aux_vis = np.zeros((H, W, 3), dtype=np.uint8)

            recording_frame = build_recording_frame(main_vis, aux_vis)
            if RECORD_RUN_VIDEO:
                if video_writer is None:
                    frame_size = (recording_frame.shape[1], recording_frame.shape[0])
                    video_writer = create_run_video_writer(frame_size)
                if video_writer is not None:
                    video_writer.write(recording_frame)

            if SHOW_DEBUG:
                cv2.imshow(MAIN_WINDOW, main_vis)
                cv2.imshow(AUX_WINDOW, aux_vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r") and fsm_state == STATE_RELOAD:
                    shooter.reset_ammo()
                    line_controller.reset()
                    second_line_controller.reset()
                    post_shoot_line_controller.reset()
                    corridor_controller.reset()
                    aura_started = False
                    post_t_until = 0.0
                    post_second_t_until = 0.0
                    post_shoot_back_until = 0.0
                    post_shoot_until = 0.0
                    post_shoot_phase = POST_SHOOT_PHASE_BACK_UP
                    lock_frames = 0
                    last_target_center = None
                    locked_target_ref = None
                    locked_target_miss_count = 0
                    pending_shot = False
                    pending_target_before_shot = None
                    confirm_delay_frames = 0
                    targets_hit = 0
                    blue_detect_frames = 0
                    post_shoot_turn_sign = 1
                    aim_stuck_since = 0.0
                    aim_last_motion_center = None
                    aim_boost_until = 0.0
                    shoot_search_direction = 1
                    shoot_search_step = 0
                    shoot_search_until = 0.0
                    fsm_state = STATE_LINE_FOLLOW
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
            music_player.stop()
        except Exception:
            pass
        try:
            shooter.cleanup()
        except Exception:
            pass
        try:
            atexit.unregister(_reload_ammo_on_exit)
        except Exception:
            pass
        try:
            cam.release()
        except Exception:
            pass
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
