from __future__ import annotations

from config import (
    BASE_SPEED,
    BLIND_SPEED,
    BLIND_TURN_FRAMES,
    BLIND_TURN_STEER,
    LOW_SPEED,
    SEARCH_FRAMES,
    SEARCH_SPEED,
    SEARCH_TURN_STEER,
    STEER_DEADZONE,
    STEER_GAIN,
    STEER_SMOOTH_ALPHA,
    T_CONFIRM_FRAMES,
    T_LINE_FRAMES,
    TOP_SLICE_LOW_WIDTH_FRAMES,
    TOP_SLICE_LOW_WIDTH_THRESHOLD,
    TURN_SPEED,
    W,
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def smooth(prev: float, target: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * target


def normalize_error(err_px: float) -> float:
    half = max(W / 2, 1)
    return clamp(err_px / half, -1.0, 1.0)


def steering_to_wheels(speed: float, steer: float):
    steer = clamp(steer, -1.0, 1.0)
    left = speed + STEER_GAIN * steer
    right = speed - STEER_GAIN * steer
    return clamp(left, -1.0, 1.0), clamp(right, -1.0, 1.0)


class Controller:
    def __init__(
        self,
        *,
        base_speed: float = BASE_SPEED,
        low_speed: float = LOW_SPEED,
        turn_speed: float = TURN_SPEED,
        blind_speed: float = BLIND_SPEED,
        search_speed: float = SEARCH_SPEED,
        steer_gain: float = STEER_GAIN,
    ):
        self.base_speed = base_speed
        self.low_speed = low_speed
        self.turn_speed = turn_speed
        self.blind_speed = blind_speed
        self.search_speed = search_speed
        self.steer_gain = steer_gain
        self.reset()

    def reset(self):
        self.mode = "TRACK"
        self.steer_cmd = 0.0
        self.last_turn_sign = 1
        self.last_green_turn_sign = 1
        self.lost_count = 0
        self.t_count = 0
        self.t_confirmed = False
        self.t_line_count = 0
        self.t_stop_latched = False
        self.top_low_width_count = 0
        self.low_speed_latched = False

    def _top_slice_requests_low_speed(self, vision_result) -> bool:
        width_top = int(getattr(vision_result, "width_top", 0) or 0)
        if 0 < width_top < TOP_SLICE_LOW_WIDTH_THRESHOLD:
            self.top_low_width_count += 1
        else:
            self.top_low_width_count = 0
        return self.top_low_width_count >= TOP_SLICE_LOW_WIDTH_FRAMES

    def _update_turn_sign(self, err_px: float):
        if err_px > STEER_DEADZONE:
            self.last_turn_sign = +1
        elif err_px < -STEER_DEADZONE:
            self.last_turn_sign = -1

    def _update_green_turn_sign(self, vision_result) -> None:
        if not getattr(vision_result, "seen_top", False):
            return

        err_px = vision_result.err_top
        if err_px > STEER_DEADZONE:
            self.last_green_turn_sign = +1
        elif err_px < -STEER_DEADZONE:
            self.last_green_turn_sign = -1

    def _line_visible_steer_target(self, vision_result) -> float:
        self._update_green_turn_sign(vision_result)

        if vision_result.seen_bot:
            err_px = vision_result.err_bot
            self._update_turn_sign(err_px)
            return normalize_error(err_px)

        if vision_result.seen_mid:
            err_px = vision_result.err_mid
            self._update_turn_sign(err_px)
            return normalize_error(err_px)

        if vision_result.seen_top:
            err_px = vision_result.err_top
            self._update_turn_sign(err_px)
            return normalize_error(err_px)

        return self.last_turn_sign * SEARCH_TURN_STEER

    def _apply_speed_limit(self, speed: float) -> float:
        if self.low_speed_latched:
            return min(speed, self.low_speed)
        return speed

    def update(self, vision_result):
        if getattr(vision_result, "t_detected", False):
            self.low_speed_latched = False

        if not self.t_confirmed:
            if getattr(vision_result, "t_detected", False):
                self.t_count += 1
            else:
                self.t_count = 0

            self.t_confirmed = self.t_count >= T_CONFIRM_FRAMES
            if self.t_confirmed:
                self.t_line_count = T_LINE_FRAMES
        elif self.t_line_count > 0:
            if getattr(vision_result, "bottom_slice_full_white", False):
                self.t_stop_latched = True
            else:
                self.t_line_count -= 1
                if self.t_line_count == 0:
                    self.t_count = 0
                    self.t_confirmed = False

        if self.t_stop_latched:
            self.mode = "T_STOP"
            self.steer_cmd = 0.0
            return 0.0, 0.0, 0.0, self.steer_cmd, self.mode

        if getattr(vision_result, "any_seen", False):
            self.mode = "TRACK"
            self.lost_count = 0

            steer_target = self._line_visible_steer_target(vision_result)
            self.steer_cmd = smooth(self.steer_cmd, steer_target, STEER_SMOOTH_ALPHA)
            use_low_speed = self._top_slice_requests_low_speed(vision_result)
            if use_low_speed:
                self.low_speed_latched = True

            steer_mag = abs(self.steer_cmd)
            if steer_mag > 0.7:
                speed = self.turn_speed
            elif steer_mag > 0.35:
                speed = 0.5 * (self.base_speed + self.turn_speed)
            else:
                speed = self.base_speed

            speed = self._apply_speed_limit(speed)
        else:
            self.lost_count += 1
            self.top_low_width_count = 0

            if self.lost_count <= BLIND_TURN_FRAMES:
                self.mode = "BLIND_TURN"
                steer_target = self.last_green_turn_sign * BLIND_TURN_STEER
                self.steer_cmd = smooth(self.steer_cmd, steer_target, 0.45)
                speed = self.blind_speed
            elif self.lost_count <= SEARCH_FRAMES:
                self.mode = "SEARCH"
                steer_target = self.last_turn_sign * SEARCH_TURN_STEER
                self.steer_cmd = smooth(self.steer_cmd, steer_target, 0.25)
                speed = self.search_speed
            else:
                self.mode = "SEARCH"
                steer_target = self.last_turn_sign * SEARCH_TURN_STEER
                self.steer_cmd = smooth(self.steer_cmd, steer_target, 0.18)
                speed = 0.0

            speed = self._apply_speed_limit(speed)

        steer = clamp(self.steer_cmd, -1.0, 1.0)
        left = clamp(speed + self.steer_gain * steer, -1.0, 1.0)
        right = clamp(speed - self.steer_gain * steer, -1.0, 1.0)
        return left, right, speed, self.steer_cmd, self.mode
