from __future__ import annotations
import time

from config import (
    W,
    BASE_SPEED, TURN_SPEED, BLIND_SPEED, SEARCH_SPEED,
    STEER_GAIN, STEER_SMOOTH_ALPHA, STEER_DEADZONE,
    BLIND_TURN_FRAMES, SEARCH_FRAMES,
    BLIND_TURN_STEER, SEARCH_TURN_STEER,
    T_CONFIRM_FRAMES,
    SHARP_TURN_HOLD_S,
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
    def __init__(self):
        self.mode = "TRACK"

        self.steer_cmd = 0.0
        self.last_turn_sign = 1
        self.lost_count = 0

        self.t_count = 0
        self.t_confirmed = False
        self.t_stop_latched = False
        self.sharp_turn_hold_until = 0.0

    def _update_turn_sign(self, err_px: float):
        if err_px > STEER_DEADZONE:
            self.last_turn_sign = +1
        elif err_px < -STEER_DEADZONE:
            self.last_turn_sign = -1

    def _line_visible_steer_target(self, vision_result) -> float:
        # Strongly prioritize the bottom of the line because that is the part
        # physically closest to the robot and gives the most useful steering cue.
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

    def update(self, vision_result):
        now = time.monotonic()

        # =========================================================
        # STEP 1: CONFIRM T
        # =========================================================
        if getattr(vision_result, "t_detected", False):
            self.t_count += 1
        else:
            self.t_count = 0

        self.t_confirmed = self.t_count >= T_CONFIRM_FRAMES
        if self.t_confirmed:
            self.t_stop_latched = True
            #self.t_stop_latched = False
        # =========================================================
        # STEP 2: PERMANENT STOP ON T
        # =========================================================
        if self.t_stop_latched:
            self.mode = "T_STOP"
            self.steer_cmd = 0.0
            return 0.0, 0.0, 0.0, self.steer_cmd, self.mode

        # =========================================================
        # STEP 3: SHARP TURN
        # =========================================================
        sharp_turn_active = getattr(vision_result, "sharp_turn_detected", False)
        if sharp_turn_active:
            self.sharp_turn_hold_until = now + SHARP_TURN_HOLD_S

        if sharp_turn_active or now < self.sharp_turn_hold_until:
            self.mode = "SHARP_TURN"
            self.lost_count = 0

            if getattr(vision_result, "any_seen", False):
                steer_target = 1.6 * self._line_visible_steer_target(vision_result)
            else:
                steer_target = self.last_turn_sign * BLIND_TURN_STEER

            steer_target = clamp(steer_target, -1.0, 1.0)
            self.steer_cmd = smooth(self.steer_cmd, steer_target, 0.6)
            speed = TURN_SPEED

            left, right = steering_to_wheels(speed, self.steer_cmd)
            return left, right, speed, self.steer_cmd, self.mode

        # =========================================================
        # STEP 4: NORMAL TRACKING
        # =========================================================
        if getattr(vision_result, "any_seen", False):
            self.mode = "TRACK"
            self.lost_count = 0

            steer_target = self._line_visible_steer_target(vision_result)
            self.steer_cmd = smooth(self.steer_cmd, steer_target, STEER_SMOOTH_ALPHA)

            # Slow down progressively as steering demand rises.
            steer_mag = abs(self.steer_cmd)
            if steer_mag > 0.7:
                speed = TURN_SPEED
            elif steer_mag > 0.35:
                speed = 0.5 * (BASE_SPEED + TURN_SPEED)
            else:
                speed = BASE_SPEED

        # =========================================================
        # STEP 5: LINE LOST
        # =========================================================
        else:
            self.lost_count += 1

            if self.lost_count <= BLIND_TURN_FRAMES:
                self.mode = "BLIND_TURN"
                steer_target = self.last_turn_sign * BLIND_TURN_STEER
                self.steer_cmd = smooth(self.steer_cmd, steer_target, 0.45)
                speed = BLIND_SPEED

            elif self.lost_count <= SEARCH_FRAMES:
                self.mode = "SEARCH"
                steer_target = self.last_turn_sign * SEARCH_TURN_STEER
                self.steer_cmd = smooth(self.steer_cmd, steer_target, 0.25)
                speed = SEARCH_SPEED

            else:
                self.mode = "SEARCH"
                steer_target = self.last_turn_sign * SEARCH_TURN_STEER
                self.steer_cmd = smooth(self.steer_cmd, steer_target, 0.18)
                speed = 0.0

        left, right = steering_to_wheels(speed, self.steer_cmd)
        return left, right, speed, self.steer_cmd, self.mode
