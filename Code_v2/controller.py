# controller.py
from __future__ import annotations
import numpy as np

from config import (
    W,
    BASE_SPEED, TURN_SPEED, BLIND_SPEED, SEARCH_SPEED,
    STEER_GAIN, STEER_SMOOTH_ALPHA, STEER_DEADZONE,
    HARD_TURN_THRESH,
    BLIND_TURN_FRAMES, SEARCH_FRAMES,
    BLIND_TURN_STEER, SEARCH_TURN_STEER,
    T_CONFIRM_FRAMES
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def smooth(prev: float, target: float, alpha: float) -> float:
    return (1 - alpha) * prev + alpha * target


def normalize_error(err_px: float) -> float:
    half = max(W / 2, 1)
    return clamp(err_px / half, -1.0, 1.0)


def steering_to_wheels(speed: float, steer: float):
    steer = clamp(steer, -1.0, 1.0)

    left = speed + STEER_GAIN * steer
    right = speed - STEER_GAIN * steer

    return clamp(left, -1, 1), clamp(right, -1, 1)


class Controller:
    def __init__(self):
        self.mode = "TRACK"

        self.steer_cmd = 0.0
        self.last_good_steer = 0.0
        self.last_turn_sign = 1

        self.lost_count = 0
        self.hard_turn_pending = False

        # T stop logic
        self.t_count = 0
        self.t_confirmed = False
        self.t_stop_latched = False

    def update(self, vision_result):
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

        # =========================================================
        # STEP 2: PERMANENT STOP ON T
        # =========================================================
        if self.t_stop_latched:
            self.mode = "T_STOP"
            self.steer_cmd = 0.0
            return 0.0, 0.0, 0.0, self.steer_cmd, self.mode

        # =========================================================
        # STEP 3: LINE VISIBLE
        # =========================================================
        if getattr(vision_result, "any_seen", False):
            self.mode = "TRACK"
            self.lost_count = 0

            steer_target = normalize_error(vision_result.steer_error)

            if (
                getattr(vision_result, "sharp_turn_detected", False) or
                abs(vision_result.steer_error) > HARD_TURN_THRESH
            ):
                self.hard_turn_pending = True
            else:
                self.hard_turn_pending = False

            self.steer_cmd = smooth(
                self.steer_cmd,
                steer_target,
                STEER_SMOOTH_ALPHA
            )

            self.last_good_steer = self.steer_cmd

            if vision_result.steer_error > STEER_DEADZONE:
                self.last_turn_sign = +1
            elif vision_result.steer_error < -STEER_DEADZONE:
                self.last_turn_sign = -1

            if self.hard_turn_pending or abs(self.steer_cmd) > 0.55:
                speed = TURN_SPEED
            else:
                speed = BASE_SPEED

        # =========================================================
        # STEP 4: LINE LOST
        # =========================================================
        else:
            self.lost_count += 1

            if self.lost_count <= BLIND_TURN_FRAMES:
                self.mode = "BLIND_TURN"

                remembered = self.last_good_steer
                forced = self.last_turn_sign * BLIND_TURN_STEER

                if np.sign(remembered) == np.sign(forced):
                    if abs(remembered) < abs(forced):
                        steer_target = forced
                    else:
                        steer_target = remembered
                else:
                    steer_target = forced

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
                speed = SEARCH_SPEED

        left, right = steering_to_wheels(speed, self.steer_cmd)

        return left, right, speed, self.steer_cmd, self.mode
