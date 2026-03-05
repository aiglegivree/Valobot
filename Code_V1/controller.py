# controller.py
from __future__ import annotations
from dataclasses import dataclass
import time

from config import (
    W, BASE_SPEED, MIN_SPEED, MAX_SPEED,
    W_LOOK, W_BOTTOM, KP, DEADBAND, STEER_LPF_ALPHA,
    LINE_LOST_TIMEOUT_S, SEARCH_FORWARD,
    SEARCH_STEER_START, SEARCH_STEER_MAX, SEARCH_RAMP_TIME
)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

@dataclass
class ControlOut:
    left: float
    right: float
    state: str
    e: float
    steer: float
    last_turn_sign: int

class LineFollower:
    def __init__(self):
        self.last_seen_time = time.time()
        self.last_turn_sign = 1
        self.steer_prev = 0.0

    def update(self, pick) -> ControlOut:
        now = time.time()

        if pick is None:
            # LINE LOST
            lost_dt = now - self.last_seen_time

            if lost_dt < LINE_LOST_TIMEOUT_S:
                # hold last steering briefly
                left = clamp(BASE_SPEED - self.steer_prev, MIN_SPEED, MAX_SPEED)
                right = clamp(BASE_SPEED + self.steer_prev, MIN_SPEED, MAX_SPEED)
                return ControlOut(left, right, "LOST_HOLD", 0.0, self.steer_prev, self.last_turn_sign)

            # search ramp
            r = clamp(lost_dt / SEARCH_RAMP_TIME, 0.0, 1.0)
            steer_mag = SEARCH_STEER_START + r * (SEARCH_STEER_MAX - SEARCH_STEER_START)
            steer = self.last_turn_sign * steer_mag

            base = SEARCH_FORWARD  # 0 => turn in place
            left = clamp(base - steer, -MAX_SPEED, MAX_SPEED)
            right = clamp(base + steer, -MAX_SPEED, MAX_SPEED)

            self.steer_prev = steer
            return ControlOut(left, right, "SEARCH", 0.0, steer, self.last_turn_sign)

        # LINE FOUND
        self.last_seen_time = now

        # normalized errors [-1, 1]
        e_look = (pick.cx_look - (W / 2.0)) / (W / 2.0)
        e_bot  = (pick.cx_bottom - (W / 2.0)) / (W / 2.0)

        e = W_LOOK * e_look + W_BOTTOM * e_bot

        if abs(e) < DEADBAND:
            steer = 0.0
        else:
            steer = KP * e

        # LPF
        steer = (1.0 - STEER_LPF_ALPHA) * self.steer_prev + STEER_LPF_ALPHA * steer
        self.steer_prev = steer

        # remember turn direction for recovery
        if abs(steer) > 1e-3:
            self.last_turn_sign = 1 if steer > 0 else -1

        left = clamp(BASE_SPEED - steer, MIN_SPEED, MAX_SPEED)
        right = clamp(BASE_SPEED + steer, MIN_SPEED, MAX_SPEED)

        return ControlOut(left, right, "TRACK", e, steer, self.last_turn_sign)