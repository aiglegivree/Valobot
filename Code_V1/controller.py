# controller.py
from __future__ import annotations
from dataclasses import dataclass
import time

from config import (
    KP, KD, EMA_ALPHA, GATE_PX,
    BASE_SPEED, MIN_SPEED, MAX_SPEED,
    LOST_TIMEOUT_S, SEARCH_STEER, SEARCH_SPEED_FACTOR,
    INVERT_STEER
)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class ControlOutput:
    speed: float   # 0..100
    steer: float   # -1..+1
    state: str     # "SEEN" or "LOST"
    e_used: float  # filtered / predicted error (px)

class LineController:
    def __init__(self) -> None:
        self.e_filt = 0.0
        self.e_prev = 0.0
        self.last_time = time.time()

        self.last_seen_time = time.time()
        self.last_turn_sign = +1  # +1 right, -1 left
        self.steer_prev = 0.0

        self.state = "LOST"

    def update(self, found: bool, e_meas: float) -> ControlOutput:
        now = time.time()
        dt = max(1e-3, now - self.last_time)
        self.last_time = now

        # Prediction is simply "keep last filtered value"
        e_pred = self.e_filt

        usable_meas = False
        if found:
            if abs(e_meas - e_pred) <= GATE_PX:
                usable_meas = True

        if usable_meas:
            # EMA filter
            self.e_filt = (EMA_ALPHA * e_meas) + ((1.0 - EMA_ALPHA) * self.e_filt)
            de = (self.e_filt - self.e_prev) / dt
            self.e_prev = self.e_filt

            steer = (KP * self.e_filt) + (KD * de)
            steer = _clamp(steer, -1.0, 1.0)

            if INVERT_STEER:
                steer = -steer

            # update memory of turn direction
            if steer > 0.05:
                self.last_turn_sign = +1
            elif steer < -0.05:
                self.last_turn_sign = -1

            self.last_seen_time = now
            self.state = "SEEN"

            speed = _clamp(BASE_SPEED, MIN_SPEED, MAX_SPEED)
            self.steer_prev = steer
            return ControlOutput(speed=speed, steer=steer, state=self.state, e_used=self.e_filt)

        # LOST mode
        t_lost = now - self.last_seen_time
        self.state = "LOST"

        steer = self.last_turn_sign * SEARCH_STEER
        if t_lost > LOST_TIMEOUT_S:
            steer = _clamp(steer * 1.25, -1.0, 1.0)

        if INVERT_STEER:
            steer = -steer

        speed = _clamp(BASE_SPEED * SEARCH_SPEED_FACTOR, MIN_SPEED, MAX_SPEED)
        self.steer_prev = steer
        return ControlOutput(speed=speed, steer=steer, state=self.state, e_used=e_pred)