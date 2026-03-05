# motors.py
from __future__ import annotations
import time

try:
    import RPi.GPIO as GPIO
except Exception as e:
    GPIO = None  # allows import on non-RPi for dev/testing

from config import (
    ENA, IN1, IN2, ENB, IN3, IN4, PWM_FREQ,
    MIN_SPEED, MAX_SPEED, LEFT_TRIM, RIGHT_TRIM, SWAP_MOTORS
)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class MotorDriver:
    """
    Differential drive using 2 DC motors + H-bridge.
    Speed inputs expected in -100..+100 (percent duty).
    """
    def __init__(self) -> None:
        if GPIO is None:
            raise RuntimeError("RPi.GPIO not available. Run on Raspberry Pi or install proper GPIO backend.")

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in (ENA, IN1, IN2, ENB, IN3, IN4):
            GPIO.setup(pin, GPIO.OUT)

        self.pwm_a = GPIO.PWM(ENA, PWM_FREQ)
        self.pwm_b = GPIO.PWM(ENB, PWM_FREQ)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

        self.stop()

    def _set_motor(self, pwm, in_fwd: int, in_rev: int, speed: float) -> None:
        # speed in -100..+100
        s = _clamp(speed, -100.0, 100.0)
        if abs(s) < 1e-6:
            GPIO.output(in_fwd, GPIO.LOW)
            GPIO.output(in_rev, GPIO.LOW)
            pwm.ChangeDutyCycle(0)
            return

        if s >= 0:
            GPIO.output(in_fwd, GPIO.HIGH)
            GPIO.output(in_rev, GPIO.LOW)
            pwm.ChangeDutyCycle(_clamp(abs(s), 0.0, 100.0))
        else:
            GPIO.output(in_fwd, GPIO.LOW)
            GPIO.output(in_rev, GPIO.HIGH)
            pwm.ChangeDutyCycle(_clamp(abs(s), 0.0, 100.0))

    def set_speeds(self, left: float, right: float) -> None:
        # Apply trims
        left *= LEFT_TRIM
        right *= RIGHT_TRIM

        # Optional swap if wiring is crossed
        if SWAP_MOTORS:
            left, right = right, left

        # Clamp
        left = _clamp(left, -100.0, 100.0)
        right = _clamp(right, -100.0, 100.0)

        self._set_motor(self.pwm_a, IN1, IN2, left)
        self._set_motor(self.pwm_b, IN3, IN4, right)

    def stop(self) -> None:
        self.set_speeds(0.0, 0.0)

    def cleanup(self) -> None:
        try:
            self.stop()
            time.sleep(0.05)
            self.pwm_a.stop()
            self.pwm_b.stop()
        finally:
            GPIO.cleanup()