from __future__ import annotations

import time

import RPi.GPIO as GPIO

from fsm_config import DRIVE_ACCEL_LIMIT_PER_SEC, DRIVE_SPEED_FACTOR

# Right motor
RIGHT_EN = 13
RIGHT_IN1 = 14
RIGHT_IN2 = 15

# Left motor
LEFT_EN = 12
LEFT_IN1 = 16
LEFT_IN2 = 17

PWM_FREQ = 1000

LEFT_TRIM = 1.00
RIGHT_TRIM = 1.00


class MotorDriver:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in [RIGHT_EN, RIGHT_IN1, RIGHT_IN2, LEFT_EN, LEFT_IN1, LEFT_IN2]:
            GPIO.setup(pin, GPIO.OUT)

        self.pwm_right = GPIO.PWM(RIGHT_EN, PWM_FREQ)
        self.pwm_left = GPIO.PWM(LEFT_EN, PWM_FREQ)

        self.pwm_right.start(0)
        self.pwm_left.start(0)

        self.last_left_cmd = 0.0
        self.last_right_cmd = 0.0
        self.last_update_time = time.monotonic()

        self.stop()

    def _limit_accel(self, current: float, target: float, max_step: float) -> float:
        if max_step <= 0.0:
            return target
        delta = target - current
        if delta > max_step:
            return current + max_step
        if delta < -max_step:
            return current - max_step
        return target

    def _set_one(self, pwm, in1, in2, value: float):
        value = max(-1.0, min(1.0, value))
        duty = abs(value) * 100.0

        if value > 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif value < 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        pwm.ChangeDutyCycle(duty)

    def set_speeds(self, left: float, right: float):
        left_target = max(-1.0, min(1.0, left * LEFT_TRIM * DRIVE_SPEED_FACTOR))
        right_target = max(-1.0, min(1.0, right * RIGHT_TRIM * DRIVE_SPEED_FACTOR))

        now = time.monotonic()
        dt = max(0.0, now - self.last_update_time)
        max_step = DRIVE_ACCEL_LIMIT_PER_SEC * dt

        left_limited = self._limit_accel(self.last_left_cmd, left_target, max_step)
        right_limited = self._limit_accel(self.last_right_cmd, right_target, max_step)

        self._set_one(self.pwm_left, LEFT_IN1, LEFT_IN2, left_limited)
        self._set_one(self.pwm_right, RIGHT_IN1, RIGHT_IN2, right_limited)

        self.last_left_cmd = left_limited
        self.last_right_cmd = right_limited
        self.last_update_time = now

    def stop(self):
        self._set_one(self.pwm_left, LEFT_IN1, LEFT_IN2, 0.0)
        self._set_one(self.pwm_right, RIGHT_IN1, RIGHT_IN2, 0.0)
        self.last_left_cmd = 0.0
        self.last_right_cmd = 0.0
        self.last_update_time = time.monotonic()

    def cleanup(self):
        self.stop()
        self.pwm_left.stop()
        self.pwm_right.stop()
        GPIO.cleanup()
