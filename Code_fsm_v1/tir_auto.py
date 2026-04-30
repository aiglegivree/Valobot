from __future__ import annotations

import math
import time

import cv2
import numpy as np
import RPi.GPIO as GPIO

from config import (
    ANGLE_FIRE_START,
    ANGLE_INCREMENT,
    ANGLE_LOADED,
    BLUE_HIGH,
    BLUE_LOW,
    CENTER_TOL,
    MAG_CAPACITY,
    MIN_TARGET_AREA,
    MIN_TARGET_WIDTH,
    MIN_TIME_BETWEEN_SHOTS,
    MOTEUR_PIN,
    POST_SHOT_CONFIRM_FRAMES,
    POST_SHOT_SETTLE_TIME,
    PWM_FREQ,
    SERVO_PIN,
    TARGET_MATCH_DIST_PX,
    TURN_SPEED,
    W,
)
from motors import MotorDriver


class Shooter:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(MOTEUR_PIN, GPIO.OUT)
        GPIO.setup(SERVO_PIN, GPIO.OUT)

        self.pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
        self.pwm.start(0)

        self.shots_fired = 0
        self.last_shot_time = 0.0

        self.set_servo_angle(ANGLE_LOADED)
        GPIO.output(MOTEUR_PIN, GPIO.LOW)

    def set_servo_angle(self, angle: float):
        duty = (angle / 18.0) + 2.5
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.22)
        self.pwm.ChangeDutyCycle(0)

    def spin_flywheel(self, on: bool):
        GPIO.output(MOTEUR_PIN, GPIO.HIGH if on else GPIO.LOW)

    def can_shoot(self) -> bool:
        if self.shots_fired >= MAG_CAPACITY:
            return False
        if time.time() - self.last_shot_time < MIN_TIME_BETWEEN_SHOTS:
            return False
        return True

    def fire_once(self) -> bool:
        if not self.can_shoot():
            return False

        fire_angle = ANGLE_FIRE_START + self.shots_fired * ANGLE_INCREMENT
        self.set_servo_angle(fire_angle)
        time.sleep(0.15)

        self.shots_fired += 1
        self.last_shot_time = time.time()
        return True

    def ammo_remaining(self) -> int:
        return max(0, MAG_CAPACITY - self.shots_fired)

    def is_empty(self) -> bool:
        return self.shots_fired >= MAG_CAPACITY

    def cleanup(self):
        try:
            self.spin_flywheel(False)
        except Exception:
            pass
        try:
            self.set_servo_angle(ANGLE_LOADED)
        except Exception:
            pass
        try:
            self.pwm.stop()
        except Exception:
            pass
        try:
            GPIO.cleanup([MOTEUR_PIN, SERVO_PIN])
        except Exception:
            pass


def detect_blue_targets(
    frame: np.ndarray,
    min_ratio: float | None = None,
    max_ratio: float | None = None,
):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_TARGET_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if w < MIN_TARGET_WIDTH:
            continue

        ratio = min(w, h) / max(w, h)
        if min_ratio is not None and ratio < min_ratio:
            continue
        if max_ratio is not None and ratio > max_ratio:
            continue

        targets.append(
            {
                "contour": contour,
                "area": float(area),
                "bbox": (x, y, w, h),
                "cx": int(x + w // 2),
                "cy": int(y + h // 2),
                "ratio": float(ratio),
            }
        )

    targets.sort(key=lambda target: target["area"], reverse=True)
    return mask, targets


def match_target_by_position(targets, ref_x, ref_y, max_dist_px):
    best = None
    best_dist = float("inf")

    for target in targets:
        distance = math.hypot(target["cx"] - ref_x, target["cy"] - ref_y)
        if distance < best_dist:
            best_dist = distance
            best = target

    if best is None:
        return None, float("inf")
    return best, best_dist


def pick_target(targets, last_target_center=None):
    if not targets:
        return None

    if last_target_center is None:
        return targets[0]

    best, dist = match_target_by_position(
        targets,
        last_target_center[0],
        last_target_center[1],
        TARGET_MATCH_DIST_PX,
    )
    if best is not None and dist <= TARGET_MATCH_DIST_PX:
        return best
    return targets[0]


def pick_target_by_shot_order(targets, confirmed_hits: int):
    if not targets:
        return None

    if confirmed_hits <= 0:
        return max(targets, key=lambda target: target["cx"])

    if confirmed_hits == 1:
        return min(targets, key=lambda target: target["cx"])

    targets_by_x = sorted(targets, key=lambda target: target["cx"])
    return targets_by_x[len(targets_by_x) // 2]


def aim_to_target(motors: MotorDriver, target):
    err = target["cx"] - (W // 2)

    if abs(err) <= CENTER_TOL:
        motors.set_speeds(0, 0)
        return True, err

    turn_speed = 0.30
    if err > 0:
        motors.set_speeds(turn_speed, -turn_speed)
    else:
        motors.set_speeds(-turn_speed, turn_speed)

    return False, err
