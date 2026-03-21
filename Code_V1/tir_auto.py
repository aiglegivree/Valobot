from __future__ import annotations
import time
import math
import cv2
import numpy as np
import RPi.GPIO as GPIO

from camera import Camera
from motors import MotorDriver
from leds import LED
from controller import Controller
from vision import detect_line, draw_debug_slices, draw_mask_slices

from config import (
    W, H,
    BAND_THICKNESS,

    # blue detection
    BLUE_LOW, BLUE_HIGH,
    MIN_TARGET_AREA,
    CENTER_TOL,
    LOCK_FRAMES_REQUIRED,
    SHOW_DEBUG,

    # shooter
    MOTEUR_PIN, SERVO_PIN,
    PWM_FREQ, ANGLE_LOADED, ANGLE_FIRE_START, ANGLE_INCREMENT,
    MAG_CAPACITY, MIN_TIME_BETWEEN_SHOTS,
    POST_SHOT_SETTLE_TIME, POST_SHOT_CONFIRM_FRAMES,
    TARGET_MATCH_DIST_PX,
    NOMBRE_DE_CIBLES,

    # target aiming
    TURN_SPEED,
)

# ============================================================
# SHOOTER
# ============================================================
class Shooter:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(MOTEUR_PIN, GPIO.OUT)
        GPIO.setup(SERVO_PIN, GPIO.OUT)

        self.motor = GPIO.output
        self.pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
        self.pwm.start(0)

        self.angle = ANGLE_LOADED
        self.shots_fired = 0
        self.last_shot_time = 0.0
        self.loaded = False

        self.set_servo_angle(ANGLE_LOADED)
        time.sleep(0.3)
        self.loaded = True

    def set_servo_angle(self, angle: float):
        duty = (angle / 18.0) + 2.5
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.20)
        self.pwm.ChangeDutyCycle(0)

    def spin_flywheel(self, on: bool):
        GPIO.output(MOTEUR_PIN, GPIO.HIGH if on else GPIO.LOW)

    def can_shoot(self) -> bool:
        if self.shots_fired >= MAG_CAPACITY:
            return False
        if (time.time() - self.last_shot_time) < MIN_TIME_BETWEEN_SHOTS:
            return False
        return True

    def fire_once(self) -> bool:
        if not self.can_shoot():
            return False

        fire_angle = ANGLE_FIRE_START + self.shots_fired * ANGLE_INCREMENT
        self.set_servo_angle(fire_angle)
        time.sleep(0.20)
        self.set_servo_angle(ANGLE_LOADED)

        self.shots_fired += 1
        self.last_shot_time = time.time()
        return True

    def is_empty(self) -> bool:
        return self.shots_fired >= MAG_CAPACITY

    def cleanup(self):
        try:
            self.spin_flywheel(False)
        except Exception:
            pass
        try:
            self.pwm.stop()
        except Exception:
            pass


# ============================================================
# BLUE TARGET DETECTION
# ============================================================
def detect_blue_targets(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_TARGET_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2

        targets.append({
            "contour": c,
            "area": area,
            "bbox": (x, y, w, h),
            "cx": cx,
            "cy": cy,
        })

    targets.sort(key=lambda t: t["area"], reverse=True)
    return mask, targets


def pick_target(targets, last_target_center=None):
    if not targets:
        return None

    if last_target_center is None:
        return targets[0]

    best = None
    best_dist = float("inf")

    for t in targets:
        d = math.hypot(t["cx"] - last_target_center[0], t["cy"] - last_target_center[1])
        if d < best_dist:
            best_dist = d
            best = t

    if best is not None and best_dist <= TARGET_MATCH_DIST_PX:
        return best

    return targets[0]


# ============================================================
# TARGET AIMING
# ============================================================
def aim_to_target(motors: MotorDriver, target) -> tuple[bool, float]:
    """
    Returns:
        locked, error_px
    """
    cx = target["cx"]
    err = cx - (W // 2)

    if abs(err) <= CENTER_TOL:
        motors.set_speeds(0, 0)
        return True, err

    steer = np.clip(err / (W / 2), -1.0, 1.0)

    # rotate in place toward target
    if steer > 0:
        motors.set_speeds(TURN_SPEED, -TURN_SPEED)
    else:
        motors.set_speeds(-TURN_SPEED, TURN_SPEED)

    return False, err


# ============================================================
# MAIN
# ============================================================
def main():
    cam = Camera()
    controller = Controller()
    motors = MotorDriver()
    leds = LED()
    shooter = Shooter()

    STATE_FOLLOW_TO_FIRST_T = "FOLLOW_TO_FIRST_T"
    STATE_SHOOTING = "SHOOTING"
    STATE_RETURN_TO_T = "RETURN_TO_T"
    STATE_AURA = "AURA"

    state = STATE_FOLLOW_TO_FIRST_T
    aura_mode = False

    lock_frames = 0
    last_target_center = None
    post_shot_ignore_frames = 0

    try:
        shooter.spin_flywheel(True)

        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # ------------------------------------------------
            # STATE 1: FOLLOW LINE UNTIL FIRST T
            # ------------------------------------------------
            if state == STATE_FOLLOW_TO_FIRST_T:
                res = detect_line(frame, angle_deg=20)
                left, right, _, _, mode = controller.update(res)

                leds.turn_on_all()
                motors.set_speeds(left, right)

                if mode == "T_STOP" or getattr(res, "is_T", False):
                    motors.set_speeds(0, 0)
                    time.sleep(0.2)
                    state = STATE_SHOOTING
                    lock_frames = 0
                    last_target_center = None

                dbg = draw_debug_slices(frame, res, BAND_THICKNESS)
                mask_dbg = draw_mask_slices(res)

                cv2.imshow("camera debug", dbg)
                cv2.imshow("mask debug", mask_dbg)

            # ------------------------------------------------
            # STATE 2: SHOOTING PHASE
            # ------------------------------------------------
            elif state == STATE_SHOOTING:
                leds.turn_on_all()

                blue_mask, targets = detect_blue_targets(frame)

                if post_shot_ignore_frames > 0:
                    post_shot_ignore_frames -= 1
                    motors.set_speeds(0, 0)
                else:
                    target = pick_target(targets, last_target_center)

                    if target is None:
                        # no target found -> stay still
                        motors.set_speeds(0, 0)
                        lock_frames = 0
                        last_target_center = None
                    else:
                        last_target_center = (target["cx"], target["cy"])
                        locked, _ = aim_to_target(motors, target)

                        if locked:
                            lock_frames += 1
                        else:
                            lock_frames = 0

                        if lock_frames >= LOCK_FRAMES_REQUIRED:
                            fired = shooter.fire_once()
                            lock_frames = 0
                            if fired:
                                post_shot_ignore_frames = POST_SHOT_CONFIRM_FRAMES
                                time.sleep(POST_SHOT_SETTLE_TIME)

                mission_done = (
                    shooter.shots_fired >= NOMBRE_DE_CIBLES
                    or shooter.is_empty()
                )

                if mission_done:
                    motors.set_speeds(0, 0)
                    time.sleep(0.2)
                    state = STATE_RETURN_TO_T

                # debug drawing for shooting
                dbg = frame.copy()
                for t in targets:
                    x, y, w, h = t["bbox"]
                    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(dbg, (t["cx"], t["cy"]), 4, (0, 0, 255), -1)

                cv2.line(dbg, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
                cv2.putText(
                    dbg,
                    f"STATE: SHOOTING | shots={shooter.shots_fired}/{MAG_CAPACITY}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("camera debug", dbg)
                cv2.imshow("mask debug", blue_mask)

            # ------------------------------------------------
            # STATE 3: RETURN TO T
            # ------------------------------------------------
            elif state == STATE_RETURN_TO_T:
                res = detect_line(frame, angle_deg=20)
                left, right, _, _, mode = controller.update(res)

                leds.turn_on_all()
                motors.set_speeds(left, right)

                if mode == "T_STOP" or getattr(res, "is_T", False):
                    motors.set_speeds(0, 0)
                    state = STATE_AURA
                    aura_mode = False

                dbg = draw_debug_slices(frame, res, BAND_THICKNESS)
                mask_dbg = draw_mask_slices(res)

                cv2.imshow("camera debug", dbg)
                cv2.imshow("mask debug", mask_dbg)

            # ------------------------------------------------
            # STATE 4: FINAL AURA
            # ------------------------------------------------
            elif state == STATE_AURA:
                if not aura_mode:
                    aura_mode = True
                    motors.set_speeds(0, 0)
                    shooter.spin_flywheel(False)
                    leds.aura_farm_start()

                leds.aura_farm_step(delay=0.05)

                aura_dbg = frame.copy()
                cv2.putText(
                    aura_dbg,
                    "STATE: AURA",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("camera debug", aura_dbg)
                cv2.imshow("mask debug", np.zeros((H, W), dtype=np.uint8))

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        try:
            leds.aura_farm_stop()
        except Exception:
            pass

        leds.turn_off_all()
        motors.set_speeds(0, 0)
        shooter.cleanup()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()