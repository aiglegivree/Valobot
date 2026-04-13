# tir_auto.py
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

    # Blue detection
    BLUE_LOW, BLUE_HIGH,
    MIN_TARGET_AREA,
    CENTER_TOL, LOCK_FRAMES_REQUIRED,
    TARGET_MATCH_DIST_PX,

    # Shooter
    MOTEUR_PIN, SERVO_PIN,
    PWM_FREQ, ANGLE_LOADED, ANGLE_FIRE_START, ANGLE_INCREMENT,
    MAG_CAPACITY, MIN_TIME_BETWEEN_SHOTS,
    POST_SHOT_SETTLE_TIME, POST_SHOT_CONFIRM_FRAMES,
    NOMBRE_DE_CIBLES,

    # Motor aiming speed
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
        self.set_servo_angle(ANGLE_LOADED)

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
            self.pwm.stop()
        except Exception:
            pass
        try:
            GPIO.cleanup([MOTEUR_PIN, SERVO_PIN])
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

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_TARGET_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2

        targets.append({
            "contour": c,
            "area": float(area),
            "bbox": (x, y, w, h),
            "cx": int(cx),
            "cy": int(cy),
        })

    targets.sort(key=lambda t: t["area"], reverse=True)
    return mask, targets


def match_target_by_position(targets, ref_x, ref_y, max_dist_px):
    best = None
    best_dist = float("inf")

    for t in targets:
        d = math.hypot(t["cx"] - ref_x, t["cy"] - ref_y)
        if d < best_dist:
            best_dist = d
            best = t

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


# ============================================================
# AIMING
# ============================================================
def aim_to_target(motors: MotorDriver, target):
    cx = target["cx"]
    err = cx - (W // 2)

    if abs(err) <= CENTER_TOL:
        motors.set_speeds(0, 0)
        return True, err

    if err > 0:
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

    # Shooting state variables
    lock_frames = 0
    last_target_center = None

    pending_shot = False
    pending_target_before_shot = None
    confirm_delay_frames = 0
    targets_hit = 0

    # Avoid immediate second T when leaving shooting arena
    return_to_t_ignore_until = 0.0

    FLYWHEEL_SPINUP_TIME = 0.7
    RETURN_T_IGNORE_SECONDS = 1.0
    HIT_AREA_RATIO_THRESHOLD = 0.45

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # ========================================================
            # 1) FOLLOW LINE UNTIL FIRST T
            # ========================================================
            if state == STATE_FOLLOW_TO_FIRST_T:
                res = detect_line(frame, angle_deg=20)
                left, right, _, _, mode = controller.update(res)

                leds.turn_on_all()
                motors.set_speeds(left, right)

                if mode == "T_STOP" or res.t_detected:
                    motors.set_speeds(0, 0)
                    shooter.spin_flywheel(True)
                    time.sleep(FLYWHEEL_SPINUP_TIME)

                    state = STATE_SHOOTING
                    lock_frames = 0
                    last_target_center = None
                    pending_shot = False
                    pending_target_before_shot = None
                    confirm_delay_frames = 0

                dbg = draw_debug_slices(frame, res, BAND_THICKNESS)
                mask_dbg = draw_mask_slices(res)

                cv2.putText(
                    dbg,
                    f"STATE: {state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("camera debug", dbg)
                cv2.imshow("mask debug", mask_dbg)

            # ========================================================
            # 2) SHOOTING PHASE
            # ========================================================
            elif state == STATE_SHOOTING:
                leds.turn_on_all()

                blue_mask, targets = detect_blue_targets(frame)

                # ---- hit confirmation after a shot ----
                if pending_shot:
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

                        # target disappeared
                        if matched is None or dist > TARGET_MATCH_DIST_PX:
                            hit_confirmed = True
                        else:
                            # target still there but much smaller
                            if matched["area"] < HIT_AREA_RATIO_THRESHOLD * old["area"]:
                                hit_confirmed = True

                        if hit_confirmed:
                            targets_hit += 1

                        pending_shot = False
                        pending_target_before_shot = None
                        lock_frames = 0
                        last_target_center = None

                # ---- normal aiming / shooting ----
                else:
                    target = pick_target(targets, last_target_center)

                    if target is None:
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
                                pending_shot = True
                                pending_target_before_shot = {
                                    "cx": target["cx"],
                                    "cy": target["cy"],
                                    "area": target["area"],
                                }
                                confirm_delay_frames = POST_SHOT_CONFIRM_FRAMES
                                time.sleep(POST_SHOT_SETTLE_TIME)

                mission_done = (targets_hit >= NOMBRE_DE_CIBLES) or shooter.is_empty()

                if mission_done and not pending_shot:
                    motors.set_speeds(0, 0)
                    shooter.spin_flywheel(False)
                    time.sleep(0.2)

                    state = STATE_RETURN_TO_T
                    return_to_t_ignore_until = time.time() + RETURN_T_IGNORE_SECONDS

                dbg = frame.copy()

                for t in targets:
                    x, y, w, h = t["bbox"]
                    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(dbg, (t["cx"], t["cy"]), 4, (0, 0, 255), -1)

                if pending_target_before_shot is not None:
                    cv2.circle(
                        dbg,
                        (pending_target_before_shot["cx"], pending_target_before_shot["cy"]),
                        12,
                        (0, 255, 255),
                        2,
                    )

                cv2.line(dbg, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)

                cv2.putText(
                    dbg,
                    f"STATE: {state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    dbg,
                    f"HITS: {targets_hit}/{NOMBRE_DE_CIBLES}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    dbg,
                    f"AMMO: {shooter.ammo_remaining()}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    dbg,
                    f"PENDING_SHOT: {pending_shot}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("camera debug", dbg)
                cv2.imshow("mask debug", blue_mask)

            # ========================================================
            # 3) RETURN TO T
            # ========================================================
            elif state == STATE_RETURN_TO_T:
                res = detect_line(frame, angle_deg=20)
                left, right, _, _, mode = controller.update(res)

                leds.turn_on_all()
                motors.set_speeds(left, right)

                t_detected = (mode == "T_STOP") or res.t_detected

                if time.time() >= return_to_t_ignore_until and t_detected:
                    motors.set_speeds(0, 0)
                    state = STATE_AURA
                    aura_mode = False

                dbg = draw_debug_slices(frame, res, BAND_THICKNESS)
                mask_dbg = draw_mask_slices(res)

                cv2.putText(
                    dbg,
                    f"STATE: {state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    dbg,
                    f"HITS: {targets_hit}/{NOMBRE_DE_CIBLES}  AMMO: {shooter.ammo_remaining()}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("camera debug", dbg)
                cv2.imshow("mask debug", mask_dbg)

            # ========================================================
            # 4) FINAL AURA
            # ========================================================
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
                cv2.putText(
                    aura_dbg,
                    f"HITS: {targets_hit}/{NOMBRE_DE_CIBLES}  SHOTS: {shooter.shots_fired}/{MAG_CAPACITY}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
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
