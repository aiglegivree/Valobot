# tir_auto.py
from __future__ import annotations
import time
import math
import cv2
import numpy as np
import RPi.GPIO as GPIO
from leds import LED

from config import (
    W, H,
    BLUE_LOW, BLUE_HIGH,
    MIN_TARGET_AREA,
    TURN_SPEED, CENTER_TOL, LOCK_FRAMES_REQUIRED,
    SHOW_DEBUG,
    MOTEUR_PIN, SERVO_PIN,
    PWM_FREQ, ANGLE_LOADED, ANGLE_FIRE_START, ANGLE_INCREMENT,
    MAG_CAPACITY, MIN_TIME_BETWEEN_SHOTS,
    POST_SHOT_SETTLE_TIME, POST_SHOT_CONFIRM_FRAMES,
    TARGET_MATCH_DIST_PX,
    NOMBRE_DE_CIBLES,
)
from camera import Camera
from motors import MotorDriver

leds = LED()


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

        self.angle = ANGLE_FIRE_START
        self.first_fire = False
        self.motor_on = False

        GPIO.output(MOTEUR_PIN, GPIO.LOW)

    def set_servo_angle(self, angle: float):
        duty_cycle = (angle / 18.0) + 2.5
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.30)
        self.pwm.ChangeDutyCycle(0)

    def load_charger(self):
        self.angle = ANGLE_LOADED
        self.first_fire = True
        self.set_servo_angle(self.angle)
        print("Reloaded")

    def fire(self):
        if self.first_fire:
            self.angle = ANGLE_FIRE_START
            self.first_fire = False
        else:
            self.angle += ANGLE_INCREMENT

        self.set_servo_angle(self.angle)
        print("FIRE")

    def motor_start(self):
        self.motor_on = True
        GPIO.output(MOTEUR_PIN, GPIO.HIGH)
        print("Shooter motor ON")

    def motor_stop(self):
        self.motor_on = False
        GPIO.output(MOTEUR_PIN, GPIO.LOW)
        print("Shooter motor OFF")

    def cleanup(self):
        self.motor_stop()
        self.pwm.stop()
        GPIO.cleanup()


# ============================================================
# VISION
# ============================================================
def detect_blue_targets(frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets: list[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_TARGET_AREA:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)

        targets.append({
            "cx": cx,
            "cy": cy,
            "area": area,
            "bbox": (x, y, w, h),
            "contour": c,
        })

    targets.sort(key=lambda t: t["area"], reverse=True)
    return targets, mask


def choose_target(targets: list[dict], image_center_x: int) -> dict | None:
    if not targets:
        return None

    best_score = None
    best_target = None

    for t in targets:
        err = abs(t["cx"] - image_center_x)
        score = t["area"] - 3.0 * err

        if best_score is None or score > best_score:
            best_score = score
            best_target = t

    return best_target


def target_distance(t1: dict, t2: dict) -> float:
    return math.hypot(t1["cx"] - t2["cx"], t1["cy"] - t2["cy"])


def find_same_target(targets: list[dict], old_target: dict, max_dist_px: float) -> dict | None:
    best = None
    best_d = None

    for t in targets:
        d = target_distance(t, old_target)
        if d <= max_dist_px and (best_d is None or d < best_d):
            best = t
            best_d = d

    return best


# ============================================================
# POST SHOT CONFIRMATION
# ============================================================
def confirm_target_hit(
    cam: Camera,
    old_target: dict
) -> tuple[bool, np.ndarray | None, np.ndarray | None, list[dict] | None]:
    """
    On confirme le hit si la cible visée n'est plus retrouvée
    sur plusieurs frames après le tir.
    """
    last_frame = None
    last_mask = None
    last_targets = None

    time.sleep(POST_SHOT_SETTLE_TIME)

    missing_count = 0
    checked_frames = 0

    for _ in range(POST_SHOT_CONFIRM_FRAMES):
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        checked_frames += 1
        frame = cv2.resize(frame, (W, H))
        targets, mask = detect_blue_targets(frame)

        last_frame = frame
        last_mask = mask
        last_targets = targets

        same = find_same_target(targets, old_target, TARGET_MATCH_DIST_PX)

        if same is None:
            missing_count += 1

        time.sleep(0.05)

    if checked_frames == 0:
        return False, last_frame, last_mask, last_targets

    # Validation plus simple et plus robuste
    hit_confirmed = missing_count >= 2
    return hit_confirmed, last_frame, last_mask, last_targets


# ============================================================
# LED END ANIMATION
# ============================================================
def run_end_animation(duration_s: float = 10.0):
    try:
        leds.aura_farm()
        time.sleep(duration_s)
    finally:
        leds.turn_off_all()


# ============================================================
# AUTO FIRE
# ============================================================
def auto_fire_blue_targets(
    cam: Camera,
    drive: MotorDriver,
    shooter: Shooter,
    nb_targets_to_hit: int,
) -> tuple[int, int]:
    """
    Tire automatiquement jusqu'à:
    - nb_targets_to_hit cibles touchées
    - ou chargeur vide

    Retourne:
      (targets_hit, shots_fired)
    """
    targets_hit = 0
    shots_fired = 0
    lock_count = 0
    last_shot_time = -1e9
    status = "init"

    center_x = W // 2

    shooter.load_charger()
    if not shooter.motor_on:
        shooter.motor_start()
        time.sleep(1.0)

    last_frame = None
    last_mask = np.zeros((H, W), dtype=np.uint8)
    last_targets: list[dict] = []

    while targets_hit < nb_targets_to_hit and shots_fired < MAG_CAPACITY:
        ret, frame = cam.read()
        if not ret or frame is None:
            drive.set_speeds(0, 0)
            status = "no frame"
            continue

        frame = cv2.resize(frame, (W, H))
        targets, mask = detect_blue_targets(frame)

        last_frame = frame
        last_mask = mask
        last_targets = targets

        target = choose_target(targets, center_x)

        if target is None:
            drive.set_speeds(-TURN_SPEED, TURN_SPEED)
            lock_count = 0
            status = f"search | hits={targets_hit} shots={shots_fired}/{MAG_CAPACITY}"

        else:
            err = target["cx"] - center_x

            if abs(err) <= CENTER_TOL:
                drive.set_speeds(0, 0)
                lock_count += 1
                status = f"locked {lock_count}/{LOCK_FRAMES_REQUIRED} | hits={targets_hit} shots={shots_fired}/{MAG_CAPACITY}"

                can_fire = (time.time() - last_shot_time) >= MIN_TIME_BETWEEN_SHOTS

                if lock_count >= LOCK_FRAMES_REQUIRED and can_fire:
                    aimed_target = dict(target)

                    shooter.fire()
                    shots_fired += 1
                    last_shot_time = time.time()
                    lock_count = 0

                    hit, frame2, mask2, targets2 = confirm_target_hit(cam, aimed_target)

                    if frame2 is not None:
                        last_frame = frame2
                    if mask2 is not None:
                        last_mask = mask2
                    if targets2 is not None:
                        last_targets = targets2

                    if hit:
                        targets_hit += 1
                        status = f"HIT confirmed | hits={targets_hit} shots={shots_fired}/{MAG_CAPACITY}"
                        print(f"HIT confirmed -> {targets_hit}/{nb_targets_to_hit}")

                        # arrêt immédiat si mission finie
                        if targets_hit >= nb_targets_to_hit:
                            drive.set_speeds(0, 0)
                            break
                    else:
                        status = f"MISS / target still present | hits={targets_hit} shots={shots_fired}/{MAG_CAPACITY}"
                        print("Miss: target still present")

            else:
                lock_count = 0

                if err < 0:
                    drive.set_speeds(-TURN_SPEED, TURN_SPEED)
                    status = f"turn left | hits={targets_hit} shots={shots_fired}/{MAG_CAPACITY}"
                else:
                    drive.set_speeds(TURN_SPEED, -TURN_SPEED)
                    status = f"turn right | hits={targets_hit} shots={shots_fired}/{MAG_CAPACITY}"

        if SHOW_DEBUG and last_frame is not None:
            debug = last_frame.copy()

            cv2.line(debug, (center_x, 0), (center_x, H), (0, 255, 0), 2)
            cv2.line(debug, (center_x - CENTER_TOL, 0), (center_x - CENTER_TOL, H), (255, 255, 0), 1)
            cv2.line(debug, (center_x + CENTER_TOL, 0), (center_x + CENTER_TOL, H), (255, 255, 0), 1)

            for i, t in enumerate(last_targets):
                x, y, w, h = t["bbox"]
                color = (0, 255, 255) if i == 0 else (180, 180, 0)
                cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
                cv2.circle(debug, (t["cx"], t["cy"]), 4, (0, 0, 255), -1)
                cv2.putText(
                    debug,
                    f"A={int(t['area'])}",
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            cv2.putText(
                debug,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                debug,
                f"Hits {targets_hit}/{nb_targets_to_hit} | Shots {shots_fired}/{MAG_CAPACITY}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow("debug", debug)
            cv2.imshow("blue_mask", last_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Cancelled by user")
                break

    drive.set_speeds(0, 0)

    if targets_hit >= nb_targets_to_hit:
        print("MISSION COMPLETE")
    elif shots_fired >= MAG_CAPACITY:
        print("OUT OF AMMO")

    return targets_hit, shots_fired


# ============================================================
# MAIN
# ============================================================
def main():
    cam = Camera()
    drive = MotorDriver()
    shooter = Shooter()
    leds.turn_off_all()

    try:
        hits, shots = auto_fire_blue_targets(
            cam=cam,
            drive=drive,
            shooter=shooter,
            nb_targets_to_hit=NOMBRE_DE_CIBLES,
        )
        print(f"Final result: hits={hits}, shots={shots}")

        run_end_animation(10.0)

    finally:
        drive.set_speeds(0, 0)
        shooter.cleanup()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()