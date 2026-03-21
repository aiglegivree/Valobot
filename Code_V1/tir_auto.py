# controller.py
from __future__ import annotations
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

from config import W, H
from camera import Camera
from motors import MotorDriver

# ============================================================
# VISION CONFIG
# ============================================================
BLUE_LOW  = np.array([95, 70, 30], dtype=np.uint8)
BLUE_HIGH = np.array([140, 255, 255], dtype=np.uint8)

MIN_TARGET_AREA = 1200
TURN_SPEED = 0.28
CENTER_TOL = 35
LOCK_FRAMES_REQUIRED = 5
SHOW_DEBUG = True

# ============================================================
# SHOOTER CONFIG
# ============================================================
MOTEUR_PIN = 19
SERVO_PIN = 18

PWM_FREQ = 50
ANGLE_LOADED = 21
ANGLE_FIRE_START = 10
ANGLE_INCREMENT = 23


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
        self.load_charger()

    def set_servo_angle(self, angle: float):
        duty_cycle = (angle / 18.0) + 2.5
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.3)
        self.pwm.ChangeDutyCycle(0)   # reduce jitter

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

    def toggle_motor(self):
        if self.motor_on:
            self.motor_stop()
        else:
            self.motor_start()

    def cleanup(self):
        self.motor_stop()
        self.pwm.stop()
        GPIO.cleanup()


def detect_blue_target(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, None, None, 0, mask, None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < MIN_TARGET_AREA:
        return False, None, None, area, mask, None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return False, None, None, area, mask, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(c)

    return True, cx, cy, area, mask, (x, y, w, h)


def aim_and_fire(cam: Camera, drive: MotorDriver, shooter: Shooter):
    lock_count = 0
    status = "search"

    if not shooter.motor_on:
        shooter.motor_start()
        time.sleep(1.0)   # flywheel spin-up

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            drive.set_speeds(0, 0)
            continue

        frame = cv2.resize(frame, (W, H))
        found, cx, cy, area, mask, bbox = detect_blue_target(frame)

        center_x = W // 2

        if not found:
            drive.set_speeds(-TURN_SPEED, TURN_SPEED)
            lock_count = 0
            status = "search"

        else:
            err = cx - center_x

            if abs(err) <= CENTER_TOL:
                drive.set_speeds(0, 0)
                lock_count += 1
                status = f"locked {lock_count}/{LOCK_FRAMES_REQUIRED}"

                if lock_count >= LOCK_FRAMES_REQUIRED:
                    status = "FIRE"
                    print("Target locked -> fire")
                    shooter.fire()
                    time.sleep(0.3)
                    return

            else:
                lock_count = 0

                if err < 0:
                    # target is on the left
                    drive.set_speeds(-TURN_SPEED, TURN_SPEED)
                    status = "turn left"
                else:
                    # target is on the right
                    drive.set_speeds(TURN_SPEED, -TURN_SPEED)
                    status = "turn right"

        if SHOW_DEBUG:
            debug = frame.copy()

            cv2.line(debug, (center_x, 0), (center_x, H), (0, 255, 0), 2)
            cv2.line(debug, (center_x - CENTER_TOL, 0), (center_x - CENTER_TOL, H), (255, 255, 0), 1)
            cv2.line(debug, (center_x + CENTER_TOL, 0), (center_x + CENTER_TOL, H), (255, 255, 0), 1)

            if found and bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    debug,
                    f"cx={cx} area={int(area)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                debug,
                status,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("debug", debug)
            cv2.imshow("blue_mask", mask)

            key = cv2.waitKey(1) & 0xFF

            # ESC cancels aim mode and returns to menu
            if key == 27:
                drive.set_speeds(0, 0)
                print("Aim cancelled")
                return


def main():
    cam = Camera()
    drive = MotorDriver()
    shooter = Shooter()

    try:
        print("Commands:")
        print("  r = reload")
        print("  f = aim at blue then fire")
        print("  m = toggle shooter motor")
        print("  q = quit")

        while True:
            cmd = input(">> ").strip().lower()

            if cmd == "r":
                drive.set_speeds(0, 0)
                shooter.load_charger()

            elif cmd == "f":
                print("Aim mode started")
                aim_and_fire(cam, drive, shooter)
                drive.set_speeds(0, 0)
                print("Back to menu")

            elif cmd == "m":
                shooter.toggle_motor()

            elif cmd == "q":
                break

            else:
                print("Unknown command")

    finally:
        drive.set_speeds(0, 0)
        shooter.cleanup()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()