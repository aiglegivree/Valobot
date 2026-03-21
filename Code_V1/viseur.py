# controller.py
from __future__ import annotations
import cv2
import numpy as np
import time

from config import W, H
from camera import Camera
from motors import MotorDriver

# ============================================================
# CONFIG
# ============================================================
BLUE_LOW = np.array([90, 80, 50], dtype=np.uint8)
BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

MIN_TARGET_AREA = 1200
TURN_SPEED = 0.28
CENTER_TOL = 35
LOCK_FRAMES_REQUIRED = 5
SHOW_DEBUG = True


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


def run():
    cam = Camera()
    motor = MotorDriver()
    lock_count = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                motor.set_speed(0, 0)
                continue

            frame = cv2.resize(frame, (W, H))
            found, cx, cy, area, mask, bbox = detect_blue_target(frame)

            center_x = W // 2

            if not found:
                # Search: rotate on the spot
                motor.set_speed(-TURN_SPEED, TURN_SPEED)
                lock_count = 0
                status = "search"

            else:
                err = cx - center_x

                if abs(err) <= CENTER_TOL:
                    motor.set_speed(0, 0)
                    lock_count += 1
                    status = f"locked {lock_count}/{LOCK_FRAMES_REQUIRED}"

                    if lock_count >= LOCK_FRAMES_REQUIRED:
                        print("fire")
                        break

                else:
                    lock_count = 0

                    if err < 0:
                        # target is on the left
                        motor.set_speed(-TURN_SPEED, TURN_SPEED)
                        status = "turn left"
                    else:
                        # target is on the right
                        motor.set_speed(TURN_SPEED, -TURN_SPEED)
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
                    cv2.putText(debug, f"cx={cx} area={int(area)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(debug, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("debug", debug)
                cv2.imshow("blue_mask", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

    finally:
        motor.set_speed(0, 0)
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
