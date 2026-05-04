from __future__ import annotations

import select
import sys
import termios
import time
import tty

import cv2
from picamera2 import Picamera2

from motors import MotorDriver


picam2 = Picamera2()
picam2.start()

SPEED = 0.70
DELTA = 1.0
KEY_TIMEOUT = 0.15
TURN_DURATION = 0.50


def get_key():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def main():
    motor = MotorDriver()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    print("Hold keys: Z S | Q/D timed turn | SPACE stop | A quit")

    last_key_time = time.time()
    left = 0.0
    right = 0.0
    turn_until = 0.0

    try:
        while True:
            key = get_key()

            frame = picam2.capture_array()
            cv2.imshow("PiCam", frame)

            if cv2.waitKey(1) & 0xFF == ord("a"):
                break

            now = time.time()

            if key is not None:
                last_key_time = now

                if key.lower() == "z":
                    left = SPEED
                    right = DELTA * SPEED
                    turn_until = 0.0
                    motor.set_speeds(left, right)

                elif key.lower() == "s":
                    left = -SPEED
                    right = -DELTA * SPEED
                    turn_until = 0.0
                    motor.set_speeds(left, right)

                elif key.lower() == "q":
                    left = -SPEED
                    right = DELTA * SPEED
                    turn_until = now + TURN_DURATION
                    motor.set_speeds(left, right)

                elif key.lower() == "d":
                    left = SPEED
                    right = -DELTA * SPEED
                    turn_until = now + TURN_DURATION
                    motor.set_speeds(left, right)

                elif key == " ":
                    left = 0.0
                    right = 0.0
                    turn_until = 0.0
                    motor.set_speeds(left, right)

                elif key.lower() == "a":
                    break

            if turn_until > 0.0 and now >= turn_until:
                left = 0.0
                right = 0.0
                turn_until = 0.0
                motor.set_speeds(left, right)

            if turn_until == 0.0 and now - last_key_time > KEY_TIMEOUT:
                motor.set_speeds(0.0, 0.0)

            time.sleep(0.01)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        motor.cleanup()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
