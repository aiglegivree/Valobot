from __future__ import annotations
import sys
import termios
import tty
import select
import time

from motors import MotorDriver

SPEED = 0.6
TURN = 0.5
KEY_TIMEOUT = 0.15   # temps max sans input → stop


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

    print("Hold keys: Z S Q D | SPACE stop | A quit")

    last_key_time = time.time()
    left = 0.0
    right = 0.0

    try:
        while True:
            key = get_key()

            if key is not None:
                last_key_time = time.time()

                if key.lower() == 'z':
                    left = SPEED
                    right = SPEED

                elif key.lower() == 's':
                    left = -SPEED
                    right = -SPEED

                elif key.lower() == 'q':
                    left = -TURN
                    right = TURN

                elif key.lower() == 'd':
                    left = TURN
                    right = -TURN

                elif key == ' ':
                    left = 0.0
                    right = 0.0

                elif key.lower() == 'a':
                    break

                motor.set_speeds(left, right)

            # sécurité : si on relâche → stop
            if time.time() - last_key_time > KEY_TIMEOUT:
                motor.set_speeds(0, 0)

            time.sleep(0.01)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        motor.cleanup()


if __name__ == "__main__":
    main()