from __future__ import annotations
import sys
import termios
import tty
import select
import time

from motors import MotorDriver   # <-- ton fichier avec la classe

SPEED = 0.6
TURN = 0.5

def get_key():
    """Lecture clavier non bloquante"""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def main():
    motor = MotorDriver()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    print("Control robot: W S A D | SPACE stop | Q quit")

    left = 0.0
    right = 0.0

    try:
        while True:
            key = get_key()

            if key is not None:
                if key.lower() == 'w':
                    left = SPEED
                    right = SPEED

                elif key.lower() == 's':
                    left = -SPEED
                    right = -SPEED

                elif key.lower() == 'a':
                    left = -TURN
                    right = TURN

                elif key.lower() == 'd':
                    left = TURN
                    right = -TURN

                elif key == ' ':
                    left = 0.0
                    right = 0.0

                elif key.lower() == 'q':
                    break

                motor.set_speeds(left, right)

            time.sleep(0.02)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        motor.cleanup()


if __name__ == "__main__":
    main()