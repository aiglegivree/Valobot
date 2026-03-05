# motors.py

import RPi.GPIO as GPIO
from config import (
    ENA, IN1, IN2,
    ENB, IN3, IN4,
    PWM_FREQ,
    INVERT_LEFT_MOTOR,
    INVERT_RIGHT_MOTOR
)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class MotorDriver:

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for p in [ENA, IN1, IN2, ENB, IN3, IN4]:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, GPIO.LOW)

        self.pwmA = GPIO.PWM(ENA, PWM_FREQ)
        self.pwmB = GPIO.PWM(ENB, PWM_FREQ)

        self.pwmA.start(0)
        self.pwmB.start(0)

    def set_speeds(self, left_speed, right_speed):
        """
        left_speed, right_speed in range [-100 .. 100]
        sign = direction
        magnitude = PWM duty
        """

        if INVERT_LEFT_MOTOR:
            left_speed = -left_speed

        if INVERT_RIGHT_MOTOR:
            right_speed = -right_speed

        # ----- LEFT MOTOR -----
        ls = int(clamp(left_speed, -100, 100))

        if ls >= 0:
            GPIO.output(IN1, 1)
            GPIO.output(IN2, 0)
        else:
            GPIO.output(IN1, 0)
            GPIO.output(IN2, 1)

        self.pwmA.ChangeDutyCycle(abs(ls))

        # ----- RIGHT MOTOR -----
        rs = int(clamp(right_speed, -100, 100))

        if rs >= 0:
            GPIO.output(IN3, 1)
            GPIO.output(IN4, 0)
        else:
            GPIO.output(IN3, 0)
            GPIO.output(IN4, 1)

        self.pwmB.ChangeDutyCycle(abs(rs))

    def stop(self):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)

        GPIO.output(IN1, 0)
        GPIO.output(IN2, 0)
        GPIO.output(IN3, 0)
        GPIO.output(IN4, 0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()