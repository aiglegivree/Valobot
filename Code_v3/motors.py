from __future__ import annotations
import RPi.GPIO as GPIO

# Based on your working test code

# Right motor
RIGHT_EN = 13
RIGHT_IN1 = 14
RIGHT_IN2 = 15

# Left motor
LEFT_EN = 12
LEFT_IN1 = 16
LEFT_IN2 = 17

PWM_FREQ = 1000

LEFT_TRIM = 1.00
RIGHT_TRIM = 1.00   # same correction as your working test


class MotorDriver:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in [RIGHT_EN, RIGHT_IN1, RIGHT_IN2, LEFT_EN, LEFT_IN1, LEFT_IN2]:
            GPIO.setup(pin, GPIO.OUT)

        self.pwm_right = GPIO.PWM(RIGHT_EN, PWM_FREQ)
        self.pwm_left = GPIO.PWM(LEFT_EN, PWM_FREQ)

        self.pwm_right.start(0)
        self.pwm_left.start(0)

        self.stop()

    def _set_one(self, pwm, in1, in2, value: float):
        value = max(-1.0, min(1.0, value))
        duty = abs(value) * 100.0

        # forward direction = same as your working code: in1=1, in2=0
        if value > 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif value < 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        pwm.ChangeDutyCycle(duty)

    def set_speeds(self, left: float, right: float):
        left *= LEFT_TRIM
        right *= RIGHT_TRIM

        self._set_one(self.pwm_left, LEFT_IN1, LEFT_IN2, left)
        self._set_one(self.pwm_right, RIGHT_IN1, RIGHT_IN2, right)

    def stop(self):
        self._set_one(self.pwm_left, LEFT_IN1, LEFT_IN2, 0.0)
        self._set_one(self.pwm_right, RIGHT_IN1, RIGHT_IN2, 0.0)

    def cleanup(self):
        self.stop()
        self.pwm_left.stop()
        self.pwm_right.stop()
        GPIO.cleanup()