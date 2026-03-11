import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor A
ENA = 13
IN1 = 14
IN2 = 15

# Motor B
ENB = 12
IN3 = 16
IN4 = 17

pins = [ENA,IN1,IN2,ENB,IN3,IN4]

for p in pins:
    GPIO.setup(p,GPIO.OUT)

# PWM for speed control
pwmA = GPIO.PWM(ENA,1000)
pwmB = GPIO.PWM(ENB,1000)

pwmA.start(0)
pwmB.start(0)

speed = 30   # slow speed (0-100)

def stop():
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    GPIO.output(IN1,0)
    GPIO.output(IN2,0)
    GPIO.output(IN3,0)
    GPIO.output(IN4,0)

try:

    while True:

        print("Forward slow")

        GPIO.output(IN1,1)
        GPIO.output(IN2,0)
        GPIO.output(IN3,1)
        GPIO.output(IN4,0)

        pwmA.ChangeDutyCycle(speed)
        pwmB.ChangeDutyCycle(speed)



except KeyboardInterrupt:
    stop()
    GPIO.cleanup()
