import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor A
ENA = 18
IN1 = 23
IN2 = 24

# Motor B
ENB = 13
IN3 = 27
IN4 = 22

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

        time.sleep(3)

        print("Stop")
        stop()
        time.sleep(2)

        print("Backward slow")

        GPIO.output(IN1,0)
        GPIO.output(IN2,1)
        GPIO.output(IN3,0)
        GPIO.output(IN4,1)

        pwmA.ChangeDutyCycle(speed)
        pwmB.ChangeDutyCycle(speed)

        time.sleep(3)

        print("Stop")
        stop()
        time.sleep(3)

except KeyboardInterrupt:
    stop()
    GPIO.cleanup()
