"""
This Raspberry Pi code was developed by newbiely.com
This Raspberry Pi code is made available for public use without any restriction
For comprehensive instructions and wiring diagrams, please visit:
https://newbiely.com/tutorials/raspberry-pi/raspberry-pi-servo-motor
"""


import RPi.GPIO as GPIO
import time

# Set the GPIO mode (BCM or BOARD)
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin connected to the servo motor
SERVO_PIN = 18

# Set the servo pin as an output pin
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Set up PWM
pwm_frequency = 50  # Set PWM frequency to 50Hz (standard for most servos)
pwm = GPIO.PWM(SERVO_PIN, pwm_frequency)
pwm.start(0)

angle = 10
first = False

def set_servo_angle(angle):
    duty_cycle = (angle / 18) + 2.5
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)  # Give the servo time to reach the desired angle
    
def load_charger():
    global angle, first
    angle = 21
    first = True
    set_servo_angle(angle)

def fire():
    global first, angle
    if first:
        angle = 10
        set_servo_angle(angle)
        first = False    
    else:
        angle += 23
        set_servo_angle(angle)


load_charger()
print(angle)
time.sleep(1)
for _ in range(7):
    fire()
    time.sleep(1)
    
load_charger()
time.sleep(1)
