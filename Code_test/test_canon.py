import RPi.GPIO as GPIO
import time

# Pins
MOTEUR_PIN = 19
SERVO_PIN = 18

# Servo PWM setup
PWM_FREQ = 50  # Hz
ANGLE_LOADED = 21
ANGLE_FIRE_START = 10
ANGLE_INCREMENT = 23

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTEUR_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Initialize PWM
pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
pwm.start(0)

# State
angle = ANGLE_FIRE_START
first_fire = False

def set_servo_angle(angle):
    duty_cycle = (angle / 18) + 2.5
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)  # Wait for servo to reach position

def load_charger():
    global angle, first_fire
    angle = ANGLE_LOADED
    first_fire = True
    set_servo_angle(angle)

def fire():
    global angle, first_fire
    if first_fire:
        angle = ANGLE_FIRE_START
        first_fire = False
    else:
        angle += ANGLE_INCREMENT
    set_servo_angle(angle)

try:
    # Motor off at start
	GPIO.output(MOTEUR_PIN, GPIO.LOW)
	print('Charger')
    # Load charger once
	load_charger()
	time.sleep(10)
	print('Moteur')
	# Turn motor on
	GPIO.output(MOTEUR_PIN, GPIO.HIGH)
	time.sleep(3)

    # Fire 5 shots with 1 second interval
	for _ in range(5):
		print('BRRRRRRAAAA')
		fire()
		time.sleep(0.5)

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    pwm.stop()
    GPIO.cleanup()
