import RPi.GPIO as GPIO
import time

# Pins
MOTEUR_PIN = 19
SERVO_PIN = 18

PWM_FREQ = 50
ANGLE_LOADED = 21
ANGLE_FIRE_START = 10
ANGLE_INCREMENT = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTEUR_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
pwm.start(0)

angle = ANGLE_FIRE_START
first_fire = False
motor_on = False


def set_servo_angle(angle):
    duty_cycle = (angle / 18) + 2.5
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)


def load_charger():
    global angle, first_fire
    angle = ANGLE_LOADED
    first_fire = True
    set_servo_angle(angle)
    print("Reloaded")


def fire():
    global angle, first_fire
    if first_fire:
        angle = ANGLE_FIRE_START
        first_fire = False
    else:
        angle += ANGLE_INCREMENT
    set_servo_angle(angle)
    print("FIRE")


def toggle_motor():
    global motor_on
    motor_on = not motor_on
    GPIO.output(MOTEUR_PIN, GPIO.HIGH if motor_on else GPIO.LOW)
    print("Motor ON" if motor_on else "Motor OFF")


try:
    GPIO.output(MOTEUR_PIN, GPIO.LOW)
    load_charger()

    print("Commands: r=reload | m=motor | t=fire | q=quit")

    while True:
        cmd = input(">> ").strip().lower()

        if cmd == 'r':
            load_charger()

        elif cmd == 'm':
            toggle_motor()

        elif cmd == 't':
            fire()

        elif cmd == 'q':
            break

except KeyboardInterrupt:
    pass

finally:
    pwm.stop()
    GPIO.cleanup()