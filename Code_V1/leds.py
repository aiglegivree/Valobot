import RPi.GPIO as GPIO


class LED:
    def __init__(self, pins=None):
        if pins is None:
            pins = [23, 24, 25, 8]

        self.pins = pins

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)  # start OFF

    def turn_on_all(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.HIGH)

    def turn_off_all(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)

    def cleanup(self):
        GPIO.cleanup()