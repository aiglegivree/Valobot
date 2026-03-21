import RPi.GPIO as GPIO
import time


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

    def aura_farm(self, stop=False, delay=0.2):
        # Stop the running pattern
        if stop:
            self._running = False
            return
        
        self.turn_off_all()
        self._running = True

        while self._running:
            # Forward: 23 → 24 → 25 → 8
            for pin in self.pins:
                if not self._running:
                    break
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(pin, GPIO.LOW)

            # Backward: 8 → 25 → 24 → 23
            for pin in reversed(self.pins):
                if not self._running:
                    break
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(pin, GPIO.LOW)

    def cleanup(self):
        GPIO.cleanup()