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
            GPIO.output(pin, GPIO.HIGH)  # OFF

        self._running = False
        self._index = 0
        self._direction = 1
        self._last_step_time = 0.0

    def turn_on_all(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)

    def turn_off_all(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.HIGH)

    def aura_farm_start(self):
        self.turn_off_all()
        self._running = True
        self._index = 0
        self._direction = 1
        self._last_step_time = 0.0

    def aura_farm_stop(self):
        self._running = False
        self.turn_off_all()

    def aura_farm_step(self, delay=0.05):
        if not self._running:
            return

        now = time.time()
        if now - self._last_step_time < delay:
            return

        self._last_step_time = now

        self.turn_off_all()
        GPIO.output(self.pins[self._index], GPIO.LOW)

        self._index += self._direction

        if self._index >= len(self.pins):
            self._index = len(self.pins) - 2
            self._direction = -1
        elif self._index < 0:
            self._index = 1
            self._direction = 1

    def cleanup(self):
        GPIO.cleanup()
        
led = LED() 

while True :
	led.turn_on_all()


"""
led.aura_farm_start()
while True :
	led.aura_farm_step()
	time.sleep(0.02)
	"""
