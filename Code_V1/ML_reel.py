# Simple test: robot turns right then left every 2 seconds

import time
from motors import MotorDriver   # your existing motor driver class

TURN_SPEED = 0.6

motor = MotorDriver()

try:
    while True:
        # Turn right
        motor.set_speeds(TURN_SPEED, -TURN_SPEED)
        time.sleep(2)

        # Turn left
        motor.set_speeds(-TURN_SPEED, TURN_SPEED)
        time.sleep(2)

except KeyboardInterrupt:
    motor.stop()
