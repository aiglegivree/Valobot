from camera import Camera
from vision import detect_line
from controller import Controller
from motors import MotorDriver

cam = Camera()
controller = Controller()
motors = MotorDriver()

while True:

    ret, frame = cam.read()

    if not ret:
        continue

    vision = detect_line(frame)

    left, right, speed, steer, mode = controller.update(vision)

    motors.set_speeds(left, right)