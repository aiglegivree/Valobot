from camera import Camera
from vision import detect_line, draw_debug_slices, draw_mask_slices
from controller import Controller
from motors import MotorDriver
from leds import LED
import cv2
from config import BAND_THICKNESS


cam = Camera()
controller = Controller()
motors = MotorDriver()
leds = LED()

aura_mode = False

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # 1) Detect line / T
        res = detect_line(frame, angle_deg=20)

        # 2) Controller
        left, right, speed, steer, mode = controller.update(res)

        # 3) Motors + LEDs
        if mode == "T_STOP":
            if not aura_mode:
                aura_mode = True
                motors.set_speeds(0, 0)
                leds.aura_farm_start()

            leds.aura_farm_step(delay=0.05)

        else:
            if aura_mode:
                aura_mode = False
                leds.aura_farm_stop()

            leds.turn_on_all()
            motors.set_speeds(left, right)

        # 4) Debug
        roi = frame[res.roi_start:, :]
        dbg = draw_debug_slices(roi, res, BAND_THICKNESS)
        mask_dbg = draw_mask_slices(res.mask, res, BAND_THICKNESS)

        cv2.imshow("camera debug", dbg)
        cv2.imshow("mask debug", mask_dbg)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

finally:
    leds.turn_off_all()
    motors.set_speeds(0, 0)
    cam.release()
    cv2.destroyAllWindows()