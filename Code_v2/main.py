# main.py
from __future__ import annotations
import cv2

from camera import Camera
from vision import detect_line, draw_debug_slices, draw_mask_slices
from controller import Controller
from motors import MotorDriver
from leds import LED
from config import SHOW_DEBUG, BAND_THICKNESS


def main():
    cam = Camera()
    controller = Controller()
    motors = MotorDriver()
    leds = LED()

    t_reached = False
    aura_mode = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # 1) vision
            res = detect_line(frame, angle_deg=20)

            # 2) control
            left, right, speed, steer, mode = controller.update(res)

            # 3) behavior
            if mode == "T_STOP":
                if not t_reached:
                    t_reached = True
                    motors.stop()
                if not aura_mode:
                    aura_mode = True
                    leds.aura_farm_start()

                # keep aura farm running once T is reached
                leds.aura_farm_step()

            else:
                if aura_mode:
                    aura_mode = False
                    leds.aura_farm_stop()
                motors.set_speeds(left, right)
                leds.turn_on_all()

            # 4) debug
            if SHOW_DEBUG:
                dbg = draw_debug_slices(frame, res, BAND_THICKNESS)
                mask_vis = draw_mask_slices(res)

                cv2.imshow("debug", dbg)
                cv2.imshow("mask", mask_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        motors.stop()

        try:
            leds.aura_farm_stop()
        except Exception:
            pass

        try:
            cam.release()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
