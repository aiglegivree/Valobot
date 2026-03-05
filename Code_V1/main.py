# main.py
from __future__ import annotations
import cv2
import time

from config import CAM_ID, W, H
from vision import measure_line
from controller import LineController
from motors import MotorDriver

def mix_speed_steer(speed: float, steer: float) -> tuple[float, float]:
    """
    Convert speed (0..100) + steer (-1..+1) to left/right speeds (-100..+100).
    Differential mixing:
      left  = speed * (1 - steer)
      right = speed * (1 + steer)
    """
    left = speed * (1.0 - steer)
    right = speed * (1.0 + steer)
    # Clamp inside motors.py anyway
    return left, right

def main() -> None:
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    if not cap.isOpened():
        raise RuntimeError("Camera not opened")

    motors = MotorDriver()
    ctrl = LineController()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                motors.stop()
                time.sleep(0.02)
                continue

            meas = measure_line(frame)
            out = ctrl.update(meas.found, meas.e_px)

            left, right = mix_speed_steer(out.speed, out.steer)
            motors.set_speeds(left, right)

            # Optional debug display (disable on headless)
            # cv2.imshow("frame", cv2.resize(frame, (W, H)))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        motors.stop()
        motors.cleanup()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()