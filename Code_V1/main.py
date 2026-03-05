# main.py  (Option A: local OpenCV window on the Raspberry Pi)
from __future__ import annotations

import time
import cv2

from config import W, H
from camera import PiCam
from vision import measure_line
from controller import LineController
from motors import MotorDriver


def mix_speed_steer(speed: float, steer: float) -> tuple[float, float]:
    """
    Convert speed (0..100) + steer (-1..+1) to left/right speeds (-100..+100).
    """
    left = speed * (1.0 - steer)
    right = speed * (1.0 + steer)
    return left, right


def main() -> None:
    cam = PiCam(size=(W, H), fps=30)

    motors = MotorDriver()
    ctrl = LineController()

    cv2.namedWindow("VALOBOT", cv2.WINDOW_NORMAL)

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                motors.stop()
                time.sleep(0.01)
                continue

            meas = measure_line(frame)
            out = ctrl.update(meas.found, meas.e_px)

            left, right = mix_speed_steer(out.speed, out.steer)
            motors.set_speeds(left, right)

            # -------- Debug overlay --------
            fps_n += 1
            if fps_n >= 15:
                dt = time.time() - fps_t0
                fps = fps_n / max(dt, 1e-6)
                fps_t0 = time.time()
                fps_n = 0

            h, w = frame.shape[:2]

            # Image center line
            cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)

            # If vision returned a binary debug image, show it in a small corner (optional)
            bw = meas.debug.get("bw", None)
            if bw is not None:
                bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
                bw_bgr = cv2.resize(bw_bgr, (w // 3, h // 3))
                frame[0:bw_bgr.shape[0], 0:bw_bgr.shape[1]] = bw_bgr

            txt1 = f"state={out.state}  speed={out.speed:.1f}  steer={out.steer:.2f}  fps={fps:.1f}"
            txt2 = f"found={meas.found}  e_px={meas.e_px:.1f}  e_used={out.e_used:.1f}"
            cv2.putText(frame, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("VALOBOT", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                motors.stop()
                break

    finally:
        motors.stop()
        motors.cleanup()
        cam.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
