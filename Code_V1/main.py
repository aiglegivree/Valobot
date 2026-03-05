# main.py
from __future__ import annotations
import time
import cv2
import numpy as np

from picamera2 import Picamera2

from config import W, H, FPS, SHOW_MASK_WINDOW
from vision import LineSelector, detect_line
from controller import LineFollower
from motors import MotorDriver

def main():
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"format": "BGR888", "size": (W, H)},
        controls={"FrameRate": int(FPS)}
    ))
    cam.start()

    motors = MotorDriver()
    selector = LineSelector(W)
    follower = LineFollower()
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.namedWindow("VALOBOT", cv2.WINDOW_NORMAL)
    if SHOW_MASK_WINDOW:
        cv2.namedWindow("MASK_ROI", cv2.WINDOW_NORMAL)

    fps_t0 = time.time()
    fps_n = 0
    fps_val = 0.0

    try:
        while True:
            frame = cam.capture_array()
            if frame is None:
                continue

            pick = detect_line(frame, selector)
            out = follower.update(pick)

            # Apply motors
            motors.set_speeds(out.left, out.right)

            # ---- Visualization ----
            vis = frame.copy()

            # ROI rectangle
            roi_start = int(H * 0.30)  # matches ROI_START_FRAC in config (keep in sync)
            cv2.rectangle(vis, (0, roi_start), (W - 1, H - 1), (0, 255, 255), 2)

            if pick is not None:
                gx_look = pick.cx_look
                gy_look = pick.roi_start + pick.y_look
                gx_bottom = pick.cx_bottom
                gy_bottom = pick.roi_start + int(pick.mask.shape[0] * 0.90)

                cv2.circle(vis, (gx_look, gy_look), 7, (0, 0, 255), -1)
                cv2.circle(vis, (gx_bottom, gy_bottom), 7, (255, 0, 0), -1)
                cv2.line(vis, (W // 2, H - 1), (gx_look, gy_look), (255, 0, 0), 2)

            cv2.putText(vis, f"state={out.state} e={out.e:+.3f} steer={out.steer:+.2f}", (10, 30),
                        font, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, f"L={out.left:+.1f} R={out.right:+.1f} turn_sign={out.last_turn_sign}", (10, 60),
                        font, 0.7, (255, 255, 255), 2)

            # FPS
            fps_n += 1
            t = time.time()
            if t - fps_t0 >= 0.5:
                fps_val = fps_n / (t - fps_t0)
                fps_t0 = t
                fps_n = 0
            cv2.putText(vis, f"FPS={fps_val:.1f}", (10, H - 15),
                        font, 0.7, (255, 255, 255), 2)

            cv2.imshow("VALOBOT", vis)

            if SHOW_MASK_WINDOW and pick is not None:
                cv2.imshow("MASK_ROI", pick.mask)
            elif SHOW_MASK_WINDOW:
                # show last computed mask even if lost (optional)
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        motors.stop()
        motors.cleanup()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()