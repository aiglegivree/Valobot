from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from leds import LED


W = 640
H = 480
CAM_ID = 0
USE_PICAMERA2 = True
ROI_START_FRAC = 0.50

SAVE_ROOT = Path(__file__).with_name("t_training_frames")


class Camera:
    def __init__(self):
        if USE_PICAMERA2:
            from picamera2 import Picamera2

            self.cam = Picamera2()
            self.cam.configure(
                self.cam.create_preview_configuration(
                    main={"size": (W, H), "format": "RGB888"}
                )
            )
            self.cam.start()
            self.type = "picamera2"
        else:
            cap = cv2.VideoCapture(CAM_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            if not cap.isOpened():
                raise RuntimeError("Camera not found")
            self.cam = cap
            self.type = "opencv"

    def read(self):
        if self.type == "picamera2":
            frame = self.cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame

        ret, frame = self.cam.read()
        return ret, frame

    def release(self):
        if self.type == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()


def make_session_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = SAVE_ROOT / f"session_{stamp}"
    (session_dir / "full").mkdir(parents=True, exist_ok=True)
    (session_dir / "roi").mkdir(parents=True, exist_ok=True)
    return session_dir


def main():
    leds = LED()
    leds.turn_on_all()
    leds_on = True
    cam = Camera()
    session_dir = make_session_dir()
    capture_count = 0
    last_saved_name = ""

    cv2.namedWindow("T Capture", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            roi_start = int(frame.shape[0] * ROI_START_FRAC)
            roi = frame[roi_start:, :]

            preview = frame.copy()
            cv2.rectangle(
                preview,
                (0, roi_start),
                (preview.shape[1] - 1, preview.shape[0] - 1),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                preview,
                "t save frame | l LEDs | q quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"Saved: {capture_count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"LEDs: {'ON' if leds_on else 'OFF'}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            if last_saved_name:
                cv2.putText(
                    preview,
                    f"Last: {last_saved_name}",
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("T Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("l"):
                leds_on = not leds_on
                if leds_on:
                    leds.turn_on_all()
                else:
                    leds.turn_off_all()
            if key == ord("t"):
                capture_count += 1
                base_name = f"t_{capture_count:03d}.png"
                full_path = session_dir / "full" / base_name
                roi_path = session_dir / "roi" / base_name
                cv2.imwrite(str(full_path), frame)
                cv2.imwrite(str(roi_path), roi)
                last_saved_name = base_name

    finally:
        try:
            leds.turn_off_all()
        except Exception:
            pass

        try:
            cam.release()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
