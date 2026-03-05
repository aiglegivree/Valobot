# camera.py
from __future__ import annotations
import time
import numpy as np

from picamera2 import Picamera2

class PiCam:
    def __init__(self, size=(640, 480), fps=30):
        self.size = tuple(size)
        self.fps = int(fps)

        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": self.size, "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.3)

    def read(self) -> tuple[bool, np.ndarray | None]:
        # returns (ok, frame_bgr)
        frame_rgb = self.picam2.capture_array()
        if frame_rgb is None:
            return False, None
        # Convert RGB->BGR without cv2 dependency here:
        frame_bgr = frame_rgb[:, :, ::-1].copy()
        return True, frame_bgr

    def close(self) -> None:
        try:
            self.picam2.stop()
        except Exception:
            pass
