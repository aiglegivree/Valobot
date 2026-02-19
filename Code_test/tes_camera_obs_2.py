#!/usr/bin/env python3
"""
Corridor center detection for low-contrast (white walls + white floor).
Outputs an estimated center offset from image center.

Approach:
- Use only bottom ROI (where wall/floor boundary is more visible)
- CLAHE for local contrast
- Scharr gradient (stronger than Sobel)
- Adaptive threshold to keep weak edges
- Column-wise energy profile -> pick left/right "walls" as peaks
- Compute corridor center and normalized error in [-1, 1]

Dependencies:
  sudo apt-get install -y python3-opencv
Optional for Pi Camera:
  sudo apt-get install -y python3-picamera2
"""

import time
import cv2
import numpy as np


# -----------------------------
# Capture (PiCamera2 preferred, fallback to /dev/video0)
# -----------------------------
class OpenCVCapture:
    def __init__(self, device=0, w=640, h=480, fps=30):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class PiCamera2Capture:
    def __init__(self, w=640, h=480, fps=30):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.2)

    def read(self):
        frame = self.picam2.capture_array()  # RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def release(self):
        self.picam2.stop()


def make_source(w=640, h=480, fps=30):
    try:
        import picamera2  # noqa
        return PiCamera2Capture(w, h, fps)
    except Exception:
        return OpenCVCapture(0, w, h, fps)


# -----------------------------
# Low-contrast corridor center detection
# -----------------------------
def corridor_center_from_gradients(frame_bgr, roi_y0_ratio=0.45):
    h, w = frame_bgr.shape[:2]
    y0 = int(h * roi_y0_ratio)
    roi = frame_bgr[y0:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Local contrast boost (helps on "all white")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # Denoise lightly (keep edges)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Strong gradient operator (Scharr)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    # Normalize to 8-bit for thresholding/visualization
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Adaptive threshold keeps weak edges in varying lighting
    thr = cv2.adaptiveThreshold(
        mag_u8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,   # block size
        -5,   # C (negative keeps more edges)
    )

    # Clean small noise and connect structures
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Column-wise "edge energy"
    col_energy = thr.sum(axis=0).astype(np.float32)  # 0..255*roi_h
    # Smooth energy to avoid spurious peaks
    col_energy = cv2.GaussianBlur(col_energy.reshape(1, -1), (1, 51), 0).ravel()

    # Find peaks on left and right halves
    mid = w // 2
    left_energy = col_energy[:mid]
    right_energy = col_energy[mid:]

    # If scene is too uniform, energy will be near-zero
    if left_energy.max() < 1e-3 and right_energy.max() < 1e-3:
        return None, thr, roi, y0

    x_left = int(np.argmax(left_energy))
    x_right = int(mid + np.argmax(right_energy))

    # Sanity: ensure walls not collapsed
    if x_right - x_left < int(0.15 * w):
        # fallback: use global max as "one wall" and infer center roughly
        x_peak = int(np.argmax(col_energy))
        x_center = x_peak
    else:
        x_center = int(0.5 * (x_left + x_right))

    # Normalized error: + means target is to the right of image center
    err = (x_center - mid) / float(mid)
    err = float(np.clip(err, -1.0, 1.0))

    return (x_left, x_right, x_center, err), thr, roi, y0


def main():
    src = make_source(640, 480, 30)

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                continue

            res, thr, roi, y0 = corridor_center_from_gradients(frame)

            vis = frame.copy()

            # Draw ROI
            cv2.rectangle(vis, (0, y0), (vis.shape[1]-1, vis.shape[0]-1), (255, 255, 255), 1)

            if res is None:
                cv2.putText(vis, "No edges detected (scene too uniform)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                xL, xR, xC, err = res
                # Convert ROI x to full image x (same), y lines across ROI area
                cv2.line(vis, (xL, y0), (xL, vis.shape[0]-1), (0, 255, 0), 2)
                cv2.line(vis, (xR, y0), (xR, vis.shape[0]-1), (0, 255, 0), 2)
                cv2.line(vis, (xC, y0), (xC, vis.shape[0]-1), (0, 0, 255), 3)

                cv2.putText(vis, f"center err={err:+.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Print value for control later
                print(f"{err:+.4f}")

            cv2.imshow("corridor_center_vis", vis)
            cv2.imshow("roi_edges_binary", thr)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        src.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
