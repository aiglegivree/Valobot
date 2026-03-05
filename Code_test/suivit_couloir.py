# Corridor following using the SAME camera pattern as your working line-following code
# (no Hough, no heavy GStreamer pipeline differences).
#
# Idea (lightweight):
#   - Take bottom ROI
#   - Compute vertical edge strength per column (Sobel d/dy) -> wall/floor boundary produces strong edges
#   - Find strongest edge on left half and right half
#   - Keep robot centered between them with PID
#
# Replace set_motors() with your real motor commands.

import cv2
import numpy as np
import time

CAM_ID = 0
W, H = 640, 480
ROI_START = int(H * 0.60)  # move up/down to “see” farther (smaller = farther)
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not cap.isOpened():
    raise RuntimeError("Caméra non détectée")

# Control
BASE_SPEED = 140      # 0..255
MAX_CORR   = 110      # steering authority
KP, KI, KD = 0.8, 0.0, 0.20
ERR_ALPHA  = 0.35

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# KEEP YOUR MOTOR COMMANDS HERE
def set_motors(left_pwm: int, right_pwm: int):
    # Replace with your motor driver code
    print(f"motors: L={left_pwm:4d}  R={right_pwm:4d}")

def stop_motors():
    set_motors(0, 0)

def find_two_edges_from_strength(col_strength: np.ndarray):
    """
    Returns (x_left, x_right, quality)
    - quality is a simple confidence metric (sum of peak strengths).
    """
    rw = col_strength.shape[0]
    mid = rw // 2

    left = col_strength[:mid]
    right = col_strength[mid:]

    xL = int(np.argmax(left)) if left.size > 0 else None
    xR = int(np.argmax(right)) + mid if right.size > 0 else None

    q = 0.0
    if xL is not None:
        q += float(left[xL])
    if xR is not None and right.size > 0:
        q += float(right[xR - mid])

    return xL, xR, q

integ = 0.0
prev_err = 0.0
err_f = 0.0
prev_t = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (W, H))

        # 1) ROI (same style as your line-following)
        roi = frame[ROI_START:H, :]
        rh, rw = roi.shape[:2]

        # 2) Preprocess (same spirit: gray + blur)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3) Vertical gradient (edge between wall/floor is a strong intensity change)
        # Use dy to emphasize horizontal edges; then sum per column
        dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_dy = cv2.convertScaleAbs(dy)

        # Reduce noise: focus only on lower part of ROI (closer to robot)
        band_y0 = int(rh * 0.35)
        band = abs_dy[band_y0:rh, :]

        # Column edge strength
        col_strength = band.sum(axis=0).astype(np.float32)

        # Smooth strength to avoid jitter
        col_strength = cv2.GaussianBlur(col_strength.reshape(1, -1), (1, 31), 0).reshape(-1)

        xL, xR, quality = find_two_edges_from_strength(col_strength)

        cx = rw * 0.5
        have = (xL is not None) and (xR is not None) and (abs(xR - xL) > rw * 0.25)

        if have:
            center = 0.5 * (xL + xR)
        else:
            center = cx  # fallback: go straight if detection weak

        # 4) Control
        err = (center - cx) / cx
        err_f = (1 - ERR_ALPHA) * err_f + ERR_ALPHA * err

        now = time.time()
        dt = max(now - prev_t, 1e-3)
        prev_t = now

        integ += err_f * dt
        deriv = (err_f - prev_err) / dt
        prev_err = err_f

        steer = KP * err_f + KI * integ + KD * deriv
        steer = clamp(steer, -1.0, 1.0)

        corr = int(steer * MAX_CORR)
        L = int(clamp(BASE_SPEED - corr, 0, 255))
        R = int(clamp(BASE_SPEED + corr, 0, 255))

        # If not confident, slow down a bit
        if not have:
            L = int(L * 0.7)
            R = int(R * 0.7)

        set_motors(L, R)

        # Debug visualization (press q)
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # draw found edges and center
        if xL is not None:
            cv2.line(dbg, (xL, 0), (xL, rh - 1), (0, 255, 0), 2)
        if xR is not None:
            cv2.line(dbg, (xR, 0), (xR, rh - 1), (0, 255, 0), 2)
        cv2.line(dbg, (int(cx), 0), (int(cx), rh - 1), (255, 0, 0), 2)
        cv2.line(dbg, (int(center), 0), (int(center), rh - 1), (0, 0, 255), 2)

        cv2.putText(dbg, f"have={have} q={quality:.0f} err={err_f:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("corridor ROI (gray)", dbg)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_motors()
    cap.release()
    cv2.destroyAllWindows()