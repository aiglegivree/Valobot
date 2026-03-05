import time
import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# Image / ROI parameters
# =========================
W, H = 640, 480

ROI_Y0 = int(H * 0.55)     # raise -> look further, lower -> more stable
N_ROWS = 25
MIN_CORRIDOR_WIDTH = 120   # px

# =========================
# Segmentation parameters
# =========================
WALL_THRESH = 180          # bright=wall, dark=floor (tune!)
OPEN_THRESH = 0.18         # turn opening score threshold (tune 0.12..0.30)

OPEN_K = 5
CLOSE_K = 7

# =========================
# Steering (normalized -1..1)
# =========================
Kp = 0.004
Kd = 0.001
Ki = 0.0

# =========================
# Turn state machine
# =========================
MODE_FOLLOW = 0
MODE_TURN_L = 1
MODE_TURN_R = 2

MIN_TURN_TIME = 0.35       # seconds
MAX_TURN_TIME = 2.0        # seconds
REACQUIRE_FRAMES = 3
TURN_STEER = 0.85          # fixed steer during turn (tune 0.6..0.9)
CORNER_CONFIRM_FRAMES = 2  # require corner condition for N frames to trigger

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def make_kernel(k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return np.ones((k, k), np.uint8)

# =========================
# Picamera2 setup
# =========================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (W, H), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(0.4)

# =========================
# Runtime state
# =========================
prev_err = 0.0
err_i = 0.0
prev_t = time.time()

mode = MODE_FOLLOW
turn_start_t = 0.0
reacquire_cnt = 0
corner_cnt = 0

k_open = make_kernel(OPEN_K)
k_close = make_kernel(CLOSE_K)

while True:
    # Picamera2 gives RGB; convert to BGR for OpenCV display
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # -------------------------
    # ROI extraction
    # -------------------------
    roi = frame[ROI_Y0:H, :].copy()

    # -------------------------
    # Preprocess + segmentation
    # -------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # walls: white, floor: black
    _, walls = cv2.threshold(gray, WALL_THRESH, 255, cv2.THRESH_BINARY)
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k_open)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k_close)

    h_roi, w_roi = walls.shape
    debug = roi.copy()

    # -------------------------
    # Corridor center estimation
    # -------------------------
    rows = np.linspace(int(h_roi * 0.25), h_roi - 1, N_ROWS).astype(int)
    midpoints = []

    for y in rows:
        row = walls[y, :]

        left_idx = np.argmax(row > 0)
        if row[left_idx] == 0:
            continue

        right_rev = np.argmax(row[::-1] > 0)
        right_idx = w_roi - 1 - right_rev
        if row[right_idx] == 0:
            continue

        width = right_idx - left_idx
        if width < MIN_CORRIDOR_WIDTH:
            continue

        mid = 0.5 * (left_idx + right_idx)
        midpoints.append(mid)

        cv2.circle(debug, (left_idx, y), 3, (0, 255, 0), -1)
        cv2.circle(debug, (right_idx, y), 3, (0, 255, 0), -1)
        cv2.circle(debug, (int(mid), y), 3, (0, 0, 255), -1)

    corridor_good = (len(midpoints) >= max(5, N_ROWS // 3))
    if corridor_good:
        mid_est = float(np.median(midpoints))
        err = (mid_est - (w_roi / 2.0))  # + => corridor center to the right
    else:
        err = prev_err

    # -------------------------
    # Opening detection (90° turns)
    # -------------------------
    floor = cv2.bitwise_not(walls)

    y1 = int(h_roi * 0.15)
    y2 = int(h_roi * 0.45)
    band = floor[y1:y2, :]

    left_floor = int(np.sum(band[:, :w_roi // 2] > 0))
    right_floor = int(np.sum(band[:, w_roi // 2:] > 0))
    den = max(1, left_floor + right_floor)
    open_score = (right_floor - left_floor) / den  # + => opening right

    # -------------------------
    # PID (FOLLOW mode)
    # -------------------------
    now = time.time()
    dt = max(1e-3, now - prev_t)
    derr = (err - prev_err) / dt
    err_i += err * dt

    # -------------------------
    # State machine
    # -------------------------
    if mode == MODE_FOLLOW:
        corner_condition = (abs(open_score) > OPEN_THRESH) and (not corridor_good)
        corner_cnt = corner_cnt + 1 if corner_condition else 0

        if corner_cnt >= CORNER_CONFIRM_FRAMES:
            mode = MODE_TURN_R if open_score > 0 else MODE_TURN_L
            turn_start_t = now
            reacquire_cnt = 0
            corner_cnt = 0
    else:
        if now - turn_start_t > MIN_TURN_TIME:
            if corridor_good:
                reacquire_cnt += 1
            else:
                reacquire_cnt = 0

            if reacquire_cnt >= REACQUIRE_FRAMES:
                mode = MODE_FOLLOW

        if now - turn_start_t > MAX_TURN_TIME:
            mode = MODE_FOLLOW

    # -------------------------
    # Steering output
    # -------------------------
    if mode == MODE_FOLLOW:
        steer = clamp((Kp * err) + (Kd * derr) + (Ki * err_i), -1.0, 1.0)
    elif mode == MODE_TURN_L:
        steer = -TURN_STEER
    else:
        steer = +TURN_STEER

    prev_err = err
    prev_t = now

    # -------------------------
    # Motor command placeholder (commented)
    # -------------------------
    # base_speed = 0.35
    # gain = 0.30
    # left_cmd  = clamp(base_speed - gain * steer, 0.0, 1.0)
    # right_cmd = clamp(base_speed + gain * steer, 0.0, 1.0)
    # send_to_motors(left_cmd, right_cmd)

    # -------------------------
    # Debug overlays
    # -------------------------
    cx = w_roi // 2
    cv2.line(debug, (cx, 0), (cx, h_roi - 1), (255, 0, 0), 2)

    if corridor_good:
        cv2.line(debug, (int(mid_est), 0), (int(mid_est), h_roi - 1), (0, 0, 255), 2)

    cv2.rectangle(debug, (0, y1), (w_roi - 1, y2), (255, 255, 0), 2)

    mode_str = {MODE_FOLLOW: "FOLLOW", MODE_TURN_L: "TURN_L", MODE_TURN_R: "TURN_R"}[mode]
    cv2.putText(debug, f"mode={mode_str}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(debug, f"err_px={err:.1f} steer={steer:.3f} mid_n={len(midpoints)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(debug, f"open_score={open_score:.2f} (L={left_floor} R={right_floor})",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("roi_debug", debug)
    cv2.imshow("walls_mask", walls)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()