import cv2
import numpy as np
import time

# =========================
# Camera / Image parameters
# =========================
CAM_ID = 0
W, H = 640, 480

# ROI: bottom of image
ROI_Y0 = int(H * 0.55)   # raise (e.g. 0.45) to "look further", lower (e.g. 0.65) to be more stable
N_ROWS = 25              # number of scanlines sampled inside ROI
MIN_CORRIDOR_WIDTH = 120 # px, rejects spurious detections

# =========================
# Segmentation parameters
# =========================
WALL_THRESH = 180        # grayscale threshold: bright = wall, dark = floor (tune!)
OPEN_THRESH = 0.18       # opening score threshold for turn detection (tune 0.12..0.30)

# Morphology (noise cleanup)
OPEN_K = 5
CLOSE_K = 7

# =========================
# Controller (steer output)
# =========================
# Steer is normalized to [-1, 1]
Kp = 0.004
Kd = 0.001
Ki = 0.0

# =========================
# Turn state machine
# =========================
MODE_FOLLOW = 0
MODE_TURN_L = 1
MODE_TURN_R = 2

MIN_TURN_TIME = 0.35     # seconds before allowing corridor reacquisition
MAX_TURN_TIME = 2.0      # safety timeout
REACQUIRE_FRAMES = 3     # number of consecutive "good corridor" frames to exit turn mode
TURN_STEER = 0.85        # fixed steer magnitude during turns (tune 0.6..0.9)

# Optional: slow down detection flicker by requiring corner condition for a few frames
CORNER_CONFIRM_FRAMES = 2

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def make_kernel(k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return np.ones((k, k), np.uint8)

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not cap.isOpened():
    raise RuntimeError("Camera not detected")

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
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (W, H))

    # -------------------------
    # ROI extraction
    # -------------------------
    roi = frame[ROI_Y0:H, :].copy()

    # -------------------------
    # Preprocess + segmentation
    # -------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # walls: white (255), floor: black (0)
    _, walls = cv2.threshold(gray, WALL_THRESH, 255, cv2.THRESH_BINARY)

    # clean mask
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k_open)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k_close)

    h_roi, w_roi = walls.shape

    # -------------------------
    # Corridor center estimation
    # -------------------------
    rows = np.linspace(int(h_roi * 0.25), h_roi - 1, N_ROWS).astype(int)

    midpoints = []
    widths = []
    debug = roi.copy()

    for y in rows:
        row = walls[y, :]

        # left edge: first wall pixel from left
        left_idx = np.argmax(row > 0)
        if row[left_idx] == 0:
            continue

        # right edge: first wall pixel from right
        right_rev = np.argmax(row[::-1] > 0)
        right_idx = w_roi - 1 - right_rev
        if row[right_idx] == 0:
            continue

        width = right_idx - left_idx
        if width < MIN_CORRIDOR_WIDTH:
            continue

        mid = 0.5 * (left_idx + right_idx)
        midpoints.append(mid)
        widths.append(width)

        # debug points
        cv2.circle(debug, (left_idx, y), 3, (0, 255, 0), -1)
        cv2.circle(debug, (right_idx, y), 3, (0, 255, 0), -1)
        cv2.circle(debug, (int(mid), y), 3, (0, 0, 255), -1)

    corridor_good = (len(midpoints) >= max(5, N_ROWS // 3))

    if corridor_good:
        mid_est = float(np.median(midpoints))
        err = (mid_est - (w_roi / 2.0))  # + => corridor center is to the right
    else:
        err = prev_err  # hold last value

    # -------------------------
    # Opening detection for 90° turns
    # -------------------------
    # floor mask (white=floor)
    floor = cv2.bitwise_not(walls)

    # upper band (look-ahead region inside ROI)
    y1 = int(h_roi * 0.15)
    y2 = int(h_roi * 0.45)
    band = floor[y1:y2, :]

    left_floor = int(np.sum(band[:, :w_roi // 2] > 0))
    right_floor = int(np.sum(band[:, w_roi // 2:] > 0))
    den = max(1, left_floor + right_floor)
    open_score = (right_floor - left_floor) / den  # + => opening right

    # -------------------------
    # PID terms (used in FOLLOW mode)
    # -------------------------
    now = time.time()
    dt = max(1e-3, now - prev_t)
    derr = (err - prev_err) / dt
    err_i += err * dt

    # -------------------------
    # Turn state machine
    # -------------------------
    if mode == MODE_FOLLOW:
        corner_condition = (abs(open_score) > OPEN_THRESH) and (not corridor_good)
        if corner_condition:
            corner_cnt += 1
        else:
            corner_cnt = 0

        if corner_cnt >= CORNER_CONFIRM_FRAMES:
            mode = MODE_TURN_R if open_score > 0 else MODE_TURN_L
            turn_start_t = now
            reacquire_cnt = 0
            corner_cnt = 0

    else:
        # TURN mode
        if now - turn_start_t > MIN_TURN_TIME:
            if corridor_good:
                reacquire_cnt += 1
            else:
                reacquire_cnt = 0

            if reacquire_cnt >= REACQUIRE_FRAMES:
                mode = MODE_FOLLOW

        # safety timeout
        if now - turn_start_t > MAX_TURN_TIME:
            mode = MODE_FOLLOW

    # -------------------------
    # Steer output
    # -------------------------
    if mode == MODE_FOLLOW:
        steer = (Kp * err) + (Kd * derr) + (Ki * err_i)
        steer = clamp(steer, -1.0, 1.0)
    elif mode == MODE_TURN_L:
        steer = -TURN_STEER
    else:  # MODE_TURN_R
        steer = +TURN_STEER

    # Update for next loop
    prev_err = err
    prev_t = now

    # -------------------------
    # (Motor command placeholder)
    # -------------------------
    # Example mapping (commented):
    # base_speed = 0.35
    # gain = 0.30
    # left_cmd = clamp(base_speed - gain * steer, 0.0, 1.0)
    # right_cmd = clamp(base_speed + gain * steer, 0.0, 1.0)
    # send_to_motors(left_cmd, right_cmd)

    # -------------------------
    # Debug overlays
    # -------------------------
    cx = w_roi // 2
    cv2.line(debug, (cx, 0), (cx, h_roi - 1), (255, 0, 0), 2)  # image center

    if corridor_good:
        cv2.line(debug, (int(mid_est), 0), (int(mid_est), h_roi - 1), (0, 0, 255), 2)

    # show the band used for opening detection
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
    cv2.imshow("floor_mask", cv2.bitwise_not(walls))

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()