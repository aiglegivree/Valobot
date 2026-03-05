import time
import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# Camera / ROI
# =========================
W, H = 640, 480
ROI_Y0 = int(H * 0.45)        # higher -> look further; lower -> more stable
LOOKAHEAD_Y_NORM = 0.35       # smaller => farther ahead (0.25..0.55)

# =========================
# Edge / Hough parameters
# =========================
CANNY1 = 50
CANNY2 = 130

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESH = 45
MIN_LINE_LEN = 50
MAX_LINE_GAP = 35

# Reject near-horizontal clutter (keep, but not too strict)
MIN_ABS_SLOPE = 0.25

# =========================
# Steering (normalized -1..1)
# =========================
Kp = 0.005
Kd = 0.001
Ki = 0.0

# =========================
# Turn state machine
# =========================
MODE_FOLLOW = 0
MODE_TURN_L = 1
MODE_TURN_R = 2

TURN_STEER = 0.85
MIN_TURN_TIME = 0.30
MAX_TURN_TIME = 2.0
REACQUIRE_FRAMES = 3
CORNER_CONFIRM_FRAMES = 2

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def line_x_at_y(x1, y1, x2, y2, yq):
    dy = (y2 - y1)
    dx = (x2 - x1)
    if abs(dy) < 1e-6:
        return None
    t = (yq - y1) / dy
    return x1 + t * dx

def robust_fit_x_at_y(lines, yq):
    xs = []
    for (x1, y1, x2, y2) in lines:
        x = line_x_at_y(x1, y1, x2, y2, yq)
        if x is None:
            continue
        if np.isfinite(x):
            xs.append(x)
    if len(xs) == 0:
        return None
    return float(np.median(xs))

# =========================
# Picamera2 setup
# =========================
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"size": (W, H), "format": "RGB888"})
picam2.configure(cfg)
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

while True:
    # Capture
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # ROI
    roi = frame[ROI_Y0:H, :].copy()
    h_roi, w_roi = roi.shape[:2]
    yq = int(h_roi * LOOKAHEAD_Y_NORM)

    # Preprocess
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edges
    edges = cv2.Canny(gray, CANNY1, CANNY2)

    # Hough segments
    segs = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESH,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP
    )

    left_lines = []
    right_lines = []

    if segs is not None:
        for s in segs[:, 0, :]:
            x1, y1, x2, y2 = map(int, s)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 2 and abs(dy) < 2:
                continue

            slope = dy / (dx + 1e-6)
            if abs(slope) < MIN_ABS_SLOPE:
                continue

            # classify by slope sign
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    # Robust x positions at look-ahead y
    x_left = robust_fit_x_at_y(left_lines, yq)
    x_right = robust_fit_x_at_y(right_lines, yq)

    # "Good" corridor: both sides exist and are not crossing (loose condition)
    corridor_good = (x_left is not None) and (x_right is not None) and ((x_right - x_left) > 60)

    if corridor_good:
        x_mid = 0.5 * (x_left + x_right)
        err = x_mid - (w_roi / 2.0)
    else:
        # If only one wall is visible, we still can produce a usable error by assuming a nominal corridor half-width in pixels.
        # This helps avoid losing control when one wall detection drops.
        NOMINAL_HALF_WIDTH_PX = 120  # tune once you see typical x_right-x_left in debug
        if (x_left is not None) and (x_right is None):
            x_mid = x_left + NOMINAL_HALF_WIDTH_PX
            err = x_mid - (w_roi / 2.0)
        elif (x_right is not None) and (x_left is None):
            x_mid = x_right - NOMINAL_HALF_WIDTH_PX
            err = x_mid - (w_roi / 2.0)
        else:
            err = prev_err

    # PID
    now = time.time()
    dt = max(1e-3, now - prev_t)
    derr = (err - prev_err) / dt
    err_i += err * dt

    # Turn inference when corridor is lost
    missing_left = (x_left is None)
    missing_right = (x_right is None)

    inferred_turn = None
    if not corridor_good:
        if missing_left and (not missing_right):
            inferred_turn = MODE_TURN_L
        elif missing_right and (not missing_left):
            inferred_turn = MODE_TURN_R
        else:
            inferred_turn = MODE_TURN_R if prev_err >= 0 else MODE_TURN_L

    # State machine
    if mode == MODE_FOLLOW:
        corner_condition = (not corridor_good) and (inferred_turn is not None)
        corner_cnt = corner_cnt + 1 if corner_condition else 0

        if corner_cnt >= CORNER_CONFIRM_FRAMES:
            mode = inferred_turn
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

    # Steer output
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
    # Debug
    # -------------------------
    debug = roi.copy()

    for (x1, y1, x2, y2) in left_lines:
        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x1, y1, x2, y2) in right_lines:
        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.line(debug, (0, yq), (w_roi - 1, yq), (255, 255, 0), 2)

    if x_left is not None:
        cv2.circle(debug, (int(x_left), yq), 6, (255, 0, 0), -1)
    if x_right is not None:
        cv2.circle(debug, (int(x_right), yq), 6, (255, 0, 0), -1)

    if (x_left is not None) or (x_right is not None):
        # draw estimated mid if computed
        x_mid_draw = int((w_roi / 2.0) + err)
        cv2.circle(debug, (x_mid_draw, yq), 7, (0, 0, 255), -1)

    cx = w_roi // 2
    cv2.line(debug, (cx, 0), (cx, h_roi - 1), (255, 0, 255), 2)

    mode_str = {MODE_FOLLOW: "FOLLOW", MODE_TURN_L: "TURN_L", MODE_TURN_R: "TURN_R"}[mode]
    cv2.putText(debug, f"mode={mode_str} good={corridor_good}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    wpx = None
    if (x_left is not None) and (x_right is not None):
        wpx = int(x_right - x_left)
    cv2.putText(debug, f"xL={None if x_left is None else int(x_left)} xR={None if x_right is None else int(x_right)} width_px={wpx}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(debug, f"err_px={err:.1f} steer={steer:.3f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("roi_debug", debug)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()