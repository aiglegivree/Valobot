import cv2
import numpy as np
import time

# -----------------------------
# SAME camera style as your line-following code
# -----------------------------
CAM_ID = 0
W, H = 640, 480
ROI_START = int(H * 0.55)   # observe farther (increase = lower ROI, decrease = higher ROI)

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not cap.isOpened():
    raise RuntimeError("Caméra non détectée / pipeline non démarrée")

# -----------------------------
# Control params
# -----------------------------
BASE_SPEED = 140         # 0..255
MAX_CORR = 120           # steering authority

KP, KI, KD = 0.9, 0.0, 0.25
ERR_ALPHA = 0.4          # smoothing

# -----------------------------
# Helpers
# -----------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def x_at_y(line, yq):
    x1, y1, x2, y2 = line
    if abs(y2 - y1) < 1e-6:
        return None
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    if not (ymin <= yq <= ymax):
        return None
    t = (yq - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

def pick_boundaries(lines, rw, rh, slope_min=0.3, slope_max=5.0):
    if lines is None:
        return None, None

    yq = rh - 1
    left_votes = []
    right_votes = []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = (x2 - x1), (y2 - y1)
        if abs(dx) < 1e-6:
            continue
        slope = dy / dx

        if abs(slope) < slope_min or abs(slope) > slope_max:
            continue

        xb = x_at_y((x1, y1, x2, y2), yq)
        if xb is None:
            continue

        length = float(np.hypot(dx, dy))
        if slope > 0:
            left_votes.append((xb, length))
        else:
            right_votes.append((xb, length))

    def weighted_pick(votes):
        if not votes:
            return None
        xs = np.array([v[0] for v in votes], dtype=np.float32)
        ws = np.array([v[1] for v in votes], dtype=np.float32)
        order = np.argsort(xs)
        xs, ws = xs[order], ws[order]
        cum = np.cumsum(ws)
        idx = int(np.searchsorted(cum, cum[-1] * 0.5))
        x_pick = float(xs[min(idx, len(xs) - 1)])
        conf = float(cum[-1])
        return x_pick, conf

    return weighted_pick(left_votes), weighted_pick(right_votes)

# -----------------------------
# KEEP YOUR MOTOR COMMANDS HERE
# -----------------------------
def set_motors(left_pwm: int, right_pwm: int):
    # Replace with your real motor driver code
    print(f"motors: L={left_pwm:4d}  R={right_pwm:4d}")

def stop_motors():
    set_motors(0, 0)

# -----------------------------
# Main loop
# -----------------------------
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

        # 1) ROI (same style)
        roi = frame[ROI_START:H, :]
        rh, rw = roi.shape[:2]

        # 2) Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3) Edge detection
        edges = cv2.Canny(gray, 60, 160)

        # 4) Lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=45,
            minLineLength=40,
            maxLineGap=20
        )

        left, right = pick_boundaries(lines, rw, rh)

        cx = rw * 0.5
        if left is not None and right is not None:
            center = 0.5 * (left[0] + right[0])
            have = 2
        elif left is not None:
            # fallback: symmetric guess
            center = cx
            have = 1
        elif right is not None:
            center = cx
            have = 1
        else:
            center = cx
            have = 0

        # 5) Control error
        err = (center - cx) / cx
        err_f = (1 - ERR_ALPHA) * err_f + ERR_ALPHA * err

        now = time.time()
        dt = max(now - prev_t, 1e-3)
        prev_t = now

        # PID
        integ += err_f * dt
        deriv = (err_f - prev_err) / dt
        prev_err = err_f

        steer = KP * err_f + KI * integ + KD * deriv
        steer = clamp(steer, -1.0, 1.0)

        corr = int(steer * MAX_CORR)

        L = int(clamp(BASE_SPEED - corr, 0, 255))
        R = int(clamp(BASE_SPEED + corr, 0, 255))

        if have == 0:
            # slow down if nothing detected
            L = int(L * 0.6)
            R = int(R * 0.6)

        set_motors(L, R)

        # Debug
        dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if left is not None:
            cv2.circle(dbg, (int(left[0]), rh - 1), 6, (0, 255, 0), -1)
        if right is not None:
            cv2.circle(dbg, (int(right[0]), rh - 1), 6, (0, 255, 0), -1)

        cv2.line(dbg, (int(cx), 0), (int(cx), rh - 1), (255, 0, 0), 2)
        cv2.line(dbg, (int(center), 0), (int(center), rh - 1), (0, 0, 255), 2)

        cv2.imshow("corridor edges ROI", dbg)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_motors()
    cap.release()
    cv2.destroyAllWindows()