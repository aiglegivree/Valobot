"""
Corridor following using the wall/floor edge (camera).
- Detect left/right floor boundaries in a bottom ROI
- Compute corridor center -> steering error
- PID -> differential drive speeds

Keep your existing motor functions:
    set_motors(left_pwm: int, right_pwm: int)
    stop_motors()
Replace the stubs at the end with your own implementation.
"""

import cv2
import numpy as np
import time

# -----------------------------
# Camera / processing settings
# -----------------------------
CAM_ID = 0
W, H = 640, 480
ROI_Y0 = int(H * 0.55)          # bottom part of image
BLUR_K = (5, 5)

# Edge / line extraction
CANNY_LOW, CANNY_HIGH = 60, 160
HOUGH_THRESH = 45
MIN_LINE_LEN = 40
MAX_LINE_GAP = 20

# Geometry filters (tune for your camera)
# We want diagonals corresponding to wall/floor boundary on each side
SLOPE_MIN = 0.3                 # ignore almost-horizontal lines
SLOPE_MAX = 5.0                 # ignore almost-vertical lines

# Control
BASE_SPEED = 140                # 0..255
MAX_STEER = 120                 # max +/- correction added/subtracted to base speed

KP, KI, KD = 0.9, 0.0, 0.25     # start here, tune
DT_MIN = 1e-3

# Optional smoothing
ERR_ALPHA = 0.4                 # 0..1 (higher = more smoothing)

# -----------------------------
# Helpers
# -----------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def weighted_x_at_y(line, y_query):
    """
    For a line segment (x1,y1,x2,y2), compute x at a given y using interpolation.
    Returns None if segment is horizontal or y is out of span.
    """
    x1, y1, x2, y2 = line
    if abs(y2 - y1) < 1e-6:
        return None
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
    if not (y_min <= y_query <= y_max):
        return None
    t = (y_query - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

def pick_boundaries(lines, w, h):
    """
    Split lines into left/right candidates by slope sign and return one representative
    boundary for each side: (x_at_bottom, confidence).
    """
    if lines is None:
        return None, None

    yq = h - 1  # bottom of ROI coordinates
    left_votes = []
    right_votes = []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6:
            continue
        slope = dy / dx

        # filter slopes
        if abs(slope) < SLOPE_MIN or abs(slope) > SLOPE_MAX:
            continue

        xb = weighted_x_at_y((x1, y1, x2, y2), yq)
        if xb is None:
            continue

        # confidence: longer segments get more weight
        length = float(np.hypot(dx, dy))

        # In image coords, left boundary usually has positive slope,
        # right boundary usually has negative slope (depends on camera).
        if slope > 0:
            left_votes.append((xb, length))
        else:
            right_votes.append((xb, length))

    def robust_vote(votes):
        if not votes:
            return None
        xs = np.array([v[0] for v in votes], dtype=np.float32)
        ws = np.array([v[1] for v in votes], dtype=np.float32)
        # weighted median-ish: sort by x and accumulate weights
        order = np.argsort(xs)
        xs, ws = xs[order], ws[order]
        cum = np.cumsum(ws)
        mid = cum[-1] * 0.5
        idx = int(np.searchsorted(cum, mid))
        x_pick = float(xs[min(idx, len(xs) - 1)])
        conf = float(cum[-1])
        return x_pick, conf

    left = robust_vote(left_votes)
    right = robust_vote(right_votes)
    return left, right

# -----------------------------
# Motor interface (KEEP YOURS)
# -----------------------------
def set_motors(left_pwm: int, right_pwm: int):
    """
    KEEP your motor commands here.
    Example:
        left_pwm/right_pwm in [-255..255] with sign for direction
        or [0..255] forward only, etc.
    """
    # TODO: replace with your implementation
    # print for debugging:
    print(f"motors: L={left_pwm:4d}  R={right_pwm:4d}")

def stop_motors():
    # TODO: replace with your implementation
    set_motors(0, 0)

# -----------------------------
# Main
# -----------------------------
def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    if not cap.isOpened():
        raise RuntimeError("Camera not detected")

    integ = 0.0
    prev_err = 0.0
    prev_time = time.time()
    err_filt = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (W, H))
            roi = frame[ROI_Y0:H, :]
            rh, rw = roi.shape[:2]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, BLUR_K, 0)

            edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=HOUGH_THRESH,
                minLineLength=MIN_LINE_LEN,
                maxLineGap=MAX_LINE_GAP,
            )

            left, right = pick_boundaries(lines, rw, rh)

            # Fallback logic:
            # - If both boundaries found -> center is average
            # - If only one found -> assume corridor width by mirroring from center
            #   (works ok if you stay roughly centered)
            cx = rw * 0.5
            have = 0
            if left is not None and right is not None:
                xL, cL = left
                xR, cR = right
                center = 0.5 * (xL + xR)
                have = 2
            elif left is not None:
                xL, cL = left
                # assume right boundary symmetric around image center
                center = 0.5 * (xL + (2 * cx - xL))
                have = 1
            elif right is not None:
                xR, cR = right
                center = 0.5 * ((2 * cx - xR) + xR)
                have = 1
            else:
                center = cx
                have = 0

            # Error normalized: negative => steer left, positive => steer right
            err = (center - cx) / cx  # roughly [-1..1]

            # Smooth error a bit to reduce jitter
            err_filt = (1 - ERR_ALPHA) * err_filt + ERR_ALPHA * err

            now = time.time()
            dt = max(now - prev_time, DT_MIN)
            prev_time = now

            # PID
            integ += err_filt * dt
            deriv = (err_filt - prev_err) / dt
            prev_err = err_filt

            steer = KP * err_filt + KI * integ + KD * deriv
            steer = clamp(steer, -1.0, 1.0)

            correction = int(steer * MAX_STEER)

            left_pwm = clamp(BASE_SPEED - correction, -255, 255)
            right_pwm = clamp(BASE_SPEED + correction, -255, 255)

            # If no detection, slow down (optional safety)
            if have == 0:
                left_pwm = int(left_pwm * 0.6)
                right_pwm = int(right_pwm * 0.6)

            set_motors(int(left_pwm), int(right_pwm))

            # Debug view (press q to quit)
            dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Draw selected boundaries / center
            if left is not None:
                cv2.circle(dbg, (int(left[0]), rh - 1), 6, (0, 255, 0), -1)
            if right is not None:
                cv2.circle(dbg, (int(right[0]), rh - 1), 6, (0, 255, 0), -1)
            cv2.line(dbg, (int(cx), 0), (int(cx), rh - 1), (255, 0, 0), 2)
            cv2.line(dbg, (int(center), 0), (int(center), rh - 1), (0, 0, 255), 2)

            cv2.imshow("ROI edges + guide", dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_motors()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()