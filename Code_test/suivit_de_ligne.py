import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from collections import deque

# ----------------------------
# CONFIG (VISION)
# ----------------------------
W, H = 640, 480
ROI_START = int(H * 0.60)
LOOKAHEAD_Y_FRAC = 0.35
BAND_THICKNESS = 10
MIN_AREA = 600
INVERT_AFTER_OTSU = True
SHOW_DEBUG = True

# If your display colors are swapped (red looks blue), fix only for visualization:
FIX_DISPLAY_COLOR_SWAP = True

# ----------------------------
# CONFIG (MOTORS - L298N, BCM pins)  (your mapping)
# ----------------------------
ENA, IN1, IN2 = 13, 27, 22
ENB, IN3, IN4 = 18, 23, 24

PWM_FREQ = 1000

# Slow speeds
BASE_SPEED = 20
MAX_SPEED  = 30
MIN_SPEED  = 18

# ----------------------------
# STEERING (less anticipation)
# ----------------------------
DEADBAND = 0.05
W_LOOK   = 0.20   # small weight -> less anticipation
W_BOTTOM = 0.80   # larger weight -> more "near" correction
Kp = 10.0

# Steering smoothing
STEER_LPF_ALPHA = 0.35  # 0..1

# ----------------------------
# "MEMORY" / RECOVERY FOR RIGHT ANGLES
# When the line disappears, keep turning in the last known turn direction
# until the line is seen again.
# ----------------------------
HISTORY_LEN = 25                # number of last detections stored (~0.4s at 60fps)
LINE_LOST_TIMEOUT_S = 0.12      # after this, enter "search" mode (turn to reacquire)
SEARCH_SPEED = 18               # slow forward during search
SEARCH_STEER_START = 10         # initial steering magnitude during search
SEARCH_STEER_MAX = 28           # max steering magnitude during search
SEARCH_RAMP_TIME = 0.6          # seconds to ramp steering up to max

# If one motor direction is inverted physically:
INVERT_LEFT_MOTOR  = False
INVERT_RIGHT_MOTOR = False


# ----------------------------
# Helper: pick correct line component
# ----------------------------
class LineSelector:
    def __init__(self, w):
        self.w = w
        self.cx_ref = w // 2

    def pick(self, mask):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return None

        h, w = mask.shape
        y_la = int(h * LOOKAHEAD_Y_FRAC)
        y0 = max(0, y_la - BAND_THICKNESS)
        y1 = min(h, y_la + BAND_THICKNESS)

        best = None  # (score, comp_id, cx_bottom, cx_look, area)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue

            comp = (labels == i).astype(np.uint8)

            # bottom slice
            yb0 = int(h * 0.80)
            bottom = comp[yb0:h, :]
            if bottom.sum() == 0:
                continue
            _, xs_b = np.where(bottom > 0)
            cx_bottom = int(xs_b.mean())

            # lookahead band
            band = comp[y0:y1, :]
            if band.sum() == 0:
                cx_look = cx_bottom
                miss_penalty = 1.0
            else:
                _, xs_l = np.where(band > 0)
                cx_look = int(xs_l.mean())
                miss_penalty = 0.0

            score = (
                1.5 * abs(cx_look - self.cx_ref) +
                0.7 * abs(cx_bottom - self.cx_ref) +
                500.0 * miss_penalty -
                0.002 * area
            )

            if (best is None) or (score < best[0]):
                best = (score, i, cx_bottom, cx_look, area)

        if best is None:
            return None

        _, comp_id, cx_bottom, cx_look, area = best

        # continuity
        alpha = 0.35
        self.cx_ref = int((1 - alpha) * self.cx_ref + alpha * cx_look)

        return {
            "comp_id": comp_id,
            "cx_bottom": cx_bottom,
            "cx_look": cx_look,
            "area": area,
            "y_look": int(mask.shape[0] * LOOKAHEAD_Y_FRAC),
        }


# ----------------------------
# MOTOR CONTROL (L298N)
# ----------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def setup_motors():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    for p in [ENA, IN1, IN2, ENB, IN3, IN4]:
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, GPIO.LOW)

    pwmA = GPIO.PWM(ENA, PWM_FREQ)
    pwmB = GPIO.PWM(ENB, PWM_FREQ)
    pwmA.start(0)
    pwmB.start(0)
    return pwmA, pwmB

def motor_set(pwmA, pwmB, left_speed, right_speed):
    if INVERT_LEFT_MOTOR:
        left_speed = -left_speed
    if INVERT_RIGHT_MOTOR:
        right_speed = -right_speed

    ls = int(clamp(left_speed, -100, 100))
    if ls >= 0:
        GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    else:
        GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    pwmA.ChangeDutyCycle(abs(ls))

    rs = int(clamp(right_speed, -100, 100))
    if rs >= 0:
        GPIO.output(IN3, 1); GPIO.output(IN4, 0)
    else:
        GPIO.output(IN3, 0); GPIO.output(IN4, 1)
    pwmB.ChangeDutyCycle(abs(rs))

def motor_stop(pwmA, pwmB):
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    GPIO.output(IN1, 0); GPIO.output(IN2, 0)
    GPIO.output(IN3, 0); GPIO.output(IN4, 0)


# ----------------------------
# MAIN
# ----------------------------
def main():
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"format": "BGR888", "size": (W, H)},
        controls={"FrameRate": 60}
    ))
    cam.start()

    pwmA, pwmB = setup_motors()

    selector = LineSelector(W)
    kernel = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fps_t0 = time.time()
    fps_n = 0
    fps_val = 0.0

    # ---- memory buffers ----
    hist = deque(maxlen=HISTORY_LEN)   # store (t, e_look, e_bot, e_blend, steer)
    steer_prev = 0.0
    last_seen_time = time.time()
    last_turn_sign = 0  # -1 = turn left, +1 = turn right (based on last steer)

    try:
        while True:
            frame = cam.capture_array()
            if frame is None:
                break

            # Visualization source (only for display)
            vis_src = frame
            if FIX_DISPLAY_COLOR_SWAP:
                vis_src = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # ROI
            roi = frame[ROI_START:H, :]

            # Preprocess + Otsu
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if INVERT_AFTER_OTSU:
                mask = 255 - mask

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            pick = selector.pick(mask)

            vis = vis_src.copy()
            cv2.rectangle(vis, (0, ROI_START), (W - 1, H - 1), (0, 255, 255), 2)

            y_look_roi = int(mask.shape[0] * LOOKAHEAD_Y_FRAC)
            y0 = ROI_START + max(0, y_look_roi - BAND_THICKNESS)
            y1 = ROI_START + min(mask.shape[0], y_look_roi + BAND_THICKNESS)
            cv2.rectangle(vis, (0, y0), (W - 1, y1), (255, 255, 0), 1)

            now = time.time()

            if pick is not None:
                last_seen_time = now

                cx_look = pick["cx_look"]
                cx_bottom = pick["cx_bottom"]
                area = pick["area"]

                gx_look = cx_look
                gy_look = ROI_START + pick["y_look"]
                gx_bottom = cx_bottom
                gy_bottom = ROI_START + int(mask.shape[0] * 0.90)

                # errors in [-1, 1]
                e_look = (cx_look - (W / 2)) / (W / 2)
                e_bot  = (cx_bottom - (W / 2)) / (W / 2)

                # blended error (less anticipation)
                e = W_LOOK * e_look + W_BOTTOM * e_bot

                # P control + deadband
                if abs(e) < DEADBAND:
                    steer = 0.0
                else:
                    steer = Kp * e

                # LPF to smooth
                steer = (1.0 - STEER_LPF_ALPHA) * steer_prev + STEER_LPF_ALPHA * steer
                steer_prev = steer

                # update turn sign memory
                if abs(steer) > 1e-6:
                    last_turn_sign = 1 if steer > 0 else -1

                # store memory
                hist.append((now, e_look, e_bot, e, steer))

                left = BASE_SPEED - steer
                right = BASE_SPEED + steer

                left = clamp(left, MIN_SPEED, MAX_SPEED)
                right = clamp(right, MIN_SPEED, MAX_SPEED)

                motor_set(pwmA, pwmB, left, right)

                # draw
                cv2.circle(vis, (gx_look, gy_look), 7, (0, 0, 255), -1)
                cv2.circle(vis, (gx_bottom, gy_bottom), 7, (255, 0, 0), -1)
                cv2.line(vis, (W // 2, H - 1), (gx_look, gy_look), (255, 0, 0), 2)

                cv2.putText(vis, f"e={e:+.3f} steer={steer:+.2f}", (10, 30),
                            font, 0.8, (0, 255, 0), 2)
                if SHOW_DEBUG:
                    cv2.putText(vis, f"area={area} turn_sign={last_turn_sign}", (10, 60),
                                font, 0.7, (255, 255, 255), 2)

            else:
                # No line: use "memory" (last turn direction) and keep turning until reacquired
                lost_dt = now - last_seen_time

                if lost_dt < LINE_LOST_TIMEOUT_S:
                    # very short loss: keep last command briefly (no sudden changes)
                    left = clamp(BASE_SPEED - steer_prev, MIN_SPEED, MAX_SPEED)
                    right = clamp(BASE_SPEED + steer_prev, MIN_SPEED, MAX_SPEED)
                    motor_set(pwmA, pwmB, left, right)
                    cv2.putText(vis, "LINE LOST (short)", (10, 30), font, 0.9, (0, 165, 255), 2)
                else:
                    # Search mode: ramp up turn in the last known direction
                    if last_turn_sign == 0:
                        # if we never had a sign, default: rotate right slowly
                        last_turn_sign = 1

                    # ramp steering magnitude with time lost
                    r = clamp(lost_dt / SEARCH_RAMP_TIME, 0.0, 1.0)
                    steer_mag = SEARCH_STEER_START + r * (SEARCH_STEER_MAX - SEARCH_STEER_START)
                    steer = last_turn_sign * steer_mag

                    # drive: slow forward + strong steering (one wheel slower)
                    left = SEARCH_SPEED - steer
                    right = SEARCH_SPEED + steer

                    # allow one side to go as low as 0 during search (but keep within [0..MAX_SPEED])
                    left = clamp(left, 0, MAX_SPEED)
                    right = clamp(right, 0, MAX_SPEED)

                    motor_set(pwmA, pwmB, left, right)

                    cv2.putText(vis, f"SEARCHING... dt={lost_dt:.2f}s", (10, 30),
                                font, 0.9, (0, 0, 255), 2)

            # FPS
            fps_n += 1
            t = time.time()
            if t - fps_t0 >= 0.5:
                fps_val = fps_n / (t - fps_t0)
                fps_t0 = t
                fps_n = 0
            cv2.putText(vis, f"FPS={fps_val:.1f}", (10, H - 15),
                        font, 0.7, (255, 255, 255), 2)

            cv2.imshow("frame", vis)
            cv2.imshow("mask (ROI)", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        motor_stop(pwmA, pwmB)
        GPIO.cleanup()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()