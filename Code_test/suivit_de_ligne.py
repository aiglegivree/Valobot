import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

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

# Fix color display (Picamera2 gives RGB sometimes; OpenCV expects BGR for imshow)
FIX_DISPLAY_COLOR_SWAP = True

# ----------------------------
# CONFIG (MOTORS - L298N, BCM pins)  (your updated mapping)
# ----------------------------
# L298N -> Raspberry Pi (BCM)
# ENB -> GPIO18 (pin 12)  PWM
# IN3 -> GPIO23 (pin 16)
# IN4 -> GPIO24 (pin 18)
# ENA -> GPIO13 (pin 33)  PWM
# IN1 -> GPIO27 (pin 13)
# IN2 -> GPIO22 (pin 15)
ENA, IN1, IN2 = 13, 27, 22
ENB, IN3, IN4 = 18, 23, 24

PWM_FREQ = 1000

# Speed limits (slow)
BASE_SPEED = 20
MAX_SPEED  = 30
MIN_SPEED  = 18

# ----------------------------
# STEERING: anticipate LESS
# Reason: lookahead point is far ahead of rotation center.
# We blend bottom+lookahead errors with small weight on lookahead.
# ----------------------------
DEADBAND = 0.05

# weights: reduce lookahead influence strongly
W_LOOK   = 0.25
W_BOTTOM = 0.75

# proportional gain on blended error (small)
Kp = 10.0

# (optional) low-pass on steering to reduce jerks
STEER_LPF_ALPHA = 0.35  # 0=no update, 1=no filtering
steer_prev = 0.0

# If one motor is physically inverted, flip it here
INVERT_LEFT_MOTOR  = False  # Motor A (ENA, IN1, IN2)
INVERT_RIGHT_MOTOR = False  # Motor B (ENB, IN3, IN4)

# Safety
LINE_LOST_TIMEOUT_S = 0.25


# ----------------------------
# Helper: pick the correct line among multiple blobs
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

            yb0 = int(h * 0.80)
            bottom = comp[yb0:h, :]
            if bottom.sum() == 0:
                continue
            ys_b, xs_b = np.where(bottom > 0)
            cx_bottom = int(xs_b.mean())

            band = comp[y0:y1, :]
            if band.sum() == 0:
                cx_look = cx_bottom
                miss_penalty = 1.0
            else:
                ys_l, xs_l = np.where(band > 0)
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
    # optional inversion
    if INVERT_LEFT_MOTOR:
        left_speed = -left_speed
    if INVERT_RIGHT_MOTOR:
        right_speed = -right_speed

    # Left (Motor A) -> ENA/IN1/IN2
    ls = int(clamp(left_speed, -100, 100))
    if ls >= 0:
        GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    else:
        GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    pwmA.ChangeDutyCycle(abs(ls))

    # Right (Motor B) -> ENB/IN3/IN4
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


def main():
    global steer_prev

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

    last_line_time = time.time()

    try:
        while True:
            frame = cam.capture_array()
            if frame is None:
                break

            # If colors look swapped in the displayed window, convert for visualization only
            vis_src = frame
            if FIX_DISPLAY_COLOR_SWAP:
                # swap R<->B for correct imshow rendering if needed
                vis_src = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # ROI (use original frame data; grayscale conversion is fine either way)
            roi = frame[ROI_START:H, :]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if INVERT_AFTER_OTSU:
                mask = 255 - mask

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            pick = selector.pick(mask)

            # Visualization
            vis = vis_src.copy()
            cv2.rectangle(vis, (0, ROI_START), (W - 1, H - 1), (0, 255, 255), 2)

            y_look_roi = int(mask.shape[0] * LOOKAHEAD_Y_FRAC)
            y0 = ROI_START + max(0, y_look_roi - BAND_THICKNESS)
            y1 = ROI_START + min(mask.shape[0], y_look_roi + BAND_THICKNESS)
            cv2.rectangle(vis, (0, y0), (W - 1, y1), (255, 255, 0), 1)

            if pick is None:
                cv2.putText(vis, "LINE NOT FOUND", (10, 30), font, 0.9, (0, 0, 255), 2)
                if (time.time() - last_line_time) > LINE_LOST_TIMEOUT_S:
                    motor_stop(pwmA, pwmB)
            else:
                last_line_time = time.time()

                cx_look = pick["cx_look"]
                cx_bottom = pick["cx_bottom"]
                area = pick["area"]

                gx_look = cx_look
                gy_look = ROI_START + pick["y_look"]
                gx_bottom = cx_bottom
                gy_bottom = ROI_START + int(mask.shape[0] * 0.90)

                # errors normalized [-1,1]
                e_look = (cx_look - (W / 2)) / (W / 2)
                e_bot  = (cx_bottom - (W / 2)) / (W / 2)

                # Blend -> less anticipation (lookahead has small weight)
                e = W_LOOK * e_look + W_BOTTOM * e_bot

                cv2.circle(vis, (gx_look, gy_look), 7, (0, 0, 255), -1)       # red (now correct if swap fixed)
                cv2.circle(vis, (gx_bottom, gy_bottom), 7, (255, 0, 0), -1)   # blue
                cv2.line(vis, (W // 2, H - 1), (gx_look, gy_look), (255, 0, 0), 2)

                cv2.putText(vis, f"e_look={e_look:+.3f} e_bot={e_bot:+.3f} e={e:+.3f}", (10, 30),
                            font, 0.7, (0, 255, 0), 2)
                if SHOW_DEBUG:
                    cv2.putText(vis, f"area={area} ref={selector.cx_ref}", (10, 60),
                                font, 0.7, (255, 255, 255), 2)

                # P-control on blended error (and filtered)
                if abs(e) < DEADBAND:
                    steer = 0.0
                else:
                    steer = Kp * e

                # low-pass filter for smoother motion
                steer = (1.0 - STEER_LPF_ALPHA) * steer_prev + STEER_LPF_ALPHA * steer
                steer_prev = steer

                left = BASE_SPEED - steer
                right = BASE_SPEED + steer

                # keep slow + forward only
                left = clamp(left, MIN_SPEED, MAX_SPEED)
                right = clamp(right, MIN_SPEED, MAX_SPEED)

                motor_set(pwmA, pwmB, left, right)

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