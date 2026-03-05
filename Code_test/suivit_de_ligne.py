import time
import cv2
import numpy as np
import pigpio
from picamera2 import Picamera2

# ----------------------------
# Motor driver (H-bridge)
# ----------------------------
class MotorDriver:
    """
    Each motor: IN1, IN2 for direction, PWM for speed.
    Commands: -1..1
    """
    def __init__(self, pi, left_in1, left_in2, left_pwm,
                       right_in1, right_in2, right_pwm,
                       pwm_freq=20000):
        self.pi = pi
        self.L1, self.L2, self.LPWM = left_in1, left_in2, left_pwm
        self.R1, self.R2, self.RPWM = right_in1, right_in2, right_pwm

        for pin in [self.L1, self.L2, self.LPWM, self.R1, self.R2, self.RPWM]:
            self.pi.set_mode(pin, pigpio.OUTPUT)

        self.pi.set_PWM_frequency(self.LPWM, pwm_freq)
        self.pi.set_PWM_frequency(self.RPWM, pwm_freq)

        self.stop()

    def _set_motor(self, in1, in2, pwm, cmd):
        cmd = max(-1.0, min(1.0, cmd))
        duty = int(abs(cmd) * 255)

        if duty == 0:
            self.pi.write(in1, 0)
            self.pi.write(in2, 0)
            self.pi.set_PWM_dutycycle(pwm, 0)
            return

        if cmd > 0:
            self.pi.write(in1, 1)
            self.pi.write(in2, 0)
        else:
            self.pi.write(in1, 0)
            self.pi.write(in2, 1)

        self.pi.set_PWM_dutycycle(pwm, duty)

    def set(self, left_cmd, right_cmd):
        self._set_motor(self.L1, self.L2, self.LPWM, left_cmd)
        self._set_motor(self.R1, self.R2, self.RPWM, right_cmd)

    def stop(self):
        self.set(0.0, 0.0)


# ----------------------------
# Line follower core
# ----------------------------
class LineFollower:
    def __init__(self, w, h):
        self.w = w
        self.h = h

        # State: expected x position in bird-view
        self.x_ref = w // 2
        self.x_ref_vel = 0.0

        # ROI in original frame
        self.roi_y0 = int(h * 0.45)
        self.roi_y1 = h
        self.roi_h = self.roi_y1 - self.roi_y0

        # Lookahead in bird-view coordinates (0..roi_h-1)
        self.lookahead_y = int(self.roi_h * 0.35)

        # Perspective transform (tune these points!)
        self.src = np.float32([
            [int(0.12*w), int(0.05*self.roi_h)],
            [int(0.88*w), int(0.05*self.roi_h)],
            [int(0.98*w), int(0.98*self.roi_h)],
            [int(0.02*w), int(0.98*self.roi_h)],
        ])
        self.dst = np.float32([
            [int(0.20*w), 0],
            [int(0.80*w), 0],
            [int(0.80*w), self.roi_h - 1],
            [int(0.20*w), self.roi_h - 1],
        ])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)

        # Controller gains
        self.kp = 0.9
        self.kd = 0.25
        self.ki = 0.0
        self.err_i = 0.0
        self.err_prev = 0.0

        # Speed policy
        self.v_max = 0.65
        self.v_min = 0.25
        self.steer_limit = 0.75

    def preprocess(self, frame_bgr):
        roi = frame_bgr[self.roi_y0:self.roi_y1, :]

        bird = cv2.warpPerspective(roi, self.M, (self.w, self.roi_h),
                                   flags=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Black line => 1 (invert)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

        return roi, bird, bw

    def pick_target(self, bw):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num <= 1:
            return None

        x_ref_pred = self.x_ref + self.x_ref_vel
        band_y = self.lookahead_y
        band_th = 6

        candidates = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 250:
                continue

            comp = (labels == i).astype(np.uint8)

            # bottom band
            yb0 = int(self.roi_h * 0.85)
            bottom = comp[yb0:self.roi_h, :]
            if bottom.sum() == 0:
                continue
            _, xs = np.where(bottom > 0)
            x_bottom = int(xs.mean())

            # lookahead band
            y0 = max(0, band_y - band_th)
            y1 = min(self.roi_h, band_y + band_th)
            band = comp[y0:y1, :]
            if band.sum() == 0:
                x_look = x_bottom
                miss_penalty = 1.0
            else:
                _, xs2 = np.where(band > 0)
                x_look = int(xs2.mean())
                miss_penalty = 0.0

            score = (
                1.2 * abs(x_look - x_ref_pred) +
                0.6 * abs(x_bottom - self.x_ref) +
                400.0 * miss_penalty -
                0.002 * area
            )
            candidates.append((score, x_bottom, x_look, area, comp))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0])
        _, x_bottom, x_look, area, comp = candidates[0]
        return x_bottom, x_look, area, comp

    def update_ref(self, x_look):
        x_prev = self.x_ref
        alpha = 0.35
        self.x_ref = int((1 - alpha) * self.x_ref + alpha * x_look)
        self.x_ref_vel = 0.6 * self.x_ref_vel + 0.4 * (self.x_ref - x_prev)

    def control(self, x_look, x_bottom, dt):
        x_center = self.w // 2
        err = (x_look - x_center) / (self.w / 2)

        self.err_i += err * dt
        derr = (err - self.err_prev) / max(dt, 1e-3)
        self.err_prev = err

        steer = self.kp * err + self.ki * self.err_i + self.kd * derr
        steer = max(-self.steer_limit, min(self.steer_limit, steer))

        curv = abs(x_look - x_bottom) / (self.w / 2)
        v = self.v_max - 0.55 * curv
        v = max(self.v_min, min(self.v_max, v))

        left = v - steer
        right = v + steer

        m = max(1.0, abs(left), abs(right))
        return left / m, right / m, err, steer, v


def main():
    # Camera config
    W, H = 320, 240

    # pigpio setup
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio not connected. Start pigpiod: sudo systemctl start pigpiod")

    # TODO: set your GPIO pins (BCM numbering)
    LEFT_IN1, LEFT_IN2, LEFT_PWM = 5, 6, 12
    RIGHT_IN1, RIGHT_IN2, RIGHT_PWM = 20, 21, 13

    motors = MotorDriver(pi,
                         LEFT_IN1, LEFT_IN2, LEFT_PWM,
                         RIGHT_IN1, RIGHT_IN2, RIGHT_PWM)

    follower = LineFollower(W, H)

    # Camera
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"format": "BGR888", "size": (W, H)},
        controls={"FrameRate": 60}
    ))
    cam.start()

    t_prev = time.time()

    # Visualization layout sizes
    scale = 1  # set to 2 if you want bigger windows
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        while True:
            frame = cam.capture_array()
            now = time.time()
            dt = now - t_prev
            t_prev = now

            roi, bird, bw = follower.preprocess(frame)
            picked = follower.pick_target(bw)

            if picked is None:
                motors.set(0.18, 0.18)
                # Visualization even when lost
                bw_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
                cv2.putText(bw_vis, "LOST LINE", (10, 20), font, 0.6, (0, 0, 255), 2)
                cv2.imshow("camera", frame)
                cv2.imshow("bird_view", bird)
                cv2.imshow("line_mask", bw_vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            x_bottom, x_look, area, comp = picked
            follower.update_ref(x_look)

            left, right, err, steer, v = follower.control(x_look, x_bottom, dt)
            motors.set(left, right)

            # ----------------------------
            # VISUALIZATION
            # ----------------------------

            # 1) Mask visualization with overlays
            vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

            # center reference line
            cv2.line(vis, (W // 2, 0), (W // 2, vis.shape[0] - 1), (255, 255, 0), 1)

            # lookahead + bottom points
            cv2.circle(vis, (x_look, follower.lookahead_y), 6, (0, 255, 0), -1)
            cv2.circle(vis, (x_bottom, int(vis.shape[0] * 0.90)), 6, (255, 0, 0), -1)

            # predicted reference
            cv2.circle(vis, (follower.x_ref, follower.lookahead_y), 7, (0, 0, 255), 2)

            # Text info
            cv2.putText(vis, f"err={err:+.2f}", (10, 20), font, 0.55, (255, 255, 255), 1)
            cv2.putText(vis, f"v={v:.2f}", (10, 40), font, 0.55, (255, 255, 255), 1)
            cv2.putText(vis, f"steer={steer:+.2f}", (10, 60), font, 0.55, (255, 255, 255), 1)
            cv2.putText(vis, f"area={area}", (10, 80), font, 0.55, (255, 255, 255), 1)

            # 2) Overlay on raw camera (driver view)
            overlay = frame.copy()
            # Map bird-view lookahead to ROI coords approximately:
            # Since bird is warped, exact inverse mapping would be cv2.perspectiveTransform.
            # For visualization only, we show a marker in ROI with same x and y at lookahead position.
            y_on_frame = follower.roi_y0 + follower.lookahead_y
            cv2.circle(overlay, (x_look, y_on_frame), 6, (0, 255, 0), -1)
            cv2.line(overlay, (W // 2, H - 1), (x_look, y_on_frame), (0, 255, 0), 2)

            # Show windows
            if scale != 1:
                cv2.imshow("camera", cv2.resize(overlay, (W * scale, H * scale)))
                cv2.imshow("bird_view", cv2.resize(bird, (W * scale, follower.roi_h * scale)))
                cv2.imshow("line_mask", cv2.resize(vis, (W * scale, follower.roi_h * scale)))
            else:
                cv2.imshow("camera", overlay)
                cv2.imshow("bird_view", bird)
                cv2.imshow("line_mask", vis)

            # ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        motors.stop()
        cam.stop()
        pi.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()