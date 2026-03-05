import time
import cv2
import numpy as np
from picamera2 import Picamera2

# ----------------------------
# Line follower (camera only)
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

        # Perspective transform (TUNE these points!)
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

        # Controller signals (still computed for visualization/tuning)
        self.kp = 0.9
        self.kd = 0.25
        self.ki = 0.0
        self.err_i = 0.0
        self.err_prev = 0.0

        self.v_max = 0.65
        self.v_min = 0.25
        self.steer_limit = 0.75

    def preprocess(self, frame_bgr):
        roi = frame_bgr[self.roi_y0:self.roi_y1, :]

        bird = cv2.warpPerspective(
            roi, self.M, (self.w, self.roi_h),
            flags=cv2.INTER_LINEAR
        )

        gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Black line => 1 (invert)
        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

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

    def debug_control_values(self, x_look, x_bottom, dt):
        """
        Computes err/steer/v for visualization only (no motors).
        """
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

        return err, steer, v


def main():
    W, H = 320, 240
    scale = 1  # set to 2 for bigger windows

    follower = LineFollower(W, H)

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"format": "BGR888", "size": (W, H)},
        controls={"FrameRate": 60}
    ))
    cam.start()

    font = cv2.FONT_HERSHEY_SIMPLEX
    t_prev = time.time()

    try:
        while True:
            frame = cam.capture_array()
            now = time.time()
            dt = now - t_prev
            t_prev = now

            roi, bird, bw = follower.preprocess(frame)
            picked = follower.pick_target(bw)

            # Visualization windows
            overlay = frame.copy()
            mask_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

            # Draw ROI rectangle on full frame
            cv2.rectangle(overlay, (0, follower.roi_y0), (W - 1, follower.roi_y1 - 1), (0, 255, 255), 1)

            # Draw perspective src points (in ROI coordinates -> shift by roi_y0)
            for (x, y) in follower.src:
                cv2.circle(overlay, (int(x), int(y + follower.roi_y0)), 4, (0, 255, 255), -1)

            # Center line in bird/mask
            cv2.line(mask_vis, (W // 2, 0), (W // 2, mask_vis.shape[0] - 1), (255, 255, 0), 1)

            if picked is None:
                cv2.putText(mask_vis, "LOST LINE", (10, 20), font, 0.6, (0, 0, 255), 2)
                cv2.putText(overlay, "LOST LINE", (10, 25), font, 0.7, (0, 0, 255), 2)
            else:
                x_bottom, x_look, area, _comp = picked
                follower.update_ref(x_look)
                err, steer, v = follower.debug_control_values(x_look, x_bottom, dt)

                # Points on mask
                cv2.circle(mask_vis, (x_look, follower.lookahead_y), 6, (0, 255, 0), -1)
                cv2.circle(mask_vis, (x_bottom, int(mask_vis.shape[0] * 0.90)), 6, (255, 0, 0), -1)
                cv2.circle(mask_vis, (follower.x_ref, follower.lookahead_y), 7, (0, 0, 255), 2)

                # Info text
                cv2.putText(mask_vis, f"err={err:+.2f}", (10, 20), font, 0.55, (255, 255, 255), 1)
                cv2.putText(mask_vis, f"steer={steer:+.2f}", (10, 40), font, 0.55, (255, 255, 255), 1)
                cv2.putText(mask_vis, f"v_policy={v:.2f}", (10, 60), font, 0.55, (255, 255, 255), 1)
                cv2.putText(mask_vis, f"area={area}", (10, 80), font, 0.55, (255, 255, 255), 1)

                # Approx draw target on full frame (y = roi_y0 + lookahead_y)
                y_on_frame = follower.roi_y0 + follower.lookahead_y
                cv2.circle(overlay, (x_look, y_on_frame), 6, (0, 255, 0), -1)
                cv2.line(overlay, (W // 2, H - 1), (x_look, y_on_frame), (0, 255, 0), 2)

            # Show
            if scale != 1:
                cv2.imshow("camera", cv2.resize(overlay, (W * scale, H * scale)))
                cv2.imshow("bird_view", cv2.resize(bird, (W * scale, follower.roi_h * scale)))
                cv2.imshow("line_mask", cv2.resize(mask_vis, (W * scale, follower.roi_h * scale)))
            else:
                cv2.imshow("camera", overlay)
                cv2.imshow("bird_view", bird)
                cv2.imshow("line_mask", mask_vis)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()