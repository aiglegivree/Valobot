import time
import cv2
import numpy as np
from picamera2 import Picamera2

# ----------------------------
# CONFIG
# ----------------------------
W, H = 640, 480
ROI_START = int(H * 0.60)        # bottom part of the image
LOOKAHEAD_Y_FRAC = 0.35          # lookahead band inside ROI (0=top of ROI, 1=bottom)
BAND_THICKNESS = 10              # pixels
MIN_AREA = 600                   # reject small blobs (noise)

# If your threshold is inverted, flip this
INVERT_AFTER_OTSU = True         # line is black on white -> True works in most cases

# Optional: show component id / debug
SHOW_DEBUG = True

# ----------------------------
# Helper: pick the "correct" line among multiple blobs
# We select the blob whose lookahead centroid is closest to the previous target (continuity).
# ----------------------------
class LineSelector:
    def __init__(self, w):
        self.w = w
        self.cx_ref = w // 2  # expected line x (in ROI coords)

    def pick(self, mask):
        # Connected components
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

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

            # bottom slice centroid (to ensure it's "in front" of the robot)
            yb0 = int(h * 0.80)
            bottom = comp[yb0:h, :]
            if bottom.sum() == 0:
                continue
            ys_b, xs_b = np.where(bottom > 0)
            cx_bottom = int(xs_b.mean())

            # lookahead band centroid
            band = comp[y0:y1, :]
            if band.sum() == 0:
                # component doesn't reach lookahead: penalize strongly
                cx_look = cx_bottom
                miss_penalty = 1.0
            else:
                ys_l, xs_l = np.where(band > 0)
                cx_look = int(xs_l.mean())
                miss_penalty = 0.0

            # Score: continuity to previous target + prefer bigger area + penalize missing lookahead
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

        # Update reference (EMA) for continuity
        alpha = 0.35
        self.cx_ref = int((1 - alpha) * self.cx_ref + alpha * cx_look)

        return {
            "comp_id": comp_id,
            "cx_bottom": cx_bottom,
            "cx_look": cx_look,
            "area": area,
            "y_look": int(mask.shape[0] * LOOKAHEAD_Y_FRAC),
        }


def main():
    # ----------------------------
    # Picamera2 setup
    # ----------------------------
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"format": "BGR888", "size": (W, H)},
        controls={"FrameRate": 60}
    ))
    cam.start()

    selector = LineSelector(W)

    kernel = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fps_t0 = time.time()
    fps_n = 0
    fps_val = 0.0

    try:
        while True:
            frame = cam.capture_array()
            if frame is None:
                break

            # 1) ROI
            roi = frame[ROI_START:H, :]

            # 2) Preprocess + Otsu
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Otsu
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if INVERT_AFTER_OTSU:
                mask = 255 - mask  # black line becomes white blob

            # cleanup
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # 3) Choose correct line among multiple candidates
            pick = selector.pick(mask)

            # Visualization
            vis = frame.copy()
            cv2.rectangle(vis, (0, ROI_START), (W - 1, H - 1), (0, 255, 255), 2)

            # draw lookahead band on ROI in global coords
            y_look_roi = int(mask.shape[0] * LOOKAHEAD_Y_FRAC)
            y0 = ROI_START + max(0, y_look_roi - BAND_THICKNESS)
            y1 = ROI_START + min(mask.shape[0], y_look_roi + BAND_THICKNESS)
            cv2.rectangle(vis, (0, y0), (W - 1, y1), (255, 255, 0), 1)

            if pick is None:
                cv2.putText(vis, "LINE NOT FOUND", (10, 30),
                            font, 0.9, (0, 0, 255), 2)
            else:
                cx_look = pick["cx_look"]
                cx_bottom = pick["cx_bottom"]
                area = pick["area"]

                # Convert ROI coords -> global coords for display
                gx_look = cx_look
                gy_look = ROI_START + pick["y_look"]
                gx_bottom = cx_bottom
                gy_bottom = ROI_START + int(mask.shape[0] * 0.90)

                # error normalized [-1, 1] using lookahead point
                e = (cx_look - (W / 2)) / (W / 2)

                cv2.circle(vis, (gx_look, gy_look), 7, (0, 0, 255), -1)      # lookahead
                cv2.circle(vis, (gx_bottom, gy_bottom), 7, (255, 0, 0), -1)  # bottom
                cv2.line(vis, (W // 2, H - 1), (gx_look, gy_look), (255, 0, 0), 2)

                cv2.putText(vis, f"cx_look={cx_look}  e={e:+.3f}", (10, 30),
                            font, 0.9, (0, 255, 0), 2)
                if SHOW_DEBUG:
                    cv2.putText(vis, f"area={area}  ref={selector.cx_ref}", (10, 65),
                                font, 0.7, (255, 255, 255), 2)

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
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()