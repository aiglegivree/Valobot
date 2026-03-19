import cv2
import numpy as np
from picamera2 import Picamera2

# ============================================================
# CONFIG
# ============================================================
W, H = 640, 480
N_IMAGES = 10                      # number of pictures to take
SAVE_PREFIX = "hsv_sample"         # saved image name prefix
USE_FULLSCREEN_ROI = False         # True = select ROI on full image
ROI_START = int(H * 0.60)          # lower part of image if not full image

# ============================================================
# CAMERA SETUP
# ============================================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (W, H), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("Camera started.")
print("Commands:")
print("  c = capture image")
print("  q = quit")
print()

captured_files = []

# ============================================================
# CAPTURE LOOP
# ============================================================
while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    display = frame_bgr.copy()

    if not USE_FULLSCREEN_ROI:
        cv2.rectangle(display, (0, ROI_START), (W - 1, H - 1), (0, 255, 0), 2)

    cv2.putText(display, "c: capture | q: quit", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Camera", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = f"{SAVE_PREFIX}_{len(captured_files):02d}.png"
        cv2.imwrite(filename, frame_bgr)
        captured_files.append(filename)
        print(f"Captured: {filename}")

        if len(captured_files) >= N_IMAGES:
            print(f"{N_IMAGES} images captured.")
            break

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()

if len(captured_files) == 0:
    raise RuntimeError("No images captured.")

# ============================================================
# ROI SELECTION + HSV COLLECTION
# ============================================================
all_h = []
all_s = []
all_v = []

for filename in captured_files:
    img = cv2.imread(filename)
    if img is None:
        continue

    if USE_FULLSCREEN_ROI:
        roi_for_selection = img.copy()
    else:
        roi_for_selection = img[ROI_START:H, :].copy()

    print(f"\nSelect the LINE region in: {filename}")
    print("Draw a rectangle around the line, then press ENTER or SPACE.")
    print("Press c to cancel this image.")

    rect = cv2.selectROI(f"Select LINE ROI - {filename}", roi_for_selection, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Select LINE ROI - {filename}")

    x, y, w, h = rect
    if w == 0 or h == 0:
        print(f"Skipped: {filename}")
        continue

    line_roi = roi_for_selection[y:y+h, x:x+w]

    hsv = cv2.cvtColor(line_roi, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].flatten()
    s_chan = hsv[:, :, 1].flatten()
    v_chan = hsv[:, :, 2].flatten()

    all_h.extend(h_chan.tolist())
    all_s.extend(s_chan.tolist())
    all_v.extend(v_chan.tolist())

if len(all_h) == 0:
    raise RuntimeError("No ROI selected. No HSV data collected.")

all_h = np.array(all_h)
all_s = np.array(all_s)
all_v = np.array(all_v)

# ============================================================
# STATS
# ============================================================
def describe_channel(name, arr):
    print(f"\n{name} statistics:")
    print(f"  min   = {np.min(arr)}")
    print(f"  max   = {np.max(arr)}")
    print(f"  mean  = {np.mean(arr):.2f}")
    print(f"  std   = {np.std(arr):.2f}")
    print(f"  p1    = {np.percentile(arr, 1):.2f}")
    print(f"  p5    = {np.percentile(arr, 5):.2f}")
    print(f"  p10   = {np.percentile(arr, 10):.2f}")
    print(f"  p25   = {np.percentile(arr, 25):.2f}")
    print(f"  p50   = {np.percentile(arr, 50):.2f}")
    print(f"  p75   = {np.percentile(arr, 75):.2f}")
    print(f"  p90   = {np.percentile(arr, 90):.2f}")
    print(f"  p95   = {np.percentile(arr, 95):.2f}")
    print(f"  p99   = {np.percentile(arr, 99):.2f}")

describe_channel("H", all_h)
describe_channel("S", all_s)
describe_channel("V", all_v)

# ============================================================
# THRESHOLD SUGGESTION
# ============================================================
# For a black line, H is usually not very useful.
# A practical first guess is:
# - allow all H
# - set Smax around the 95th percentile
# - set Vmax around the 95th percentile
s_max = int(np.percentile(all_s, 95))
v_max = int(np.percentile(all_v, 95))

print("\nSuggested first black threshold:")
print(f"lower_black = np.array([0, 0, 0])")
print(f"upper_black = np.array([180, {s_max}, {v_max}])")

# ============================================================
# HISTOGRAM DISPLAY
# ============================================================
def draw_histogram(values, bins, width, height, title, color_bgr):
    hist, bin_edges = np.histogram(values, bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)

    if hist.max() > 0:
        hist = hist / hist.max()

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(bins):
        x1 = int(i * width / bins)
        x2 = int((i + 1) * width / bins)
        bar_h = int(hist[i] * (height - 30))
        cv2.rectangle(canvas, (x1, height - 1), (x2, height - 1 - bar_h), color_bgr, -1)

    cv2.putText(canvas, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas

hist_h = draw_histogram(all_h, 180, 900, 300, "Hue distribution", (255, 200, 0))
hist_s = draw_histogram(all_s, 256, 900, 300, "Saturation distribution", (0, 255, 255))
hist_v = draw_histogram(all_v, 256, 900, 300, "Value distribution", (200, 200, 200))

cv2.imshow("H histogram", hist_h)
cv2.imshow("S histogram", hist_s)
cv2.imshow("V histogram", hist_v)

print("\nPress any key on the histogram windows to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()