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

    cv2.putText(
        display,
        "c: capture | q: quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

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
line_h, line_s, line_v = [], [], []
bg_h, bg_s, bg_v = [], [], []

for filename in captured_files:
    img = cv2.imread(filename)
    if img is None:
        print(f"Could not read: {filename}")
        continue

    if USE_FULLSCREEN_ROI:
        roi_for_selection = img.copy()
    else:
        roi_for_selection = img[ROI_START:H, :].copy()

    # Show reminder image
    preview = roi_for_selection.copy()
    cv2.putText(
        preview,
        "Select LINE ROI, then NON-LINE ROI",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    cv2.imshow(f"Preview - {filename}", preview)
    cv2.waitKey(500)
    cv2.destroyWindow(f"Preview - {filename}")

    # ---------------- LINE ROI ----------------
    print(f"\nSelect the LINE region in: {filename}")
    print("Draw a rectangle around the line, then press ENTER or SPACE.")
    print("Press c to cancel this selection.")

    rect_line = cv2.selectROI(
        f"LINE ROI - {filename}",
        roi_for_selection,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow(f"LINE ROI - {filename}")

    x, y, w, h = rect_line
    if w == 0 or h == 0:
        print(f"Skipped LINE ROI for: {filename}")
        continue

    line_roi = roi_for_selection[y:y+h, x:x+w]
    hsv_line = cv2.cvtColor(line_roi, cv2.COLOR_BGR2HSV)

    line_h.extend(hsv_line[:, :, 0].flatten().tolist())
    line_s.extend(hsv_line[:, :, 1].flatten().tolist())
    line_v.extend(hsv_line[:, :, 2].flatten().tolist())

    # ---------------- NON-LINE ROI ----------------
    print(f"Select a NON-LINE region in: {filename}")
    print("Choose floor/table/background pixels, not the line.")
    print("Press ENTER or SPACE to validate, c to cancel.")

    rect_bg = cv2.selectROI(
        f"NON-LINE ROI - {filename}",
        roi_for_selection,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow(f"NON-LINE ROI - {filename}")

    x, y, w, h = rect_bg
    if w == 0 or h == 0:
        print(f"Skipped NON-LINE ROI for: {filename}")
        continue

    bg_roi = roi_for_selection[y:y+h, x:x+w]
    hsv_bg = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV)

    bg_h.extend(hsv_bg[:, :, 0].flatten().tolist())
    bg_s.extend(hsv_bg[:, :, 1].flatten().tolist())
    bg_v.extend(hsv_bg[:, :, 2].flatten().tolist())

if len(line_h) == 0:
    raise RuntimeError("No LINE ROI selected. No line HSV data collected.")

if len(bg_h) == 0:
    raise RuntimeError("No NON-LINE ROI selected. No background HSV data collected.")

line_h = np.array(line_h)
line_s = np.array(line_s)
line_v = np.array(line_v)

bg_h = np.array(bg_h)
bg_s = np.array(bg_s)
bg_v = np.array(bg_v)

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

print("\n================ LINE PIXELS ================")
describe_channel("H", line_h)
describe_channel("S", line_s)
describe_channel("V", line_v)

print("\n============== NON-LINE PIXELS ==============")
describe_channel("H", bg_h)
describe_channel("S", bg_s)
describe_channel("V", bg_v)

# ============================================================
# THRESHOLD SUGGESTION
# ============================================================
# For a black line:
# - H is usually not very useful
# - S and especially V matter more
# A practical first guess is based on upper percentiles of line pixels.
line_s95 = int(np.percentile(line_s, 95))
line_v95 = int(np.percentile(line_v, 95))

print("\nSuggested first black threshold from LINE pixels only:")
print("lower_black = np.array([0, 0, 0])")
print(f"upper_black = np.array([180, {line_s95}, {line_v95}])")

# Optional separation-based suggestion
line_v95_sep = np.percentile(line_v, 95)
bg_v05_sep = np.percentile(bg_v, 5)
line_s95_sep = np.percentile(line_s, 95)
bg_s05_sep = np.percentile(bg_s, 5)

print("\nSeparation hints:")
print(f"LINE V p95       = {line_v95_sep:.2f}")
print(f"NON-LINE V p5    = {bg_v05_sep:.2f}")
print(f"LINE S p95       = {line_s95_sep:.2f}")
print(f"NON-LINE S p5    = {bg_s05_sep:.2f}")

if line_v95_sep < bg_v05_sep:
    suggested_v = int((line_v95_sep + bg_v05_sep) / 2)
    print(f"Suggested Vmax from separation = {suggested_v}")
else:
    suggested_v = int(line_v95_sep)
    print("V distributions overlap too much; using LINE p95 as fallback.")

if line_s95_sep < bg_s05_sep:
    suggested_s = int((line_s95_sep + bg_s05_sep) / 2)
    print(f"Suggested Smax from separation = {suggested_s}")
else:
    suggested_s = int(line_s95_sep)
    print("S distributions overlap too much; using LINE p95 as fallback.")

print("\nPractical threshold to test:")
print("lower_black = np.array([0, 0, 0])")
print(f"upper_black = np.array([180, {suggested_s}, {suggested_v}])")

# ============================================================
# HISTOGRAM DISPLAY
# ============================================================
def draw_histogram(values, bins, width, height, title, color_bgr):
    max_value = 180 if bins == 180 else 256
    hist, _ = np.histogram(values, bins=bins, range=(0, max_value))
    hist = hist.astype(np.float32)

    if hist.max() > 0:
        hist = hist / hist.max()

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(bins):
        x1 = int(i * width / bins)
        x2 = int((i + 1) * width / bins)
        bar_h = int(hist[i] * (height - 40))
        cv2.rectangle(
            canvas,
            (x1, height - 1),
            (x2, height - 1 - bar_h),
            color_bgr,
            -1
        )

    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    return canvas

# LINE histograms
hist_line_h = draw_histogram(line_h, 180, 900, 300, "LINE H distribution", (255, 200, 0))
hist_line_s = draw_histogram(line_s, 256, 900, 300, "LINE S distribution", (0, 255, 255))
hist_line_v = draw_histogram(line_v, 256, 900, 300, "LINE V distribution", (200, 200, 200))

# NON-LINE histograms
hist_bg_h = draw_histogram(bg_h, 180, 900, 300, "NON-LINE H distribution", (0, 100, 255))
hist_bg_s = draw_histogram(bg_s, 256, 900, 300, "NON-LINE S distribution", (0, 0, 255))
hist_bg_v = draw_histogram(bg_v, 256, 900, 300, "NON-LINE V distribution", (255, 255, 255))

cv2.imshow("LINE H histogram", hist_line_h)
cv2.imshow("LINE S histogram", hist_line_s)
cv2.imshow("LINE V histogram", hist_line_v)

cv2.imshow("NON-LINE H histogram", hist_bg_h)
cv2.imshow("NON-LINE S histogram", hist_bg_s)
cv2.imshow("NON-LINE V histogram", hist_bg_v)

print("\nPress any key on a histogram window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()