import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# Initialisation caméra Pi
# =========================
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(preview_config)
picam2.start()

print("Press 'q' to quit.")


def fit_bounded_line_x_of_y(points):
    """
    Fit x = m*y + b to a set of points and keep only the supported y-range.
    points: list of (x, y)

    returns:
        (m, b, y_min, y_max) or None
    """
    if len(points) < 2:
        return None

    pts = np.array(points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]

    A = np.vstack([ys, np.ones_like(ys)]).T

    try:
        m, b = np.linalg.lstsq(A, xs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))

    return float(m), float(b), y_min, y_max


def draw_bounded_line(img, line, color, thickness=5):
    if line is None:
        return

    m, b, y_min, y_max = line
    x1 = int(m * y_min + b)
    x2 = int(m * y_max + b)

    cv2.line(img, (x1, y_min), (x2, y_max), color, thickness)
    cv2.circle(img, (x1, y_min), 6, color, -1)
    cv2.circle(img, (x2, y_max), 6, color, -1)


def filter_segments_by_median_slope(segments, slope_tol=0.5):
    """
    segments: list of (x1, y1, x2, y2)
    slope is defined in x = m*y + b form, so m = dx / dy
    """
    if len(segments) <= 2:
        return segments

    slopes = []
    valid_segments = []

    for seg in segments:
        x1, y1, x2, y2 = seg
        dy = y2 - y1
        if abs(dy) < 1e-6:
            continue

        m = (x2 - x1) / dy
        slopes.append(m)
        valid_segments.append(seg)

    if len(valid_segments) <= 2:
        return valid_segments

    median_slope = np.median(slopes)

    filtered = []
    for seg, m in zip(valid_segments, slopes):
        if abs(m - median_slope) < slope_tol:
            filtered.append(seg)

    return filtered


def bottom_point_of_bounded_line(line):
    """
    Returns the bottom endpoint of a bounded fitted line:
        (x_bottom, y_bottom)
    where y_bottom = y_max
    """
    if line is None:
        return None

    m, b, y_min, y_max = line
    x_bottom = int(m * y_max + b)
    y_bottom = int(y_max)
    return x_bottom, y_bottom


# smoothing state
smoothed_error = None
alpha = 0.8   # larger alpha = more smoothing

while True:
    # Capture image from Raspberry Pi camera
    frame = picam2.capture_array()

    if frame is None:
        print("Error: Could not capture frame from Raspberry Pi camera.")
        break

    original = frame.copy()

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny
    edges = cv2.Canny(blur, 20, 40)

    # # ROI
    height, width = edges.shape
    # mask = np.zeros_like(edges)

    # roi_top = int(height * 0.4)

    # polygon = np.array([[
    #     (0, height - 1),
    #     (0, roi_top),
    #     (width - 1, roi_top),
    #     (width - 1, height - 1)
    # ]], dtype=np.int32)

    # cv2.fillPoly(mask, polygon, 255)
    # edges_roi = cv2.bitwise_and(edges, mask)

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=80,
        maxLineGap=30
    )

    line_frame = original.copy()

    left_segments = []
    right_segments = []
    front_segments = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)

            if length < 10:
                continue

            x_mid = (x1 + x2) / 2
            y_bottom = max(y1, y2)
            ratio = abs(dy) / (abs(dx) + 1e-6)

            # Reject almost perfectly vertical lines
            if ratio > 5.0:
                continue

            # Side-boundary candidates
            if ratio > 0.3:
                # Side lines should reach low enough in the image
                if y_bottom < int(height * 0.65):
                    continue

                if x_mid < width / 2:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))

            # Front candidates
            else:
                front_segments.append((x1, y1, x2, y2))

    # Robustness filter
    left_segments_filtered = filter_segments_by_median_slope(left_segments, slope_tol=0.5)
    right_segments_filtered = filter_segments_by_median_slope(right_segments, slope_tol=0.5)

    # Draw raw filtered side segments
    for x1, y1, x2, y2 in left_segments_filtered:
        cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)   # blue

    for x1, y1, x2, y2 in right_segments_filtered:
        cv2.line(line_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)   # red

    for x1, y1, x2, y2 in front_segments:
        cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # yellow

    # Convert filtered segments to points for fitting
    left_points = []
    for x1, y1, x2, y2 in left_segments_filtered:
        left_points.append((x1, y1))
        left_points.append((x2, y2))

    right_points = []
    for x1, y1, x2, y2 in right_segments_filtered:
        right_points.append((x1, y1))
        right_points.append((x2, y2))

    # Fit bounded segments
    left_line = fit_bounded_line_x_of_y(left_points)
    right_line = fit_bounded_line_x_of_y(right_points)

    # Draw bounded fitted segments
    draw_bounded_line(line_frame, left_line, (255, 255, 0), 5)  # cyan
    draw_bounded_line(line_frame, right_line, (0, 255, 0), 5)   # green

    # Bottom endpoints + center error + smoothing
    left_bottom = bottom_point_of_bounded_line(left_line)
    right_bottom = bottom_point_of_bounded_line(right_line)

    raw_error = None
    smooth_error_to_show = None

    # Draw image center reference
    image_center_x = width // 2
    cv2.line(line_frame, (image_center_x, 0), (image_center_x, height - 1), (255, 255, 255), 1)

    if left_bottom is not None:
        cv2.circle(line_frame, left_bottom, 8, (255, 255, 0), -1)

    if right_bottom is not None:
        cv2.circle(line_frame, right_bottom, 8, (0, 255, 0), -1)

    if left_bottom is not None and right_bottom is not None:
        x_left_bottom, y_left_bottom = left_bottom
        x_right_bottom, y_right_bottom = right_bottom

        corridor_center_x = int((x_left_bottom + x_right_bottom) / 2)
        corridor_center_y = int((y_left_bottom + y_right_bottom) / 2)

        cv2.circle(line_frame, (corridor_center_x, corridor_center_y), 10, (0, 165, 255), -1)
        cv2.line(
            line_frame,
            (image_center_x, corridor_center_y),
            (corridor_center_x, corridor_center_y),
            (0, 165, 255),
            2
        )

        raw_error = corridor_center_x - image_center_x

        if smoothed_error is None:
            smoothed_error = raw_error
        else:
            smoothed_error = alpha * smoothed_error + (1 - alpha) * raw_error

        smooth_error_to_show = smoothed_error

    # Debug text
    cv2.putText(line_frame, f"Left raw: {len(left_segments)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(line_frame, f"Left filt: {len(left_segments_filtered)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(line_frame, f"Right raw: {len(right_segments)}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(line_frame, f"Right filt: {len(right_segments_filtered)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(line_frame, f"Front: {len(front_segments)}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if raw_error is not None:
        cv2.putText(line_frame, f"Raw error: {raw_error:.1f}px", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    if smooth_error_to_show is not None:
        cv2.putText(line_frame, f"Smooth error: {smooth_error_to_show:.1f}px", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Edges", edges)
    cv2.imshow("Robust Bounded Fits", line_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()