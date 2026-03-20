import cv2
import numpy as np
import math
from camera import Camera


proto = False

def fill_holes(mask):
    h, w = mask.shape

    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)

    return filled


def draw_hough_line(image, rho, theta, color, thickness=2):
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def median_line(lines):
    """
    lines: list of (rho, theta)
    returns: (median_rho, median_theta) or None
    """
    if not lines:
        return None

    rhos = np.array([line[0] for line in lines], dtype=np.float32)
    thetas = np.array([line[1] for line in lines], dtype=np.float32)

    median_rho = float(np.median(rhos))
    median_theta = float(np.median(thetas))

    return median_rho, median_theta

if proto == True:
    cap = cv2.VideoCapture("movement.avi")

    if not cap.isOpened():
        print("Error: Cannot open video")
        exit()
else:
    cap = Camera()

while True:
    ret, frame = cap.read()

    if proto and not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    original = frame.copy()
    H, W, _ = original.shape

    blur = cv2.GaussianBlur(original, (3, 3), 1)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array([31, 98, 0], dtype=np.uint8)
    upper = np.array([74, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = fill_holes(mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((17, 17), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLines(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=34
    )

    horizontal_lines = []
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            deg = math.degrees(theta)

            # Horizontal-ish: theta near 90 deg
            if 85 < deg < 95:
                horizontal_lines.append((rho, theta))

            # Left-leaning group
            elif 30 < deg < 80:
                left_lines.append((rho, theta))

            # Right-leaning group
            elif 100 < deg < 150:
                right_lines.append((rho, theta))

    # Get one representative per class
    horizontal_rep = median_line(horizontal_lines)
    left_rep = median_line(left_lines)
    right_rep = median_line(right_lines)

    # Draw only representatives
    if horizontal_rep is not None:
        draw_hough_line(original, horizontal_rep[0], horizontal_rep[1], (0, 255, 255), 3)  # yellow

    if left_rep is not None:
        draw_hough_line(original, left_rep[0], left_rep[1], (0, 0, 255), 3)  # red

    if right_rep is not None:
        draw_hough_line(original, right_rep[0], right_rep[1], (0, 255, 0), 3)  # green

    cv2.imshow("Video Playback", original)
    cv2.imshow("Mask", mask)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()