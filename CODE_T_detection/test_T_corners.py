import cv2
import numpy as np
from leds import LED

# Configuration camera
W = 640
H = 480
CAM_ID = 0
USE_PICAMERA2 = True

# ==============================
# PRE-FILTER
# ==============================
GAUSS_KSIZE = 5  # odd

# ==============================
# BLACK HSV THRESHOLD
# ==============================
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0

BLACK_H_HIGH = 180
BLACK_S_HIGH = 255
BLACK_V_HIGH = 130

# ==============================
# MORPHO
# ==============================
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2

# ==============================
# ROI
# ==============================
ROI_START_FRAC = 0.50

# ==============================
# CORNER DETECTION
# ==============================
PADDING = 1
APPROX_EPSILON_FRAC = 0.02
MIN_CONTOUR_AREA = 1200
TARGET_SHARP_TURN_CORNERS = 6
TARGET_T_CORNERS = 8


class Camera:
    def __init__(self):
        if USE_PICAMERA2:
            from picamera2 import Picamera2

            self.cam = Picamera2()
            self.cam.configure(
                self.cam.create_preview_configuration(
                    main={"size": (W, H), "format": "RGB888"}
                )
            )
            self.cam.start()
            self.type = "picamera2"
        else:
            cap = cv2.VideoCapture(CAM_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            if not cap.isOpened():
                raise RuntimeError("Camera not found")
            self.cam = cap
            self.type = "opencv"

    def read(self):
        if self.type == "picamera2":
            frame = self.cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame

        ret, frame = self.cam.read()
        return ret, frame

    def release(self):
        if self.type == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()


def keep_only_bottom_connected(mask):
    """
    Keep only white pixels connected to the bottom line.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    queue = []

    for x in range(w):
        if mask[h - 1, x] == 255:
            visited[h - 1, x] = 1
            queue.append((h - 1, x))

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if visited[ny, nx] == 0 and mask[ny, nx] == 255:
                    visited[ny, nx] = 1
                    queue.append((ny, nx))

    out = np.zeros_like(mask)
    out[visited == 1] = 255
    return out


def compute_mask(frame):
    h_frame, _ = frame.shape[:2]
    roi_start = int(h_frame * ROI_START_FRAC)
    roi = frame[roi_start:]

    roi_blur = cv2.GaussianBlur(roi, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = keep_only_bottom_connected(mask)

    return mask, roi_start


def add_one_pixel_padding(mask):
    """
    Add a 1-pixel black border around the mask so contour extraction
    also behaves correctly when the shape touches the image edge.
    """
    return cv2.copyMakeBorder(
        mask,
        PADDING,
        PADDING,
        PADDING,
        PADDING,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )


def extract_corners_from_contour(mask):
    """
    Detect corners from the contour polygon approximation.
    This is different from the skeleton/endpoint method used in test_T.py.
    """
    padded = add_one_pixel_padding(mask)

    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], None, padded

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_CONTOUR_AREA:
        return [], biggest, padded

    perimeter = cv2.arcLength(biggest, True)
    epsilon = APPROX_EPSILON_FRAC * perimeter
    approx = cv2.approxPolyDP(biggest, epsilon, True)

    corners = []
    for point in approx.reshape(-1, 2):
        x, y = int(point[0]) - PADDING, int(point[1]) - PADDING
        corners.append((x, y))

    return corners, biggest, padded


def is_sharp_turn(corner_count):
    return corner_count == TARGET_SHARP_TURN_CORNERS


def is_t_shape(corner_count):
    return corner_count == TARGET_T_CORNERS


def draw_contour_overlay(mask, contour, corners):
    disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if contour is not None and len(contour) > 0:
        shifted = contour - np.array([[[PADDING, PADDING]]], dtype=contour.dtype)
        cv2.drawContours(disp, [shifted], -1, (0, 255, 0), 2)

    for x, y in corners:
        if 0 <= x < disp.shape[1] and 0 <= y < disp.shape[0]:
            cv2.circle(disp, (x, y), 5, (255, 0, 0), -1)

    return disp


def main():
    leds = LED()
    leds.turn_off_all()

    cam = Camera()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            mask, _ = compute_mask(frame)
            corners, contour, _ = extract_corners_from_contour(mask)

            corner_count = len(corners)
            is_turn = is_sharp_turn(corner_count)
            is_t = is_t_shape(corner_count)

            disp = draw_contour_overlay(mask, contour, corners)

            cv2.putText(
                disp,
                f"Corners: {corner_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                disp,
                f"Sharp turn: {'YES' if is_turn else 'NO'}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if is_turn else (0, 0, 255),
                2,
            )

            cv2.putText(
                disp,
                f"T: {'YES' if is_t else 'NO'}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if is_t else (0, 0, 255),
                2,
            )

            cv2.imshow("mask_corners", disp)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
