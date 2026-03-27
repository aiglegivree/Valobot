import cv2
import numpy as np
from leds import LED

# Configuration caméra
W = 640
H = 480
CAM_ID = 0
USE_PICAMERA2 = True

# ==============================
# PRE-FILTER
# ==============================
GAUSS_KSIZE = 5  # impair

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
        else:
            ret, frame = self.cam.read()
            return ret, frame

    def release(self):
        if self.type == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()


def keep_only_bottom_connected(mask):
    """
    Garde uniquement les pixels blancs connectés à la ligne du bas.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    queue = []

    # Seeds = pixels blancs sur la dernière ligne
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

    # blur avant threshold
    roi_blur = cv2.GaussianBlur(roi, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)

    # closing
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)

    # garder seulement ce qui est connecté au bas
    mask = keep_only_bottom_connected(mask)

    return mask, roi_start


def skeletonize(mask):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask)

    skel = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


def count_endpoints_and_junctions(skeleton):
    """
    Comptage robuste avec padding pour gérer les bords.
    """

    # --- padding ---
    sk = (skeleton > 0).astype(np.uint8)
    sk_pad = np.pad(sk, pad_width=1, mode='constant', constant_values=0)

    h, w = sk_pad.shape

    endpoints = []
    junctions = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if sk_pad[y, x] == 0:
                continue

            patch = sk_pad[y - 1:y + 2, x - 1:x + 2]
            n = int(np.sum(patch)) - 1

            if n == 1:
                endpoints.append((x - 1, y - 1))  # ← remove padding offset
            elif n >= 3:
                junctions.append((x - 1, y - 1))

    return endpoints, junctions

def main():
    leds = LED()
    leds.turn_off_all()

    cam = Camera()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            mask, roi_start = compute_mask(frame)
            skeleton = skeletonize(mask)

            endpoints, junctions = count_endpoints_and_junctions(skeleton)

            branch_count = len(endpoints)

            # représentation unique : mask en BGR
            disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # afficher le squelette en vert
            disp[skeleton == 255] = (0, 255, 0)

            # afficher endpoints en bleu
            for x, y in endpoints:
                cv2.circle(disp, (x, y), 4, (255, 0, 0), -1)

            # afficher junctions en rouge
            for x, y in junctions:
                cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)

            # texte
            cv2.putText(
                disp,
                f"Branches: {branch_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                disp,
                f"Endpoints: {len(endpoints)}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                disp,
                f"Junctions: {len(junctions)}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("mask", disp)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()