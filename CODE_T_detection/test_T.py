import cv2
import numpy as np
from leds import LED

leds = LED()
leds.turn_on_all()
# ==============================
# CAMERA
# ==============================
W = 640
H = 480
CAM_ID = 0

# ==============================
# BLACK HSV THRESHOLD (comme demandé)
# ==============================
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0

BLACK_H_HIGH = 180
BLACK_S_HIGH = 170
BLACK_V_HIGH = 110

# ==============================
# MORPHO (comme vision actuelle)
# ==============================
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2

# ==============================
# ROI (même logique vision)
# ==============================
ROI_START_FRAC = 0.50


def compute_mask(frame):
    Hf, Wf = frame.shape[:2]

    # ROI bas
    roi_start = int(Hf * ROI_START_FRAC)
    roi = frame[roi_start:]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW])
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH])

    mask = cv2.inRange(hsv, low, high)

    # morpho close
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)

    return mask, roi_start


def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        mask, roi_start = compute_mask(frame)

        # Calcul des moyennes par colonne et par ligne
        H_mask, W_mask = mask.shape
        column_means = []
        row_means = []
        
        for col in range(W_mask):
            y_indices = np.where(mask[:, col] == 255)[0]
            if len(y_indices) > 0:
                mean_y_roi = np.mean(y_indices)
                column_means.append((col, int(mean_y_roi)))
        
        for row in range(H_mask):
            x_indices = np.where(mask[row, :] == 255)[0]
            if len(x_indices) > 0:
                mean_x = np.mean(x_indices)
                row_means.append((int(mean_x), row))

        # display
        frame_disp = frame.copy()
        frame_disp[roi_start:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Dessiner les lignes sur l'overlay
        roi_disp = frame_disp[roi_start:]
        if column_means:
            cv2.polylines(roi_disp, [np.array(column_means, dtype=np.int32)], False, (0, 255, 0), 2)
        if row_means:
            cv2.polylines(roi_disp, [np.array(row_means, dtype=np.int32)], False, (255, 0, 0), 2)
        

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("overlay", frame_disp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()