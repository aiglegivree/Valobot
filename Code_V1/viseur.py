from __future__ import annotations
import cv2
import numpy as np
import time

from motors import MotorDriver
from camera import read_frame  

# ============================================================
# CONFIG
# ============================================================
W, H = 640, 480

# Seuil bleu en HSV
# A ajuster selon ta lumière
BLUE_LOW  = np.array([90, 80, 50], dtype=np.uint8)
BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

# Taille minimale de cible pour éviter le bruit
MIN_TARGET_AREA = 1200

# Commande rotation
TURN_SPEED = 0.30

# Tolérance de centrage (en pixels)
CENTER_TOL = 35

# Nombre de frames centrées avant de tirer
LOCK_FRAMES_REQUIRED = 5

# Debug affichage
SHOW_DEBUG = True


# ============================================================
# DETECTION CIBLE BLEUE
# ============================================================
def detect_blue_target(frame):
    """
    Retourne:
        found (bool)
        cx (float or None): centre x de la cible
        cy (float or None): centre y de la cible
        area (float): aire du plus grand contour
        mask (np.ndarray): masque binaire pour debug
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)

    # Nettoyage bruit
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, None, None, 0, mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < MIN_TARGET_AREA:
        return False, None, None, area, mask

    M = cv2.moments(c)
    if M["m00"] == 0:
        return False, None, None, area, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return True, cx, cy, area, mask


# ============================================================
# CONTROLE
# ============================================================
def aim_and_fire():
    motor = MotorDriver()
    lock_count = 0

    try:
        while True:
            frame = read_frame()
            if frame is None:
                motor.set_speed(0, 0)
                continue

            frame = cv2.resize(frame, (W, H))

            found, cx, cy, area, mask = detect_blue_target(frame)

            img_center_x = W // 2

            if not found:
                # Pas de cible -> rotation de recherche
                motor.set_speed(-TURN_SPEED, TURN_SPEED)
                lock_count = 0
                status = "search"

            else:
                err = cx - img_center_x

                if abs(err) <= CENTER_TOL:
                    # cible centrée
                    motor.set_speed(0, 0)
                    lock_count += 1
                    status = f"locked ({lock_count}/{LOCK_FRAMES_REQUIRED})"

                    if lock_count >= LOCK_FRAMES_REQUIRED:
                        print("fire")
                        time.sleep(0.5)
                        lock_count = 0

                else:
                    lock_count = 0

                    # tourne vers la cible
                    if err < 0:
                        # cible à gauche -> tourne à gauche
                        motor.set_speed(-TURN_SPEED, TURN_SPEED)
                        status = "turn left"
                    else:
                        # cible à droite -> tourne à droite
                        motor.set_speed(TURN_SPEED, -TURN_SPEED)
                        status = "turn right"

            if SHOW_DEBUG:
                debug = frame.copy()

                # ligne centre image
                cv2.line(debug, (img_center_x, 0), (img_center_x, H), (0, 255, 0), 2)

                if found:
                    cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(debug, f"cx={cx} area={int(area)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

                cv2.putText(debug, status,
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

                cv2.imshow("debug", debug)
                cv2.imshow("mask_blue", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

    finally:
        motor.set_speed(0, 0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    aim_and_fire()