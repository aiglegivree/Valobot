import cv2
import numpy as np
from leds import LED

# Configuration caméra
W = 640
H = 480
CAM_ID = 0
USE_PICAMERA2 = True


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


leds = LED()
leds.turn_on_all()

# ==============================
# BLACK HSV THRESHOLD
# ==============================
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0

BLACK_H_HIGH = 180
BLACK_S_HIGH = 170
BLACK_V_HIGH = 110

# ==============================
# MORPHO
# ==============================
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2

# ==============================
# DETECTION T
# ==============================
MIN_HORIZONTAL_LENGTH = 250  # Longueur minimum pour la composante horizontale

# ==============================
# ROI
# ==============================
ROI_START_FRAC = 0.50  # ROI sur la moitié inférieure


def compute_mask(frame):
    Hf, Wf = frame.shape[:2]

    # ROI bas
    roi_start = int(Hf * ROI_START_FRAC)
    roi = frame[roi_start:]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)

    # morpho close
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)

    return mask, roi_start


def skeletonize(mask):
    """
    Retourne le squelette binaire du mask.
    Utilise cv2.ximgproc.thinning si disponible, sinon fallback morphologique.
    """
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


def main():
    cam = Camera()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            mask, roi_start = compute_mask(frame)
            skeleton = skeletonize(mask)

            # Utiliser le squelette pour la PCA
            points = np.column_stack(np.where(skeleton == 255))

            # Valeurs par défaut pour éviter les erreurs si aucun point
            is_T = False
            start1 = end1 = start2 = end2 = None
            color1 = (255, 0, 0)
            color2 = (0, 0, 255)
            thickness1 = 1
            thickness2 = 3
            length_left = 0.0
            length_right = 0.0
            most_vertical = "N/A"
            vertical_angle = 0.0
            horizontal_angle = 0.0
            vertical_length = 0.0
            horizontal_length = 0.0
            bar_position = "N/A"
            mean = None

            if len(points) > 1:
                # Centre des points
                mean = np.mean(points, axis=0)
                centered = points - mean

                # Covariance
                cov = np.cov(centered.T)

                # Valeurs/vecteurs propres
                eigvals, eigvecs = np.linalg.eigh(cov)

                # Trier décroissant
                idx = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, idx]
                eigvals = eigvals[idx]

                # Composantes principales
                pc1 = eigvecs[:, 0]
                pc2 = eigvecs[:, 1]

                # Longueur d'affichage
                scale1 = 2 * np.sqrt(max(eigvals[0], 0))
                scale2 = 2 * np.sqrt(max(eigvals[1], 0))

                # Extrémités des lignes PCA
                start1 = mean - scale1 * pc1
                end1 = mean + scale1 * pc1

                start2 = mean - scale2 * pc2
                end2 = mean + scale2 * pc2

                # Longueurs gauche / droite de la composante secondaire
                length_left = np.linalg.norm(mean - start2)
                length_right = np.linalg.norm(end2 - mean)

                # Angles
                angle1 = np.degrees(np.arctan2(pc1[0], pc1[1]))
                angle2 = np.degrees(np.arctan2(pc2[0], pc2[1]))

                angle1 = (angle1 + 180) % 180
                angle2 = (angle2 + 180) % 180

                # Détermine quelle composante est la plus verticale
                vert_parallel_pc1 = abs(pc1[0])
                vert_parallel_pc2 = abs(pc2[0])

                vertical_color = (0, 0, 255)   # rouge
                other_color = (255, 0, 0)      # bleu
                vertical_thickness = 3
                other_thickness = 1

                if vert_parallel_pc1 > vert_parallel_pc2:
                    most_vertical = "principale"
                    color1 = vertical_color
                    thickness1 = vertical_thickness
                    color2 = other_color
                    thickness2 = other_thickness
                    vertical_angle = angle1
                    horizontal_angle = angle2
                    vertical_length = scale1
                    horizontal_length = scale2
                else:
                    most_vertical = "secondaire"
                    color1 = other_color
                    thickness1 = other_thickness
                    color2 = vertical_color
                    thickness2 = vertical_thickness
                    vertical_angle = angle2
                    horizontal_angle = angle1
                    vertical_length = scale2
                    horizontal_length = scale1

                # Critères T
                angle_diff = abs(vertical_angle - horizontal_angle)
                is_perpendicular = abs(angle_diff - 90) < 15
                is_vertical = abs(vertical_angle - 90) < 15
                is_horizontal = (
                    abs(horizontal_angle - 0) < 15
                    or abs(horizontal_angle - 180) < 15
                )

                # Position de la barre
                bar_balance_ratio = length_left / length_right if length_right > 1e-6 else 0
                if bar_balance_ratio < 0.7:
                    bar_position = "DROITE"
                elif bar_balance_ratio > 1.4:
                    bar_position = "GAUCHE"
                else:
                    bar_position = "CENTRE"

                is_T = (
                    is_perpendicular
                    and is_vertical
                    and is_horizontal
                    and (horizontal_length > MIN_HORIZONTAL_LENGTH)
                )

            # ==============================
            # AFFICHAGE
            # ==============================
            frame_disp = frame.copy()

            # Overlay ROI = mask en gris + squelette en vert
            overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay[skeleton == 255] = (0, 255, 0)
            frame_disp[roi_start:] = overlay

            roi_disp = frame_disp[roi_start:]

            # Dessin PCA sur l'overlay
            if start1 is not None and end1 is not None and start2 is not None and end2 is not None:
                cv2.line(
                    roi_disp,
                    (int(start1[1]), int(start1[0])),
                    (int(end1[1]), int(end1[0])),
                    color1,
                    thickness1,
                )
                cv2.line(
                    roi_disp,
                    (int(start2[1]), int(start2[0])),
                    (int(end2[1]), int(end2[0])),
                    color2,
                    thickness2,
                )

                # Centre PCA
                cv2.circle(roi_disp, (int(mean[1]), int(mean[0])), 4, (0, 255, 255), -1)

            # Texte debug
            cv2.putText(frame_disp, f"Longueur gauche: {length_left:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Longueur droite: {length_right:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Plus verticale: {most_vertical}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Angle vert: {vertical_angle:.1f} deg", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Angle horiz: {horizontal_angle:.1f} deg", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Long vert: {vertical_length:.1f}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Long horiz: {horizontal_length:.1f}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Barre: {bar_position}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"T detecte: {'Oui' if is_T else 'Non'}", (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if is_T else (0, 0, 255), 2)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("skeleton", skeleton)
            cv2.imshow("overlay", frame_disp)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()