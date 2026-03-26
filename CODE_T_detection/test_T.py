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
    cam = Camera()

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        mask, roi_start = compute_mask(frame)

        # Collect points for PCA
        points = np.column_stack(np.where(mask == 255))
        if len(points) > 0:
            # Center the data
            mean = np.mean(points, axis=0)
            centered = points - mean
            
            # Covariance matrix
            cov = np.cov(centered.T)
            
            # Eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Sort by eigenvalue descending
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]
            eigvals = eigvals[idx]
            
            # Principal components
            pc1 = eigvecs[:, 0]
            pc2 = eigvecs[:, 1]
            
            # Scale for line length (e.g., 2 standard deviations)
            scale1 = 2 * np.sqrt(eigvals[0])
            scale2 = 2 * np.sqrt(eigvals[1])
            
            # Line endpoints for PC1
            start1 = mean - scale1 * pc1
            end1 = mean + scale1 * pc1
            
            # Line endpoints for PC2
            start2 = mean - scale2 * pc2
            end2 = mean + scale2 * pc2
            
            # Calcul des longueurs de la ligne secondaire à gauche et droite de la principale
            # La ligne secondaire passe par mean, donc gauche: start2 à mean, droite: mean à end2
            length_left = np.linalg.norm(mean - start2)
            length_right = np.linalg.norm(end2 - mean)
            
            # Déterminer quelle composante est la plus parallèle à la verticale (axe y)
            vert_parallel_pc1 = abs(pc1[0])  # composante y
            vert_parallel_pc2 = abs(pc2[0])
            vertical_color = (0, 0, 255)  # rouge
            other_color = (255, 0, 0)  # bleu
            vertical_thickness = 3
            other_thickness = 1
            if vert_parallel_pc1 > vert_parallel_pc2:
                most_vertical = "principale"
                color1 = vertical_color
                thickness1 = vertical_thickness
                color2 = other_color
                thickness2 = other_thickness
            else:
                most_vertical = "secondaire"
                color1 = other_color
                thickness1 = other_thickness
                color2 = vertical_color
                thickness2 = vertical_thickness

        # display
        frame_disp = frame.copy()
        frame_disp[roi_start:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Dessiner les lignes PCA sur l'overlay
        roi_disp = frame_disp[roi_start:]
        if len(points) > 0:
            cv2.line(roi_disp, (int(start1[1]), int(start1[0])), (int(end1[1]), int(end1[0])), color1, thickness1)
            cv2.line(roi_disp, (int(start2[1]), int(start2[0])), (int(end2[1]), int(end2[0])), color2, thickness2)
            
            # Afficher les longueurs et la composante verticale
            cv2.putText(frame_disp, f"Longueur gauche: {length_left:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Longueur droite: {length_right:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"Plus verticale: {most_vertical}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("overlay", frame_disp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()