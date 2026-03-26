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

        # Calcul des moyennes globales pour référence
        y_coords, x_coords = np.where(mask == 255)
        if len(x_coords) > 0:
            avg_x = np.mean(x_coords)
        else:
            avg_x = W / 2  # valeur par défaut

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

        # Calcul de l'intersection des deux lignes
        x_int = avg_x  # valeur par défaut
        y_int = H_mask / 2
        if len(column_means) >= 2 and len(row_means) >= 2:
            # Ajustement linéaire pour la ligne colonne: y = a * x + b_col
            x_col = [p[0] for p in column_means]
            y_col = [p[1] for p in column_means]
            a, b_col = np.polyfit(x_col, y_col, 1)
            
            # Ajustement linéaire pour la ligne ligne: x = m * y + b
            x_row = [p[0] for p in row_means]
            y_row = [p[1] for p in row_means]
            m, b = np.polyfit(y_row, x_row, 1)
            
            # Résoudre pour l'intersection
            denom = 1 - a * m
            if abs(denom) > 1e-6:
                y_int = (a * b + b_col) / denom
                x_int = m * y_int + b
                # Dessiner le point d'intersection
                cv2.circle(roi_disp, (int(x_int), int(y_int)), 5, (0, 0, 255), -1)

        # Séparer les points de la ligne horizontale en gauche et droite du point d'intersection
        left_points = [p for p in row_means if p[0] < x_int]
        right_points = [p for p in row_means if p[0] > x_int]
        length_left = calc_length(left_points)
        length_right = calc_length(right_points)

        # display
        frame_disp = frame.copy()
        frame_disp[roi_start:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Dessiner les lignes sur l'overlay
        roi_disp = frame_disp[roi_start:]
        if column_means:
            cv2.polylines(roi_disp, [np.array(column_means, dtype=np.int32)], False, (0, 255, 0), 2)
        if row_means:
            cv2.polylines(roi_disp, [np.array(row_means, dtype=np.int32)], False, (255, 0, 0), 2)
        
        # Afficher les longueurs
        cv2.putText(frame_disp, f"Longueur gauche: {length_left:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_disp, f"Longueur droite: {length_right:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calcul du produit scalaire des deux lignes
        if len(column_means) >= 2 and len(row_means) >= 2:
            vec_col = (W_mask - 1, column_means[-1][1] - column_means[0][1])
            vec_row = (row_means[-1][0] - row_means[0][0], H_mask - 1)
            dot_product = vec_col[0] * vec_row[0] + vec_col[1] * vec_row[1]
            cv2.putText(frame_disp, f"Produit scalaire: {dot_product:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame_disp, "Produit scalaire: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("overlay", frame_disp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()