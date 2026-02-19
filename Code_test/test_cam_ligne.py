import cv2
import numpy as np

CAM_ID = 0
W, H = 640, 480
ROI_START = int(H * 0.6)  # bas de l'image

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not cap.isOpened():
    raise RuntimeError("Caméra non détectée")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (W, H))

    # 1) ROI
    roi = frame[ROI_START:H, :]

    # 2) Preprocess + seuil
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu (auto). Inversion pour ligne noire sur sol clair.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = 255 - mask  # retire si ta ligne est blanche sur fond noir

    # Nettoyage
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3) Centroïde (moments)
    M = cv2.moments(mask)
    found = M["m00"] > 1e-3

    vis = frame.copy()
    cv2.rectangle(vis, (0, ROI_START), (W - 1, H - 1), (0, 255, 255), 2)

    if found:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # cx, cy sont dans la ROI -> coordonnées globales pour afficher
        gx, gy = cx, cy + ROI_START

        # erreur normalisée [-1, 1]
        e = (cx - (W / 2)) / (W / 2)

        cv2.circle(vis, (gx, gy), 6, (0, 0, 255), -1)
        cv2.line(vis, (W // 2, H - 1), (gx, gy), (255, 0, 0), 2)
        cv2.putText(vis, f"cx={cx}  e={e:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "LINE NOT FOUND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("frame", vis)
    cv2.imshow("mask (ROI)", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
