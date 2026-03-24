from camera import Camera
from t_detector import TDetector, draw_t_debug
import cv2

cam = Camera()
detector = TDetector()

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        res = detector.detect(frame)

        if res.t_detected:
            print("T detected")

        dbg = draw_t_debug(frame, res)
        cv2.imshow("T detection", dbg)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cam.release()
    cv2.destroyAllWindows()