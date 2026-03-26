from camera import Camera
from t_detector import TDetector, draw_t_debug
import cv2
from leds import LED

cam = Camera()
detector = TDetector()
leds= LED()
leds.turn_on_all()
try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        res = detector.detect(frame)

        if res.t_valid_now:
            print("candidate:", res.score, res.debug.get("best_candidate"))
        if res.t_detected:
            print("T detected")

        dbg = draw_t_debug(frame, res)
        cv2.imshow("T detection", dbg)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
    leds.turn_off_all()