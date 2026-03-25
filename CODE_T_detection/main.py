from camera import Camera
from t_detector import TDetector, draw_t_debug
from leds import LED
import cv2

cam = Camera()
detector = TDetector()
led=LED()
led.turn_on_all()
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
		cv2.imshow("Mask",res.mask)
		
		if cv2.waitKey(1) & 0xFF == 27:
			break

finally:
    cam.release()
    cv2.destroyAllWindows()
