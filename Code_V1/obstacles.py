import cv2
import numpy as np
from camera import Camera
import math

# Function to fit a line through segment endpoints
def fit_one_line(segments):
	if len(segments) == 0:
		return None
	points = []
	for x1, y1, x2, y2 in segments:
		points.append([x1, y1])
		points.append([x2, y2])
	points = np.array(points)

	# Fit line
	vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

	height, width = original.shape[:2]

	# Extend line to image borders
	if abs(vx) < 1e-5:
		# vertical line
		x1, y1 = int(x0), 0
		x2, y2 = int(x0), height
	else:
		# y = vy/vx * (x - x0) + y0
		y1 = int((-x0 * vy / vx) + y0)
		y2 = int(((width - x0) * vy / vx) + y0)
		x1, x2 = 0, width

	return (x1, y1, x2, y2)

picam = Camera()

while True:
	ret, frame = picam.read()
	original = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	height, width = gray.shape
	_, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	
	blur = cv2.GaussianBlur(mask, (7,7), 0)
	edges = cv2.Canny(blur, 115, 150, apertureSize = 3)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
	edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	
	lines = cv2.HoughLinesP(
		edges_closed,
		rho=1,
		theta=np.pi/180,
		threshold=20,
		minLineLength = 30,
		maxLineGap = 30
	)
	left_segments = []
	right_segments = []
	front_segments = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Ignore very short segments
            if np.hypot(x2-x1, y2-y1) < 20:
                continue

            delta_x = x2 - x1
            delta_y = y1 - y2  # careful with y-axis in images
            angle = math.degrees(math.atan2(delta_y, delta_x))

            # Classify by angle
            if 10 < angle < 70:
                left_segments.append([x1, y1, x2, y2])
            elif -70 < angle < -10:
                right_segments.append([x1, y1, x2, y2])
            elif -10 <= angle <= 10:
                front_segments.append([x1, y1, x2, y2])

    # Draw fitted lines
    left_line = fit_one_line(left_segments)
    right_line = fit_one_line(right_segments)
    front_line = fit_one_line(front_segments)

    if left_line:
        cv2.line(original, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0,0,255), 3)
    if right_line:
        cv2.line(original, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0,255,0), 3)
    if front_line:
        cv2.line(original, (front_line[0], front_line[1]), (front_line[2], front_line[3]), (255,0,0), 3)
			
			
			
			
			#cv2.line(original, (x1, y1), (x2, y2), (0,255,0), 2)
	
	cv2.imshow("Original frame", original)
	cv2.imshow("edges frame", edges_closed)

	cv2.waitKey(1)
