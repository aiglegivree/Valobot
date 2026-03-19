import cv2
import numpy as np
import time
from camera import Camera

picam = Camera()

# Get frame size (grab one frame first)
frame = picam.get_frame()
height, width, _ = frame.shape

# Define video writer (adjust codec if needed)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

start_time = time.time()

while True:
    frame = picam.get_frame()

    # Write frame to video
    out.write(frame)

    # Optional: show preview
    cv2.imshow('Recording', frame)

    # Stop after 20 seconds
    if time.time() - start_time > 20:
        break

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('DONE')

# Cleanup
out.release()
cv2.destroyAllWindows()