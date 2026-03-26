#!/usr/bin/env python3
"""
Live LAB tuner for Raspberry Pi camera feed using Picamera2 + OpenCV.
 
Controls:
  q      Quit
  s      Save current LAB values to lab_values.txt
"""
 
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from leds import LED
 
 
WINDOW_CONTROLS = "LAB Controls"
WINDOW_PREVIEW  = "Camera Preview"
WINDOW_MASK     = "Mask"
WINDOW_RESULT   = "Segmented Result"
 
leds = LED()
leds.turn_off_all()
 
 
def nothing(_):
    pass
 
 
def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 420, 250)
 
    # OpenCV LAB ranges for 8-bit images:
    # L: 0-255  (lightness)
    # A: 0-255  (green=0 .. red=255, neutral=128)
    # B: 0-255  (blue=0  .. yellow=255, neutral=128)
    cv2.createTrackbar("L Min", WINDOW_CONTROLS,   0, 255, nothing)
    cv2.createTrackbar("L Max", WINDOW_CONTROLS, 255, 255, nothing)
    cv2.createTrackbar("A Min", WINDOW_CONTROLS,   0, 255, nothing)
    cv2.createTrackbar("A Max", WINDOW_CONTROLS, 255, 255, nothing)
    cv2.createTrackbar("B Min", WINDOW_CONTROLS,   0, 255, nothing)
    cv2.createTrackbar("B Max", WINDOW_CONTROLS, 255, 255, nothing)
 
    cv2.createTrackbar("Blur",   WINDOW_CONTROLS, 0, 20, nothing)
    cv2.createTrackbar("Erode",  WINDOW_CONTROLS, 0, 10, nothing)
    cv2.createTrackbar("Dilate", WINDOW_CONTROLS, 0, 10, nothing)
 
 
def get_trackbar_values():
    l_min = cv2.getTrackbarPos("L Min", WINDOW_CONTROLS)
    l_max = cv2.getTrackbarPos("L Max", WINDOW_CONTROLS)
    a_min = cv2.getTrackbarPos("A Min", WINDOW_CONTROLS)
    a_max = cv2.getTrackbarPos("A Max", WINDOW_CONTROLS)
    b_min = cv2.getTrackbarPos("B Min", WINDOW_CONTROLS)
    b_max = cv2.getTrackbarPos("B Max", WINDOW_CONTROLS)
    blur        = cv2.getTrackbarPos("Blur",   WINDOW_CONTROLS)
    erode_iter  = cv2.getTrackbarPos("Erode",  WINDOW_CONTROLS)
    dilate_iter = cv2.getTrackbarPos("Dilate", WINDOW_CONTROLS)
 
    lower = np.array([l_min, a_min, b_min], dtype=np.uint8)
    upper = np.array([l_max, a_max, b_max], dtype=np.uint8)
 
    return lower, upper, blur, erode_iter, dilate_iter
 
 
def save_lab(lower, upper, blur, erode_iter, dilate_iter, path="lab_values.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"LOWER_LAB = {lower.tolist()}\n")
        f.write(f"UPPER_LAB = {upper.tolist()}\n")
        f.write(f"BLUR   = {blur}\n")
        f.write(f"ERODE  = {erode_iter}\n")
        f.write(f"DILATE = {dilate_iter}\n")
    print(f"Saved LAB settings to {path}")
 
 
def main():
    picam2 = Picamera2()
 
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
 
    time.sleep(1.0)
 
    create_trackbars()
 
    cv2.namedWindow(WINDOW_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_RESULT,  cv2.WINDOW_NORMAL)
 
    last_print = None
 
    try:
        while True:
            frame_rgb = picam2.capture_array()
 
            frame_to_convert = frame_rgb
            if cv2.getTrackbarPos("Blur", WINDOW_CONTROLS) > 0:
                blur = cv2.getTrackbarPos("Blur", WINDOW_CONTROLS)
                k = blur * 2 + 1
                frame_to_convert = cv2.GaussianBlur(frame_rgb, (k, k), 0)
 
            lab = cv2.cvtColor(frame_to_convert, cv2.COLOR_RGB2LAB)
 
            lower, upper, blur, erode_iter, dilate_iter = get_trackbar_values()
 
            mask = cv2.inRange(lab, lower, upper)
 
            if erode_iter > 0:
                mask = cv2.erode(mask, None, iterations=erode_iter)
            if dilate_iter > 0:
                mask = cv2.dilate(mask, None, iterations=dilate_iter)
 
            result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
 
            preview_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result_bgr  = cv2.cvtColor(result,    cv2.COLOR_RGB2BGR)
 
            cv2.imshow(WINDOW_PREVIEW, preview_bgr)
            cv2.imshow(WINDOW_MASK,    mask)
            cv2.imshow(WINDOW_RESULT,  result_bgr)
 
            current_values = (tuple(lower.tolist()), tuple(upper.tolist()), blur, erode_iter, dilate_iter)
            if current_values != last_print:
                print(
                    f"Lower LAB: {lower.tolist()} | Upper LAB: {upper.tolist()} | "
                    f"Blur: {blur} | Erode: {erode_iter} | Dilate: {dilate_iter}"
                )
                last_print = current_values
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_lab(lower, upper, blur, erode_iter, dilate_iter)
 
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
 
 
if __name__ == "__main__":
    main()
 