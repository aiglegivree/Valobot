#!/usr/bin/env python3
"""
Live HSV tuner for Raspberry Pi camera feed using Picamera2 + OpenCV.

Controls:
  q      Quit
  s      Save current HSV values to hsv_values.txt
"""

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from leds import LED


WINDOW_CONTROLS = "HSV Controls"
WINDOW_PREVIEW = "Camera Preview"
WINDOW_MASK = "Mask"
WINDOW_RESULT = "Segmented Result"
leds = LED()
leds.turn_off_all()

def nothing(_):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 420, 300)

    # OpenCV HSV ranges for 8-bit images:
    # H: 0-179, S: 0-255, V: 0-255
    cv2.createTrackbar("H Min", WINDOW_CONTROLS, 0, 179, nothing)
    cv2.createTrackbar("H Max", WINDOW_CONTROLS, 179, 179, nothing)
    cv2.createTrackbar("S Min", WINDOW_CONTROLS, 0, 255, nothing)
    cv2.createTrackbar("S Max", WINDOW_CONTROLS, 255, 255, nothing)
    cv2.createTrackbar("V Min", WINDOW_CONTROLS, 0, 255, nothing)
    cv2.createTrackbar("V Max", WINDOW_CONTROLS, 255, 255, nothing)

    # Optional cleanup controls
    cv2.createTrackbar("Blur", WINDOW_CONTROLS, 0, 20, nothing)   # odd kernel size applied later
    cv2.createTrackbar("Erode", WINDOW_CONTROLS, 0, 10, nothing)
    cv2.createTrackbar("Dilate", WINDOW_CONTROLS, 0, 10, nothing)


def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H Min", WINDOW_CONTROLS)
    h_max = cv2.getTrackbarPos("H Max", WINDOW_CONTROLS)
    s_min = cv2.getTrackbarPos("S Min", WINDOW_CONTROLS)
    s_max = cv2.getTrackbarPos("S Max", WINDOW_CONTROLS)
    v_min = cv2.getTrackbarPos("V Min", WINDOW_CONTROLS)
    v_max = cv2.getTrackbarPos("V Max", WINDOW_CONTROLS)
    blur = cv2.getTrackbarPos("Blur", WINDOW_CONTROLS)
    erode_iter = cv2.getTrackbarPos("Erode", WINDOW_CONTROLS)
    dilate_iter = cv2.getTrackbarPos("Dilate", WINDOW_CONTROLS)

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

    return lower, upper, blur, erode_iter, dilate_iter


def save_hsv(lower, upper, blur, erode_iter, dilate_iter, path="hsv_values.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"LOWER_HSV = {lower.tolist()}\n")
        f.write(f"UPPER_HSV = {upper.tolist()}\n")
        f.write(f"BLUR = {blur}\n")
        f.write(f"ERODE = {erode_iter}\n")
        f.write(f"DILATE = {dilate_iter}\n")
    print(f"Saved HSV settings to {path}")


def main():
    picam2 = Picamera2()

    # Configure a preview-sized stream for responsive tuning.
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Let auto-exposure/auto-white-balance settle briefly.
    time.sleep(1.0)

    create_trackbars()

    cv2.namedWindow(WINDOW_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_RESULT, cv2.WINDOW_NORMAL)

    last_print = None

    try:
        while True:
            # Picamera2 with RGB888 returns an RGB image array.
            frame_rgb = picam2.capture_array()

            # Convert RGB -> HSV
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

            lower, upper, blur, erode_iter, dilate_iter = get_trackbar_values()

            # Optional smoothing before thresholding
            if blur > 0:
                k = blur * 2 + 1  # ensures odd kernel size: 3,5,7,...
                frame_rgb_proc = cv2.GaussianBlur(frame_rgb, (k, k), 0)
                hsv = cv2.cvtColor(frame_rgb_proc, cv2.COLOR_RGB2HSV)

            # Threshold in HSV space
            mask = cv2.inRange(hsv, lower, upper)

            # Optional morphology to reduce noise
            if erode_iter > 0:
                mask = cv2.erode(mask, None, iterations=erode_iter)
            if dilate_iter > 0:
                mask = cv2.dilate(mask, None, iterations=dilate_iter)

            # Apply mask to original image
            result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)

            # OpenCV displays color images as BGR, so convert for correct display.
            preview_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            cv2.imshow(WINDOW_PREVIEW, preview_bgr)
            cv2.imshow(WINDOW_MASK, mask)
            cv2.imshow(WINDOW_RESULT, result_bgr)

            current_values = (tuple(lower.tolist()), tuple(upper.tolist()), blur, erode_iter, dilate_iter)
            if current_values != last_print:
                print(
                    f"Lower HSV: {lower.tolist()} | Upper HSV: {upper.tolist()} | "
                    f"Blur: {blur} | Erode: {erode_iter} | Dilate: {dilate_iter}"
                )
                last_print = current_values

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_hsv(lower, upper, blur, erode_iter, dilate_iter)

    finally:
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
