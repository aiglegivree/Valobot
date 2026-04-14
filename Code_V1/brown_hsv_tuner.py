from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from camera import Camera
from leds import LED
from config import ROI_START_FRAC, W, H


WINDOW_CONTROLS = "Brown HSV Controls"
WINDOW_CAMERA = "Camera"
WINDOW_ROI = "ROI"
WINDOW_MASK = "Brown Mask"
WINDOW_RESULT = "Brown Result"
SAVE_PATH = Path(__file__).with_name("brown_hsv_values.txt")


def nothing(_):
    pass


def odd_at_least_one(value: int) -> int:
    value = max(1, value)
    if value % 2 == 0:
        value += 1
    return value


def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 460, 360)

    cv2.createTrackbar("H low", WINDOW_CONTROLS, 5, 179, nothing)
    cv2.createTrackbar("S low", WINDOW_CONTROLS, 80, 255, nothing)
    cv2.createTrackbar("V low", WINDOW_CONTROLS, 40, 255, nothing)
    cv2.createTrackbar("H high", WINDOW_CONTROLS, 30, 179, nothing)
    cv2.createTrackbar("S high", WINDOW_CONTROLS, 255, 255, nothing)
    cv2.createTrackbar("V high", WINDOW_CONTROLS, 200, 255, nothing)
    cv2.createTrackbar("Blur", WINDOW_CONTROLS, 3, 15, nothing)
    cv2.createTrackbar("Open kernel", WINDOW_CONTROLS, 3, 15, nothing)
    cv2.createTrackbar("Open iter", WINDOW_CONTROLS, 0, 10, nothing)
    cv2.createTrackbar("Close kernel", WINDOW_CONTROLS, 3, 15, nothing)
    cv2.createTrackbar("Close iter", WINDOW_CONTROLS, 0, 10, nothing)


def get_values():
    h_low = cv2.getTrackbarPos("H low", WINDOW_CONTROLS)
    s_low = cv2.getTrackbarPos("S low", WINDOW_CONTROLS)
    v_low = cv2.getTrackbarPos("V low", WINDOW_CONTROLS)
    h_high = cv2.getTrackbarPos("H high", WINDOW_CONTROLS)
    s_high = cv2.getTrackbarPos("S high", WINDOW_CONTROLS)
    v_high = cv2.getTrackbarPos("V high", WINDOW_CONTROLS)
    blur = odd_at_least_one(cv2.getTrackbarPos("Blur", WINDOW_CONTROLS))
    open_kernel = odd_at_least_one(cv2.getTrackbarPos("Open kernel", WINDOW_CONTROLS))
    open_iter = cv2.getTrackbarPos("Open iter", WINDOW_CONTROLS)
    close_kernel = odd_at_least_one(cv2.getTrackbarPos("Close kernel", WINDOW_CONTROLS))
    close_iter = cv2.getTrackbarPos("Close iter", WINDOW_CONTROLS)

    return {
        "h_low": min(h_low, h_high),
        "s_low": min(s_low, s_high),
        "v_low": min(v_low, v_high),
        "h_high": max(h_low, h_high),
        "s_high": max(s_low, s_high),
        "v_high": max(v_low, v_high),
        "blur": blur,
        "open_kernel": open_kernel,
        "open_iter": open_iter,
        "close_kernel": close_kernel,
        "close_iter": close_iter,
    }


def save_values(values: dict):
    text = (
        f"BROWN_H_LOW = {values['h_low']}\n"
        f"BROWN_S_LOW = {values['s_low']}\n"
        f"BROWN_V_LOW = {values['v_low']}\n"
        f"BROWN_H_HIGH = {values['h_high']}\n"
        f"BROWN_S_HIGH = {values['s_high']}\n"
        f"BROWN_V_HIGH = {values['v_high']}\n"
        f"BLUR = {values['blur']}\n"
        f"OPEN_KERNEL = {values['open_kernel']}\n"
        f"OPEN_ITER = {values['open_iter']}\n"
        f"CLOSE_KERNEL = {values['close_kernel']}\n"
        f"CLOSE_ITER = {values['close_iter']}\n"
    )
    SAVE_PATH.write_text(text, encoding="utf-8")


def build_brown_mask(frame: np.ndarray, values: dict) -> tuple[np.ndarray, np.ndarray, int]:
    roi_start = int(H * ROI_START_FRAC)
    roi = frame[roi_start:, :]

    roi_blur = cv2.GaussianBlur(roi, (values["blur"], values["blur"]), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    low = np.array([values["h_low"], values["s_low"], values["v_low"]], dtype=np.uint8)
    high = np.array([values["h_high"], values["s_high"], values["v_high"]], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)

    if values["open_iter"] > 0:
        open_kernel = np.ones((values["open_kernel"], values["open_kernel"]), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=values["open_iter"])
    if values["close_iter"] > 0:
        close_kernel = np.ones((values["close_kernel"], values["close_kernel"]), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=values["close_iter"])

    result = cv2.bitwise_and(roi, roi, mask=mask)
    return roi, mask, result


def main():
    leds = LED()
    leds.turn_on_all()
    leds_on = True
    cam = Camera()

    create_trackbars()

    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_ROI, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_RESULT, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            values = get_values()
            roi, mask, result = build_brown_mask(frame, values)

            camera_view = frame.copy()
            roi_start = int(H * ROI_START_FRAC)
            cv2.rectangle(
                camera_view,
                (0, roi_start),
                (W - 1, H - 1),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                camera_view,
                f"q quit | s save | l LEDs {'ON' if leds_on else 'OFF'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                camera_view,
                f"H:[{values['h_low']},{values['h_high']}] "
                f"S:[{values['s_low']},{values['s_high']}] "
                f"V:[{values['v_low']},{values['v_high']}]",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_CAMERA, camera_view)
            cv2.imshow(WINDOW_ROI, roi)
            cv2.imshow(WINDOW_MASK, mask)
            cv2.imshow(WINDOW_RESULT, result)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("s"):
                save_values(values)
            if key == ord("l"):
                leds_on = not leds_on
                if leds_on:
                    leds.turn_on_all()
                else:
                    leds.turn_off_all()

    finally:
        try:
            leds.turn_off_all()
        except Exception:
            pass

        try:
            cam.release()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
