from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from camera import Camera
from config import (
    BLUE_HIGH_CORRIDOR,
    BLUE_HIGH_SHOOTING,
    BLUE_LOW_CORRIDOR,
    BLUE_LOW_SHOOTING,
    MIN_TARGET_AREA_CORRIDOR,
    MIN_TARGET_AREA_SHOOTING,
    MIN_TARGET_WIDTH_CORRIDOR,
    MIN_TARGET_WIDTH_SHOOTING,
)
from leds import LED


WINDOW_CONTROLS = "Blue Controls"
WINDOW_CAMERA = "Blue Camera"
WINDOW_MASK = "Blue Mask"
WINDOW_DEBUG = "Blue Debug"
CONFIG_PATH = Path(__file__).with_name("config.py")
PROFILE_CORRIDOR = "CORRIDOR"
PROFILE_SHOOTING = "SHOOTING"


def nothing(_):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 460, 380)

    cv2.createTrackbar("H low", WINDOW_CONTROLS, 0, 180, nothing)
    cv2.createTrackbar("S low", WINDOW_CONTROLS, 0, 255, nothing)
    cv2.createTrackbar("V low", WINDOW_CONTROLS, 0, 255, nothing)
    cv2.createTrackbar("H high", WINDOW_CONTROLS, 0, 180, nothing)
    cv2.createTrackbar("S high", WINDOW_CONTROLS, 0, 255, nothing)
    cv2.createTrackbar("V high", WINDOW_CONTROLS, 0, 255, nothing)
    cv2.createTrackbar("Min area", WINDOW_CONTROLS, 0, 10000, nothing)
    cv2.createTrackbar("Min width", WINDOW_CONTROLS, 0, 320, nothing)


def get_profile_settings(profile_name: str) -> dict[str, object]:
    if profile_name == PROFILE_CORRIDOR:
        return {
            "blue_low": BLUE_LOW_CORRIDOR,
            "blue_high": BLUE_HIGH_CORRIDOR,
            "min_area": MIN_TARGET_AREA_CORRIDOR,
            "min_width": MIN_TARGET_WIDTH_CORRIDOR,
        }
    return {
        "blue_low": BLUE_LOW_SHOOTING,
        "blue_high": BLUE_HIGH_SHOOTING,
        "min_area": MIN_TARGET_AREA_SHOOTING,
        "min_width": MIN_TARGET_WIDTH_SHOOTING,
    }


def set_trackbar_values(values: dict[str, object]) -> None:
    cv2.setTrackbarPos("H low", WINDOW_CONTROLS, int(values["blue_low"][0]))
    cv2.setTrackbarPos("S low", WINDOW_CONTROLS, int(values["blue_low"][1]))
    cv2.setTrackbarPos("V low", WINDOW_CONTROLS, int(values["blue_low"][2]))
    cv2.setTrackbarPos("H high", WINDOW_CONTROLS, int(values["blue_high"][0]))
    cv2.setTrackbarPos("S high", WINDOW_CONTROLS, int(values["blue_high"][1]))
    cv2.setTrackbarPos("V high", WINDOW_CONTROLS, int(values["blue_high"][2]))
    cv2.setTrackbarPos("Min area", WINDOW_CONTROLS, min(int(values["min_area"]), 10000))
    cv2.setTrackbarPos("Min width", WINDOW_CONTROLS, min(int(values["min_width"]), 320))


def get_values() -> dict[str, object]:
    h_low = cv2.getTrackbarPos("H low", WINDOW_CONTROLS)
    s_low = cv2.getTrackbarPos("S low", WINDOW_CONTROLS)
    v_low = cv2.getTrackbarPos("V low", WINDOW_CONTROLS)
    h_high = cv2.getTrackbarPos("H high", WINDOW_CONTROLS)
    s_high = cv2.getTrackbarPos("S high", WINDOW_CONTROLS)
    v_high = cv2.getTrackbarPos("V high", WINDOW_CONTROLS)
    min_area = cv2.getTrackbarPos("Min area", WINDOW_CONTROLS)
    min_width = cv2.getTrackbarPos("Min width", WINDOW_CONTROLS)

    if h_high < h_low:
        h_high = h_low
    if s_high < s_low:
        s_high = s_low
    if v_high < v_low:
        v_high = v_low

    return {
        "blue_low": (h_low, s_low, v_low),
        "blue_high": (h_high, s_high, v_high),
        "min_area": min_area,
        "min_width": min_width,
    }


def detect_blue_targets(frame: np.ndarray, values: dict[str, object]):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(values["blue_low"], dtype=np.uint8),
        np.array(values["blue_high"], dtype=np.uint8),
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < int(values["min_area"]):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < int(values["min_width"]):
            continue
        targets.append(
            {
                "area": float(area),
                "bbox": (x, y, w, h),
                "cx": int(x + w // 2),
                "cy": int(y + h // 2),
            }
        )

    targets.sort(key=lambda target: target["area"], reverse=True)
    return mask, targets


def replace_config_value(text: str, key: str, value: str) -> str:
    marker = f"{key} = "
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(marker):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{key} = {value}"
            return "\n".join(lines) + "\n"
    raise ValueError(f"Key {key} not found in config.py")


def save_values_to_config(values: dict[str, object], profile_name: str):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    if profile_name == PROFILE_CORRIDOR:
        updates = {
            "BLUE_LOW_CORRIDOR": repr(tuple(int(v) for v in values["blue_low"])),
            "BLUE_HIGH_CORRIDOR": repr(tuple(int(v) for v in values["blue_high"])),
            "MIN_TARGET_AREA_CORRIDOR": str(int(values["min_area"])),
            "MIN_TARGET_WIDTH_CORRIDOR": str(int(values["min_width"])),
        }
    else:
        updates = {
            "BLUE_LOW_SHOOTING": repr(tuple(int(v) for v in values["blue_low"])),
            "BLUE_HIGH_SHOOTING": repr(tuple(int(v) for v in values["blue_high"])),
            "MIN_TARGET_AREA_SHOOTING": str(int(values["min_area"])),
            "MIN_TARGET_WIDTH_SHOOTING": str(int(values["min_width"])),
        }

    for key, value in updates.items():
        text = replace_config_value(text, key, value)

    CONFIG_PATH.write_text(text, encoding="utf-8")


def main():
    leds = LED()
    leds.turn_on_all()
    leds_on = True
    cam = Camera()

    create_trackbars()
    active_profile = PROFILE_SHOOTING
    set_trackbar_values(get_profile_settings(active_profile))
    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            values = get_values()
            mask, targets = detect_blue_targets(frame, values)

            camera_view = frame.copy()
            debug_view = frame.copy()
            for target in targets:
                x, y, w, h = target["bbox"]
                cv2.rectangle(debug_view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(debug_view, (target["cx"], target["cy"]), 4, (0, 0, 255), -1)

            cv2.putText(debug_view, f"Targets: {len(targets)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_view, f"PROFILE: {active_profile}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(debug_view, f"BLUE_LOW: {values['blue_low']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_view, f"BLUE_HIGH: {values['blue_high']}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_view, f"MIN_TARGET_AREA: {values['min_area']}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_view, f"MIN_TARGET_WIDTH: {values['min_width']}", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_view, "p switch profile | l toggle leds | s save config", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW_CAMERA, camera_view)
            cv2.imshow(WINDOW_MASK, mask)
            cv2.imshow(WINDOW_DEBUG, debug_view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("l"):
                leds_on = not leds_on
                if leds_on:
                    leds.turn_on_all()
                else:
                    leds.turn_off_all()
            if key == ord("p"):
                active_profile = PROFILE_CORRIDOR if active_profile == PROFILE_SHOOTING else PROFILE_SHOOTING
                set_trackbar_values(get_profile_settings(active_profile))
            if key == ord("s"):
                save_values_to_config(values, active_profile)
                print(f"Saved {active_profile.lower()} blue tuner settings to Code_fsm_v1/config.py")

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
