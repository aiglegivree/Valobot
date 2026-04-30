from __future__ import annotations

from pathlib import Path


CONFIG_PATH = Path(__file__)

DEFAULT_OBSTACLE_SETTINGS = {
    "canny_low": 75,
    "canny_high": 90,
    "contour_min_length": 30,
    "hough_threshold": 60,
    "left_angle_low": 30,
    "left_angle_high": 75,
    "right_angle_low": 105,
    "right_angle_high": 150,
    "infront_angle_low": 85,
    "infront_angle_high": 100,
}

FILE_TEMPLATE = 'from __future__ import annotations\n\nfrom pathlib import Path\n\n\nCONFIG_PATH = Path(__file__)\n\nDEFAULT_OBSTACLE_SETTINGS = {{\n    "canny_low": {canny_low},\n    "canny_high": {canny_high},\n    "contour_min_length": {contour_min_length},\n    "hough_threshold": {hough_threshold},\n    "left_angle_low": {left_angle_low},\n    "left_angle_high": {left_angle_high},\n    "right_angle_low": {right_angle_low},\n    "right_angle_high": {right_angle_high},\n    "infront_angle_low": {infront_angle_low},\n    "infront_angle_high": {infront_angle_high},\n}}\n\nFILE_TEMPLATE = {file_template!r}\n\n\ndef load_obstacle_settings() -> dict[str, int]:\n    return DEFAULT_OBSTACLE_SETTINGS.copy()\n\n\ndef _normalized_settings(settings: dict[str, int]) -> dict[str, int]:\n    saved = DEFAULT_OBSTACLE_SETTINGS.copy()\n    for key in saved:\n        value = settings.get(key)\n        if isinstance(value, int):\n            saved[key] = value\n\n    if saved["canny_high"] < saved["canny_low"]:\n        saved["canny_high"] = saved["canny_low"]\n\n    for prefix in ("left", "right", "infront"):\n        low_key = f"{{prefix}}_angle_low"\n        high_key = f"{{prefix}}_angle_high"\n        if saved[high_key] < saved[low_key]:\n            saved[high_key] = saved[low_key]\n\n    return saved\n\n\ndef save_obstacle_settings(settings: dict[str, int]) -> None:\n    saved = _normalized_settings(settings)\n    CONFIG_PATH.write_text(FILE_TEMPLATE.format(file_template=FILE_TEMPLATE, **saved), encoding="utf-8")\n'


def load_obstacle_settings() -> dict[str, int]:
    return DEFAULT_OBSTACLE_SETTINGS.copy()


def _normalized_settings(settings: dict[str, int]) -> dict[str, int]:
    saved = DEFAULT_OBSTACLE_SETTINGS.copy()
    for key in saved:
        value = settings.get(key)
        if isinstance(value, int):
            saved[key] = value

    if saved["canny_high"] < saved["canny_low"]:
        saved["canny_high"] = saved["canny_low"]

    for prefix in ("left", "right", "infront"):
        low_key = f"{prefix}_angle_low"
        high_key = f"{prefix}_angle_high"
        if saved[high_key] < saved[low_key]:
            saved[high_key] = saved[low_key]

    return saved


def save_obstacle_settings(settings: dict[str, int]) -> None:
    saved = _normalized_settings(settings)
    CONFIG_PATH.write_text(FILE_TEMPLATE.format(file_template=FILE_TEMPLATE, **saved), encoding="utf-8")
