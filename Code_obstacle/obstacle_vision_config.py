from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CANNY_LOW = 75
DEFAULT_CANNY_HIGH = 128
DEFAULT_CONTOUR_MIN_LENGTH = 32
DEFAULT_HOUGH_THRESHOLD = 86

DEFAULT_OBSTACLE_SETTINGS = {
    "canny_low": DEFAULT_CANNY_LOW,
    "canny_high": DEFAULT_CANNY_HIGH,
    "contour_min_length": DEFAULT_CONTOUR_MIN_LENGTH,
    "hough_threshold": DEFAULT_HOUGH_THRESHOLD,
}

SETTINGS_PATH = Path(__file__).with_name("test_obs_settings.json")


def load_obstacle_settings() -> dict[str, int]:
    settings = DEFAULT_OBSTACLE_SETTINGS.copy()
    if not SETTINGS_PATH.exists():
        return settings

    try:
        loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings

    for key in settings:
        value = loaded.get(key)
        if isinstance(value, int):
            settings[key] = value
    return settings


def save_obstacle_settings(settings: dict[str, int]) -> None:
    saved = DEFAULT_OBSTACLE_SETTINGS.copy()
    for key in saved:
        value = settings.get(key)
        if isinstance(value, int):
            saved[key] = value
    SETTINGS_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
