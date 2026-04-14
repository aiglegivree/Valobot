from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera import Camera
from config import ROI_START_FRAC

try:
    from leds import LED
except Exception:
    LED = None


CONFIG_PATH = Path(__file__).with_name("config.py")
CAPTURE_ROOT = Path(__file__).with_name("roi_hsv_samples")
WINDOW_NAME = "ROI HSV Capture"

CAPTURE_COUNT = 20
CAPTURE_DURATION_S = 20.0
LOW_PERCENTILE = 5.0
HIGH_PERCENTILE = 95.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture 20 ROI images in 20 seconds and update HSV thresholds in config.py."
    )
    parser.add_argument(
        "--target",
        choices=("brown", "black"),
        default="brown",
        help="Which HSV thresholds to update in config.py.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=CAPTURE_COUNT,
        help="Number of ROI images to capture.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=CAPTURE_DURATION_S,
        help="Total capture duration in seconds.",
    )
    parser.add_argument(
        "--leds",
        action="store_true",
        default=True,
        help="Turn all LEDs on during capture.",
    )
    parser.add_argument(
        "--no-leds",
        dest="leds",
        action="store_false",
        help="Keep LEDs off during capture.",
    )
    return parser.parse_args()


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


def save_hsv_to_config(target: str, low: np.ndarray, high: np.ndarray):
    prefix = target.upper()
    text = CONFIG_PATH.read_text(encoding="utf-8")
    updates = {
        f"{prefix}_H_LOW": str(int(low[0])),
        f"{prefix}_S_LOW": str(int(low[1])),
        f"{prefix}_V_LOW": str(int(low[2])),
        f"{prefix}_H_HIGH": str(int(high[0])),
        f"{prefix}_S_HIGH": str(int(high[1])),
        f"{prefix}_V_HIGH": str(int(high[2])),
    }

    for key, value in updates.items():
        text = replace_config_value(text, key, value)

    CONFIG_PATH.write_text(text, encoding="utf-8")


def compute_hsv_bounds(rois: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hsv_pixels = []

    for roi in rois:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_pixels.append(hsv.reshape(-1, 3))

    all_pixels = np.concatenate(hsv_pixels, axis=0)
    low = np.percentile(all_pixels, LOW_PERCENTILE, axis=0)
    high = np.percentile(all_pixels, HIGH_PERCENTILE, axis=0)

    low = np.clip(np.floor(low), [0, 0, 0], [180, 255, 255]).astype(np.uint8)
    high = np.clip(np.ceil(high), [0, 0, 0], [180, 255, 255]).astype(np.uint8)
    return low, high


def make_output_dir(target: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = CAPTURE_ROOT / f"{target}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_summary(out_dir: Path, target: str, low: np.ndarray, high: np.ndarray):
    summary = (
        f"target={target}\n"
        f"H_LOW={int(low[0])}\n"
        f"S_LOW={int(low[1])}\n"
        f"V_LOW={int(low[2])}\n"
        f"H_HIGH={int(high[0])}\n"
        f"S_HIGH={int(high[1])}\n"
        f"V_HIGH={int(high[2])}\n"
        f"low_percentile={LOW_PERCENTILE}\n"
        f"high_percentile={HIGH_PERCENTILE}\n"
    )
    (out_dir / "hsv_summary.txt").write_text(summary, encoding="utf-8")


def main():
    args = parse_args()
    capture_count = max(1, args.count)
    duration_s = max(1.0, args.duration)
    interval_s = duration_s / capture_count
    out_dir = make_output_dir(args.target)

    leds = None
    if args.leds and LED is not None:
        try:
            leds = LED()
            leds.turn_on_all()
        except Exception:
            leds = None

    cam = Camera()
    rois: list[np.ndarray] = []
    capture_index = 0
    start_time = time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while capture_index < capture_count:
            ret, frame = cam.read()
            if not ret:
                continue

            h_frame = frame.shape[0]
            roi_start = int(h_frame * ROI_START_FRAC)
            roi = frame[roi_start:].copy()

            now = time.time()
            elapsed = now - start_time
            next_capture_time = capture_index * interval_s

            preview = frame.copy()
            cv2.rectangle(
                preview,
                (0, roi_start),
                (preview.shape[1] - 1, preview.shape[0] - 1),
                (0, 255, 255),
                2,
            )

            remaining = max(0.0, duration_s - elapsed)
            cv2.putText(
                preview,
                f"Target: {args.target}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"Capture {capture_index}/{capture_count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"Time left: {remaining:.1f}s",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                "q to quit early",
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if elapsed >= next_capture_time:
                capture_index += 1
                rois.append(roi)
                roi_path = out_dir / f"roi_{capture_index:02d}.png"
                cv2.imwrite(str(roi_path), roi)
                cv2.putText(
                    preview,
                    f"Saved {roi_path.name}",
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if not rois:
            raise RuntimeError("No ROI images were captured.")

        low, high = compute_hsv_bounds(rois)
        save_hsv_to_config(args.target, low, high)
        save_summary(out_dir, args.target, low, high)

        result = np.zeros((180, 640, 3), dtype=np.uint8)
        cv2.putText(
            result,
            f"{args.target.upper()} HSV saved to config.py",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result,
            f"LOW  = ({int(low[0])}, {int(low[1])}, {int(low[2])})",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result,
            f"HIGH = ({int(high[0])}, {int(high[1])}, {int(high[2])})",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, result)
        cv2.waitKey(3000)

    finally:
        try:
            cam.release()
        except Exception:
            pass

        if leds is not None:
            try:
                leds.turn_off_all()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
