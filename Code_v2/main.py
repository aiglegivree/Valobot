# main.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import time
import cv2
import numpy as np

from camera import Camera
from vision import detect_line, draw_debug_slices, draw_mask_slices
from controller import Controller
from motors import MotorDriver
from leds import LED
from config import (
    SHOW_DEBUG,
    BAND_THICKNESS,
    RECORD_RUN_VIDEO,
    RECORD_RUN_FPS,
    RECORDINGS_DIR,
)


def build_recording_frame(debug_frame: np.ndarray, mask_frame: np.ndarray) -> np.ndarray:
    mask_bgr = mask_frame
    if len(mask_bgr.shape) == 2:
        mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2BGR)

    target_height = debug_frame.shape[0]
    target_width = debug_frame.shape[1]
    resized_mask = cv2.resize(mask_bgr, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    combined = np.hstack((debug_frame, resized_mask))
    cv2.putText(
        combined,
        "Camera + ROI/debug",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        combined,
        "Mask",
        (target_width + 15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )
    return combined


def create_run_video_writer(frame_size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not RECORD_RUN_VIDEO:
        return None

    recordings_dir = Path(__file__).resolve().parent / RECORDINGS_DIR
    recordings_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = recordings_dir / f"run_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, RECORD_RUN_FPS, frame_size)

    if not writer.isOpened():
        return None

    print(f"Recording run video to: {output_path}")
    return writer


def main():
    cam = Camera()
    controller = Controller()
    motors = MotorDriver()
    leds = LED()
    video_writer: cv2.VideoWriter | None = None

    t_reached = False
    aura_mode = False
    prev_frame_time: float | None = None
    fps_display = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            now = time.perf_counter()
            if prev_frame_time is not None:
                dt = now - prev_frame_time
                if dt > 1e-6:
                    fps_display = 1.0 / dt
            prev_frame_time = now

            # 1) vision
            res = detect_line(frame, angle_deg=20)

            # 2) control
            left, right, speed, steer, mode = controller.update(res)

            # 3) behavior
            if mode == "T_STOP":
                if not t_reached:
                    t_reached = True
                    motors.stop()
                if not aura_mode:
                    aura_mode = True
                    leds.aura_farm_start()

                # keep aura farm running once T is reached
                leds.aura_farm_step()
	
            else:
                if aura_mode:
                    aura_mode = False
                    leds.aura_farm_stop()
                motors.set_speeds(left, right)
                leds.turn_on_all()

            # 4) debug
            dbg = draw_debug_slices(frame, res, BAND_THICKNESS)
            cv2.putText(
                dbg,
                f"FPS: {fps_display:.1f}",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 0),
                1,
            )
            mask_vis = draw_mask_slices(res)

            if RECORD_RUN_VIDEO:
                recording_frame = build_recording_frame(dbg, mask_vis)
                if video_writer is None:
                    frame_size = (recording_frame.shape[1], recording_frame.shape[0])
                    video_writer = create_run_video_writer(frame_size)
                if video_writer is not None:
                    video_writer.write(recording_frame)

            if SHOW_DEBUG:
                cv2.imshow("debug", dbg)
                cv2.imshow("mask", mask_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        motors.stop()

        try:
            leds.aura_farm_stop()
        except Exception:
            pass

        try:
            cam.release()
        except Exception:
            pass

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
