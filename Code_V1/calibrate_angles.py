#!/usr/bin/env python3
"""
Angle calibration tool for line classification.

Shows the actual angles (in degrees) of detected front, left, and right wall lines
so you can fine-tune the angle range thresholds.

Current ranges:
  - Front (horizontal): 80 <= deg <= 100
  - Left: 30 < deg < 80
  - Right: 100 < deg < 150

Press 'q' to quit.
"""

import cv2
from obstacles import CorridorNavigator


def main():
    navigator = CorridorNavigator(
        proto=True,
        video="movement.avi",
        debug_angles=True,
    )

    print("\n" + "="*60)
    print("ANGLE CALIBRATION TOOL")
    print("="*60)
    print("\nShowing detected angles for:\n")
    print(f"  Front (yellow):  80 <= deg <= 100")
    print(f"  Left (red):       30 <  deg <  80")
    print(f"  Right (green):   100 <  deg < 150")
    print("\nAdjust angle ranges in classify_lines() based on what you see.")
    print("Press 'q' to quit\n")
    print("="*60 + "\n")

    while True:
        ret, frame = navigator.read_frame()
        if not ret:
            break

        output = navigator.process_frame(frame, update=False, draw=True)
        cv2.imshow("Detected Line Angles", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    navigator.release()
    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
