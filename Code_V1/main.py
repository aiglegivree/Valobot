from camera import Camera
from vision import detect_line, draw_debug_slices, draw_mask_slices
from controller import Controller
from motors import MotorDriver
import cv2
from config import BAND_THICKNESS

cam = Camera()
controller = Controller()
motors = MotorDriver()

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    # 1️ Detect line and compute top-down mask
    res = detect_line(frame, angle_deg=20)

    # 2️ Update controller and motors
    left, right, speed, steer, mode = controller.update(res)
    motors.set_speeds(left, right)

    # 3 Prepare top-down ROI for debug visualization
    # The mask is already top-down corrected; warp the frame ROI to match
    roi_start = res.roi_start
    frame_roi = frame[roi_start:]  # crop ROI
    h_td, w_td = res.mask.shape
    frame_top_down = cv2.resize(frame_roi, (w_td, h_td))

    # 4 Draw debug overlays on top-down frame
    dbg = draw_debug_slices(
        frame,
        res.mask,
        res.roi_start,
        res.y_top,
        res.y_mid,
        res.y_bot,
        res.cx_top,
        res.cx_mid,
        res.cx_bot,
        BAND_THICKNESS,
        res.steer_error,
        res.t_detected,
    )

    # 5 Draw mask visualization (top-down)
    mask_dbg = draw_mask_slices(
        mask=res.mask,
        y_top=res.y_top,
        y_mid=res.y_mid,
        y_bot=res.y_bot,
        cx_top=res.cx_top,
        cx_mid=res.cx_mid,
        cx_bot=res.cx_bot
    )

    # 6 Display
    cv2.imshow("camera debug", dbg)
    cv2.imshow("mask debug", mask_dbg)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
