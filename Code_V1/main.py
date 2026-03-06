from camera import Camera
from vision import detect_line
from controller import Controller
from motors import MotorDriver

cam = Camera()
controller = Controller()
motors = MotorDriver()

while True:

    ret, frame = cam.read()

    if not ret:
        continue

    vision = detect_line(frame)

    left, right, speed, steer, mode = controller.update(vision)

    motors.set_speeds(left, right)

    res = detect_line(frame)

    dbg = draw_debug_slices(
        frame_bgr=frame,
        mask=res.mask,
        roi_start=res.roi_start,
        y_top=res.y_top,
        y_mid=res.y_mid,
        y_bot=res.y_bot,
        cx_top=res.cx_top,
        cx_mid=res.cx_mid,
        cx_bot=res.cx_bot,
        band_thickness=6
    )

    mask_dbg = draw_mask_slices(
        mask=res.mask,
        y_top=res.y_top,
        y_mid=res.y_mid,
        y_bot=res.y_bot,
        cx_top=res.cx_top,
        cx_mid=res.cx_mid,
        cx_bot=res.cx_bot
    )

    cv2.imshow("camera debug", dbg)
    cv2.imshow("mask debug", mask_dbg)