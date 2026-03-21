import cv2
from obstacles import CorridorNavigator
from motors import MotorDriver
from leds import LED


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    leds = LED()
    leds.turn_off_all()
    # ------------------------------------------------------------
    # Vision + motors
    # ------------------------------------------------------------
    navigator = CorridorNavigator(proto=False)
    motors = MotorDriver()

    # ------------------------------------------------------------
    # Forward-control tuning
    # ------------------------------------------------------------
    base_speed = 0.35       # forward speed in [-1, 1]
    k_p = 0.2               # proportional steering gain
    max_correction = 0.20    # limits steering effect

    # Fallback behavior when target is temporarily not visible
    lost_target_speed = 0.25

    try:
        while True:
            ret, vis = navigator.step(draw=True)
            if not ret:
                print("No frame received. Stopping.")
                break

            data = navigator.get_control_data()

            state = data["state"]
            target_x = data["target_x"]
            target_visible = data["target_visible"]
            frame_width = data["frame_width"]

            # ----------------------------------------------------
            # TURN condition reached -> stop
            # ----------------------------------------------------
            if state == "TURN":
                motors.stop()
                print("TURN detected. Stopping forward mode.")
                cv2.imshow("Navigation", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                break

            # ----------------------------------------------------
            # FORWARD mode
            # ----------------------------------------------------
            if state == "FORWARD":
                if target_visible and frame_width is not None:
                    image_center_x = frame_width / 2.0

                    # error > 0  -> corridor center is to the right
                    # error < 0  -> corridor center is to the left
                    error_pixels = target_x - image_center_x

                    # normalize error to roughly [-1, 1]
                    error_norm = error_pixels / image_center_x

                    correction = k_p * error_norm
                    correction = clamp(correction, -max_correction, max_correction)

                    left_speed = base_speed + correction
                    right_speed = base_speed - correction

                    left_speed = clamp(left_speed, -1.0, 1.0)
                    right_speed = clamp(right_speed, -1.0, 1.0)

                    motors.set_speeds(left_speed, right_speed)


                else:
                    # If target is not visible, slow down instead of steering blindly
                    motors.set_speeds(lost_target_speed, lost_target_speed)

            cv2.imshow("Navigation", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        motors.cleanup()
        navigator.release()


if __name__ == "__main__":
    main()
