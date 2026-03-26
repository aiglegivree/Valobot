import cv2
import time
from obstacles import CorridorNavigator
from motors import MotorDriver
from leds import LED


def clamp(value, low, high):
    return max(low, min(high, value))


def execute_turn(navigator, motors, direction, dead_zone_seconds, max_seconds, turn_speed):
    """
    Turns in `direction` (+1=right, -1=left) for up to `max_seconds`.
    Suppresses state updates for the first `dead_zone_seconds`.
    Returns True if navigator switches back to FORWARD, False if max_seconds exceeded.
    """
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time

        if elapsed >= max_seconds:
            return False

        motors.set_speeds(direction * turn_speed, -direction * turn_speed)

        allow_update = (elapsed >= dead_zone_seconds)
        ret, vis = navigator.step(draw=True, update=allow_update)

        if vis is not None:
            cv2.imshow("Navigation", vis)
        cv2.waitKey(1)

        if allow_update and navigator.get_control_data()["state"] == "FORWARD":
            return True


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
    base_speed = 0.2
    k_p = 0.2
    max_correction = 0.20
    lost_target_speed = 0.2

    # ------------------------------------------------------------
    # Turn tuning
    # ------------------------------------------------------------
    turn_speed = 0.3
    dead_zone_seconds = 0.5   # suppress state updates for first 1s of turn
    max_seconds_90 = 3.0      # budget for ~90° turn
    max_seconds_180 = 6.0     # budget for ~180° fallback turn

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
            # TURN condition reached
            # ----------------------------------------------------
            if state == "TURN":
                motors.stop()
                turn_candidate = data["turn_candidate"]

                direction = 1 if turn_candidate == "RIGHT" else -1

                # First attempt: turn toward candidate (~90°)
                success = execute_turn(
                    navigator, motors,
                    direction=direction,
                    dead_zone_seconds=dead_zone_seconds,
                    max_seconds=max_seconds_90,
                    turn_speed=turn_speed
                )

                if not success:
                    print("90° turn failed, attempting 180° in opposite direction.")
                    motors.stop()
                    # Second attempt: turn opposite direction (~180°)
                    success = execute_turn(
                        navigator, motors,
                        direction=-direction,
                        dead_zone_seconds=0.0,
                        max_seconds=max_seconds_180,
                        turn_speed=turn_speed
                    )

                if not success:
                    print("ERROR: could not find corridor after full rotation. Stopping.")
                    motors.stop()
                    break

                print("Turn complete, resuming FORWARD.")

            # ----------------------------------------------------
            # FORWARD mode
            # ----------------------------------------------------
            elif state == "FORWARD":
                if target_visible and frame_width is not None:
                    image_center_x = frame_width / 2.0
                    error_pixels = target_x - image_center_x
                    error_norm = error_pixels / image_center_x

                    correction = k_p * error_norm
                    correction = clamp(correction, -max_correction, max_correction)

                    left_speed = clamp(base_speed + correction, -1.0, 1.0)
                    right_speed = clamp(base_speed - correction, -1.0, 1.0)

                    motors.set_speeds(left_speed, right_speed)
                else:
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
