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
    Returns True if navigator switches back to FORWARD, False if timed out.
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

    navigator = CorridorNavigator(
        proto=False,
        # Pass alignment tuning here if you want to override defaults:
        # corridor_left_theta=0.8,
        # corridor_right_theta=2.3,
        # alignment_threshold_deg=8.0,
        # alignment_gain_px_per_rad=200.0,
        # single_line_wall_offset=80,
    )
    motors = MotorDriver()

    # ----------------------------------------------------------------
    # TUNING — forward control
    # base_speed: straight-line speed (0–1)
    # k_p: proportional gain on lateral error. Raise if correction is
    #       too slow, lower if robot oscillates side to side.
    # max_correction: caps the speed differential. Prevents spinning.
    # lost_target_speed: speed when no target is visible (go straight).
    # error_deadband_px: ignore errors smaller than this to avoid jitter.
    # ----------------------------------------------------------------
    base_speed = 0.25
    k_p = 0.1
    max_correction = 0.20
    lost_target_speed = 0.2
    error_deadband_px = 5

    # ----------------------------------------------------------------
    # TUNING — turning
    # turn_speed: rotation speed during a turn (0–1).
    # dead_zone_seconds: how long to suppress state updates at the start
    #   of a turn. Must be long enough to rotate past the old corridor.
    #   Raise if robot exits turn too early seeing the old corridor.
    # max_seconds_90: time budget for the first ~90° turn attempt.
    # max_seconds_180: time budget for the 180° fallback attempt.
    # dead_zone_180: dead zone for the fallback turn. Must be long enough
    #   to clear the origin corridor (~dead_zone_seconds + some margin).
    # ----------------------------------------------------------------
    turn_speed = 0.25
    dead_zone_seconds = 0.8
    max_seconds_90 = 2.4
    max_seconds_180 = 4.8
    dead_zone_180 = max_seconds_90 + dead_zone_seconds

    try:
        while True:
            ret, vis = navigator.step(draw=True)
            if not ret:
                print("No frame received. Stopping.")
                break

            data = navigator.get_control_data()
            state = data["state"]

            # ----------------------------------------------------
            # TURN state
            # ----------------------------------------------------
            if state == "TURN":
                motors.stop()
                turn_candidate = data["turn_candidate"]
                direction = 1 if turn_candidate == "RIGHT" else -1

                success = execute_turn(
                    navigator, motors,
                    direction=direction,
                    dead_zone_seconds=dead_zone_seconds,
                    max_seconds=max_seconds_90,
                    turn_speed=turn_speed,
                )

                if not success:
                    print("90° turn failed, attempting 180° in opposite direction.")
                    motors.stop()
                    success = execute_turn(
                        navigator, motors,
                        direction=-direction,
                        dead_zone_seconds=dead_zone_180,
                        max_seconds=max_seconds_180,
                        turn_speed=turn_speed,
                    )

                if not success:
                    print("ERROR: could not find corridor. Stopping.")
                    motors.stop()
                    break

                print("Turn complete, resuming FORWARD.")
                continue  # re-enter loop cleanly with a fresh step()

            # ----------------------------------------------------
            # FORWARD state
            # ----------------------------------------------------
            elif state == "FORWARD":
                target_x = data["target_x"]
                target_visible = data["target_visible"]
                frame_width = data["frame_width"]

                if target_visible and frame_width is not None:
                    image_center_x = frame_width / 2.0
                    error_pixels = target_x - image_center_x

                    if abs(error_pixels) < error_deadband_px:
                        error_pixels = 0.0

                    error_norm = error_pixels / image_center_x
                    correction = clamp(k_p * error_norm, -max_correction, max_correction)

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