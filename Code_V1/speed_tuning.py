import cv2
from motors import MotorDriver


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    motors = MotorDriver()

    # ------------------------------------------------------------
    # Initial tuning values
    # ------------------------------------------------------------
    base_speed = 0.25
    trim = 0.0
    trim_step = 0.005

    print("Motor trim tuning")
    print("-----------------")
    print("Controls:")
    print("  i : increase trim  (left faster, right slower)")
    print("  d : decrease trim  (right faster, left slower)")
    print("  space : stop motors")
    print("  q : quit")
    print()

    try:
        while True:
            left_speed = clamp(base_speed + trim, -1.0, 1.0)
            right_speed = clamp(base_speed - trim, -1.0, 1.0)

            motors.set_speeds(left_speed, right_speed)

            print(
                f"\rbase_speed={base_speed:+.3f}   "
                f"trim={trim:+.3f}   "
                f"left={left_speed:+.3f}   "
                f"right={right_speed:+.3f}",
                end="",
                flush=True
            )

            key = cv2.waitKey(50) & 0xFF

            if key == ord("i"):
                trim = clamp(trim + trim_step, -1.0, 1.0)

            elif key == ord("d"):
                trim = clamp(trim - trim_step, -1.0, 1.0)

            elif key == ord("r"):
                trim = 0.0

            elif key == ord(" "):
                motors.stop()

            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        motors.stop()
        motors.cleanup()
        cv2.destroyAllWindows()
        print("\nExited cleanly.")


if __name__ == "__main__":
    main()