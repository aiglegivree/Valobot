import cv2
import numpy as np

def find_hsv_thresholds_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not determine frame count for: {video_path}")

    window_controls = "HSV Controls"
    window_mask = "Mask"
    window_result = "Segmented Result"
    window_original = "Original Frame"

    cv2.namedWindow(window_controls, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_mask, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_original, cv2.WINDOW_NORMAL)

    def nothing(x):
        pass

    # HSV trackbars
    cv2.createTrackbar("H min", window_controls, 0, 179, nothing)
    cv2.createTrackbar("H max", window_controls, 179, 179, nothing)
    cv2.createTrackbar("S min", window_controls, 0, 255, nothing)
    cv2.createTrackbar("S max", window_controls, 255, 255, nothing)
    cv2.createTrackbar("V min", window_controls, 0, 255, nothing)
    cv2.createTrackbar("V max", window_controls, 255, 255, nothing)

    # Frame selector
    cv2.createTrackbar("Frame", window_controls, 0, total_frames - 1, nothing)

    current_frame_idx = 0
    last_frame_idx = -1
    frame = None

    while True:
        current_frame_idx = cv2.getTrackbarPos("Frame", window_controls)

        # Only reload frame if frame index changed
        if current_frame_idx != last_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"Could not read frame {current_frame_idx}")
                break

            last_frame_idx = current_frame_idx

        h_min = cv2.getTrackbarPos("H min", window_controls)
        h_max = cv2.getTrackbarPos("H max", window_controls)
        s_min = cv2.getTrackbarPos("S min", window_controls)
        s_max = cv2.getTrackbarPos("S max", window_controls)
        v_min = cv2.getTrackbarPos("V min", window_controls)
        v_max = cv2.getTrackbarPos("V max", window_controls)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        display = frame.copy()
        cv2.putText(
            display,
            f"Frame: {current_frame_idx}/{total_frames - 1}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(window_original, display)
        cv2.imshow(window_mask, mask)
        cv2.imshow(window_result, result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            print("Final HSV thresholds:")
            print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"Chosen frame: {current_frame_idx}")
            break

        elif key == ord('d'):  # next frame
            current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
            cv2.setTrackbarPos("Frame", window_controls, current_frame_idx)

        elif key == ord('a'):  # previous frame
            current_frame_idx = max(current_frame_idx - 1, 0)
            cv2.setTrackbarPos("Frame", window_controls, current_frame_idx)

    cap.release()
    cv2.destroyAllWindows()

    return lower, upper


lower, upper = find_hsv_thresholds_from_video("fixed.avi")

print(lower)
print(upper)