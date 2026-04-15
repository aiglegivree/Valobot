from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from leds import LED


# Camera
W = 640
H = 480
CAM_ID = 0
USE_PICAMERA2 = True

# ROI / mask preprocessing
ROI_START_FRAC = 0.50
GAUSS_KSIZE = 5
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0
BLACK_H_HIGH = 180
BLACK_S_HIGH = 255
BLACK_V_HIGH = 130
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2
MIN_CONTOUR_AREA = 1800

# ORB
TRAIN_ROOT = Path(__file__).with_name("t_training_frames")
NORMALIZED_SIZE = 200
ORB_FEATURES = 500
MIN_REFERENCE_KEYPOINTS = 12
MIN_LIVE_KEYPOINTS = 12
MIN_GOOD_MATCHES = 14
RATIO_TEST = 0.78
MAX_MEAN_DISTANCE = 55.0


class Camera:
    def __init__(self):
        if USE_PICAMERA2:
            from picamera2 import Picamera2

            self.cam = Picamera2()
            self.cam.configure(
                self.cam.create_preview_configuration(
                    main={"size": (W, H), "format": "RGB888"}
                )
            )
            self.cam.start()
            self.type = "picamera2"
        else:
            cap = cv2.VideoCapture(CAM_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            if not cap.isOpened():
                raise RuntimeError("Camera not found")
            self.cam = cap
            self.type = "opencv"

    def read(self):
        if self.type == "picamera2":
            frame = self.cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame

        ret, frame = self.cam.read()
        return ret, frame

    def release(self):
        if self.type == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()


def keep_only_bottom_connected(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    queue: list[tuple[int, int]] = []

    for x in range(w):
        if mask[h - 1, x] == 255:
            visited[h - 1, x] = 1
            queue.append((h - 1, x))

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if visited[ny, nx] == 0 and mask[ny, nx] == 255:
                    visited[ny, nx] = 1
                    queue.append((ny, nx))

    out = np.zeros_like(mask)
    out[visited == 1] = 255
    return out


def compute_mask(frame: np.ndarray) -> tuple[np.ndarray, int]:
    h_frame, _ = frame.shape[:2]
    roi_start = int(h_frame * ROI_START_FRAC)
    roi = frame[roi_start:]

    roi_blur = cv2.GaussianBlur(roi, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)
    mask = keep_only_bottom_connected(mask)
    return mask, roi_start


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return None

    return contour


def normalize_contour_mask(mask: np.ndarray, contour: np.ndarray, out_size: int = NORMALIZED_SIZE) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    crop = mask[y:y + h, x:x + w]

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    scale = min((out_size - 12) / max(w, 1), (out_size - 12) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    x0 = (out_size - new_w) // 2
    y0 = (out_size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def build_orb_input(binary_mask: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(binary_mask, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(normalized, (3, 3), 0)


def create_orb():
    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=10,
    )


def iter_reference_images() -> list[Path]:
    if not TRAIN_ROOT.exists():
        return []

    roi_images = sorted(TRAIN_ROOT.glob("session_*/roi/*.png"))
    if roi_images:
        return roi_images

    return sorted(TRAIN_ROOT.glob("**/*.png"))


def build_reference_database(orb) -> list[dict]:
    references: list[dict] = []

    for image_path in iter_reference_images():
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        mask = preprocess_reference_image(image)
        contour = largest_contour(mask)
        if contour is None:
            continue

        normalized = normalize_contour_mask(mask, contour)
        orb_input = build_orb_input(normalized)
        keypoints, descriptors = orb.detectAndCompute(orb_input, None)

        if descriptors is None or len(keypoints) < MIN_REFERENCE_KEYPOINTS:
            continue

        references.append(
            {
                "path": image_path,
                "normalized": normalized,
                "orb_input": orb_input,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )

    return references


def preprocess_reference_image(image_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image_bgr, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    low = np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8)
    high = np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)


def match_against_references(live_descriptors: np.ndarray, references: list[dict], matcher) -> dict | None:
    best: dict | None = None

    for reference in references:
        raw_matches = matcher.knnMatch(reference["descriptors"], live_descriptors, k=2)
        good_matches = []

        for pair in raw_matches:
            if len(pair) < 2:
                continue

            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                good_matches.append(m)

        if not good_matches:
            continue

        mean_distance = float(np.mean([match.distance for match in good_matches]))
        candidate = {
            "reference": reference,
            "good_matches": good_matches,
            "good_match_count": len(good_matches),
            "mean_distance": mean_distance,
        }

        if best is None:
            best = candidate
            continue

        if candidate["good_match_count"] > best["good_match_count"]:
            best = candidate
            continue

        if candidate["good_match_count"] == best["good_match_count"] and candidate["mean_distance"] < best["mean_distance"]:
            best = candidate

    return best


def classify_match(best_match: dict | None, live_keypoint_count: int) -> tuple[bool, str]:
    if live_keypoint_count < MIN_LIVE_KEYPOINTS:
        return False, "Not enough live keypoints"

    if best_match is None:
        return False, "No ORB match"

    if best_match["good_match_count"] < MIN_GOOD_MATCHES:
        return False, "Too few good matches"

    if best_match["mean_distance"] > MAX_MEAN_DISTANCE:
        return False, "Match distance too high"

    return True, "T detected"


def build_match_view(best_match: dict | None, live_input: np.ndarray, live_keypoints: list, live_bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if best_match is None:
        return cv2.cvtColor(live_input, cv2.COLOR_GRAY2BGR)

    view = cv2.drawMatches(
        best_match["reference"]["orb_input"],
        best_match["reference"]["keypoints"],
        live_input,
        live_keypoints,
        best_match["good_matches"][:30],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    label = best_match["reference"]["path"].name
    cv2.putText(view, f"Ref: {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if live_bbox is not None:
        x, y, w, h = live_bbox
        cv2.putText(view, f"BBox: {w}x{h}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return view


def main():
    leds = LED()
    leds.turn_on_all()

    orb = create_orb()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    references = build_reference_database(orb)
    cam = Camera()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            mask, roi_start = compute_mask(frame)
            contour = largest_contour(mask)

            live_input = np.zeros((NORMALIZED_SIZE, NORMALIZED_SIZE), dtype=np.uint8)
            live_keypoints = []
            live_descriptors = None
            best_match = None
            bbox = None

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, roi_start),
                (overlay.shape[1] - 1, overlay.shape[0] - 1),
                (0, 255, 255),
                2,
            )

            status_text = "No contour"
            is_t = False

            if contour is not None:
                bbox = cv2.boundingRect(contour)
                x, y, w, h = bbox
                normalized = normalize_contour_mask(mask, contour)
                live_input = build_orb_input(normalized)
                live_keypoints, live_descriptors = orb.detectAndCompute(live_input, None)

                if live_descriptors is not None and references:
                    best_match = match_against_references(live_descriptors, references, matcher)

                is_t, status_text = classify_match(best_match, len(live_keypoints))
                color = (0, 255, 0) if is_t else (0, 0, 255)
                cv2.rectangle(overlay, (x, y + roi_start), (x + w, y + h + roi_start), color, 3)
            elif not references:
                status_text = "No T references found"

            cv2.putText(overlay, f"ORB refs: {len(references)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(
                overlay,
                f"T: {'YES' if is_t else 'NO'}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if is_t else (0, 0, 255),
                2,
            )
            cv2.putText(overlay, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            if best_match is not None:
                cv2.putText(
                    overlay,
                    f"Good matches: {best_match['good_match_count']}",
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    overlay,
                    f"Mean dist: {best_match['mean_distance']:.1f}",
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

            match_view = build_match_view(best_match, live_input, live_keypoints, bbox)
            keypoint_view = cv2.drawKeypoints(
                live_input,
                live_keypoints,
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            if not references:
                cv2.putText(
                    match_view,
                    "Capture T images first with capture_t_frames.py",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("T ORB detection", overlay)
            cv2.imshow("T ORB mask", mask)
            cv2.imshow("T ORB normalized", keypoint_view)
            cv2.imshow("T ORB matches", match_view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
