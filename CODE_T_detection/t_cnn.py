from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from leds import LED


# Camera
W = 640
H = 480
CAM_ID = 0
USE_PICAMERA2 = True

# Preprocessing
GAUSS_KSIZE = 5
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0
BLACK_H_HIGH = 180
BLACK_S_HIGH = 255
BLACK_V_HIGH = 130
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2
ROI_START_FRAC = 0.50
MIN_CONTOUR_AREA = 1200

# CNN
IMG_SIZE = 96
DEFAULT_THRESHOLD = 0.80
MODEL_PATH = Path(__file__).with_name("t_cnn_model.keras")
POS_ROOT = Path(__file__).with_name("t_training_frames")
NEG_ROOT = Path(__file__).with_name("not_t_training_frames")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run a CNN T detector.")
    parser.add_argument(
        "--mode",
        choices=("train", "live"),
        default="train",
        help="train from saved images or run live with a trained model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="T probability threshold in live mode",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="training epochs",
    )
    return parser.parse_args()


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
    roi_start = int(frame.shape[0] * ROI_START_FRAC)
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


def collect_positive_paths() -> list[Path]:
    paths = sorted(POS_ROOT.glob("session_*/roi/*.png"))
    if paths:
        return paths
    return sorted(POS_ROOT.glob("**/*.png"))


def collect_negative_paths() -> list[Path]:
    if not NEG_ROOT.exists():
        return []
    return sorted(NEG_ROOT.glob("**/*.png"))


def crop_from_masked_image(image_bgr: np.ndarray) -> np.ndarray | None:
    mask = cv2.inRange(
        cv2.cvtColor(cv2.GaussianBlur(image_bgr, (GAUSS_KSIZE, GAUSS_KSIZE), 0), cv2.COLOR_BGR2HSV),
        np.array([BLACK_H_LOW, BLACK_S_LOW, BLACK_V_LOW], dtype=np.uint8),
        np.array([BLACK_H_HIGH, BLACK_S_HIGH, BLACK_V_HIGH], dtype=np.uint8),
    )
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_IT)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    return image_bgr[y:y + h, x:x + w]


def preprocess_patch(patch_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = gray

    resized = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[..., None]


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for path in collect_positive_paths():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        patch = crop_from_masked_image(image)
        if patch is None:
            patch = image
        x_list.append(preprocess_patch(patch))
        y_list.append(1)

    for path in collect_negative_paths():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        patch = crop_from_masked_image(image)
        if patch is None:
            patch = image
        x_list.append(preprocess_patch(patch))
        y_list.append(0)

    if not x_list:
        raise RuntimeError("No training images found.")
    if len(set(y_list)) < 2:
        raise RuntimeError(
            "Need both classes for CNN training. "
            "Use t_training_frames for T images and not_t_training_frames for non-T images."
        )

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return x, y


def build_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for the CNN script. "
            "Install it first on the target machine."
        ) from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(epochs: int):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for the CNN script. "
            "Install it first on the target machine."
        ) from exc

    x, y = load_dataset()
    model = build_model()

    augmenter = keras.Sequential(
        [
            keras.layers.RandomRotation(0.5),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomTranslation(0.08, 0.08),
        ]
    )

    x_train = augmenter(x, training=True).numpy()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y,
        epochs=max(1, epochs),
        batch_size=min(16, len(x_train)),
        validation_split=0.2 if len(x_train) >= 10 else 0.0,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(MODEL_PATH))
    print(f"Saved CNN model to {MODEL_PATH}")
    print(f"Samples: {len(x)}")
    print(f"Positives: {int(np.sum(y == 1))}")
    print(f"Negatives: {int(np.sum(y == 0))}")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")


def classify_patch(model, patch_bgr: np.ndarray) -> float:
    x = preprocess_patch(patch_bgr)[None, ...]
    prob = float(model.predict(x, verbose=0)[0, 0])
    return prob


def live_mode(threshold: float):
    try:
        from tensorflow import keras
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for the CNN script. "
            "Install it first on the target machine."
        ) from exc

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}. Train first with --mode train.")

    model = keras.models.load_model(str(MODEL_PATH))
    leds = LED()
    leds.turn_on_all()
    cam = Camera()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            mask, roi_start = compute_mask(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, roi_start),
                (overlay.shape[1] - 1, overlay.shape[0] - 1),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                f"CNN thr={threshold:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            best_prob = 0.0
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                patch = frame[roi_start + y:roi_start + y + h, x:x + w]
                if patch.size == 0:
                    continue

                prob = classify_patch(model, patch)
                best_prob = max(best_prob, prob)
                color = (0, 255, 0) if prob >= threshold else (0, 165, 255)
                cv2.rectangle(overlay, (x, roi_start + y), (x + w, roi_start + y + h), color, 2)
                cv2.putText(
                    overlay,
                    f"{prob:.2f}",
                    (x, max(roi_start + y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.putText(
                overlay,
                f"T: {'YES' if best_prob >= threshold else 'NO'} best={best_prob:.2f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if best_prob >= threshold else (0, 0, 255),
                2,
            )

            cv2.imshow("T CNN detection", overlay)
            cv2.imshow("T CNN mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        try:
            leds.turn_off_all()
        except Exception:
            pass
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.mode == "train":
        train_model(args.epochs)
    else:
        live_mode(args.threshold)


if __name__ == "__main__":
    main()
