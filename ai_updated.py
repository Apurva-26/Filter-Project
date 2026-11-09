import os
import sys
import urllib.request
import cv2
import numpy as np

import kiosk  # touchscreen Back button + on-screen buttons (see kiosk.py)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "face_detection_yunet_2023mar.onnx"
)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

CLOWN_NOSE_PATH = os.path.join(SCRIPT_DIR, "clown_nose.png")


def load_model() -> None:
    """Download the YuNet model if needed."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100000:
        return

    print("Downloading YuNet face model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def overlay_clown_nose(frame, nose_img, center_x, center_y, size):

    nose_resized = cv2.resize(nose_img, (size, size))

    if nose_resized.shape[2] == 4:
        b, g, r, alpha = cv2.split(nose_resized)
        nose_rgb = cv2.merge((b, g, r))
        alpha = alpha.astype(float) / 255.0
    else:
        nose_rgb = nose_resized
        alpha = np.ones((size, size), dtype=float)

    x1 = center_x - size // 2
    y1 = center_y - size // 2
    x2 = x1 + size
    y2 = y1 + size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    actual_width = x2 - x1
    actual_height = y2 - y1

    nose_rgb = nose_rgb[:actual_height, :actual_width]
    alpha = alpha[:actual_height, :actual_width]

    roi = frame[y1:y2, x1:x2]

    for channel in range(3):
        roi[:, :, channel] = (
            alpha * nose_rgb[:, :, channel]
            + (1 - alpha) * roi[:, :, channel]
        )

    frame[y1:y2, x1:x2] = roi

    return frame


def enlarge_eyes(frame, left_eye, right_eye, radius=35, scale=1.8):

    for eye_x, eye_y in [left_eye, right_eye]:

        eye_x = int(eye_x)
        eye_y = int(eye_y)

        x1 = max(0, eye_x - radius)
        y1 = max(0, eye_y - radius)
        x2 = min(frame.shape[1], eye_x + radius)
        y2 = min(frame.shape[0], eye_y + radius)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        h, w = roi.shape[:2]

        enlarged = cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        eh, ew = enlarged.shape[:2]

        crop_x = (ew - w) // 2
        crop_y = (eh - h) // 2

        enlarged = enlarged[
            crop_y:crop_y + h,
            crop_x:crop_x + w
        ]

        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.circle(
            mask,
            (w // 2, h // 2),
            min(w, h) // 2,
            255,
            -1
        )

        roi_bg = cv2.bitwise_and(
            roi,
            roi,
            mask=cv2.bitwise_not(mask)
        )

        eye_fg = cv2.bitwise_and(
            enlarged,
            enlarged,
            mask=mask
        )

        frame[y1:y2, x1:x2] = cv2.add(
            roi_bg,
            eye_fg
        )

    return frame


def main():

    load_model()

    clown_nose = cv2.imread(
        CLOWN_NOSE_PATH,
        cv2.IMREAD_UNCHANGED
    )

    if clown_nose is None:
        print("Error: Could not load clown nose image.")
        return

    backend = cv2.CAP_ANY

    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW

    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (640, 480),
        score_threshold=0.8
    )

    current_filter = "clown"
    filter_enabled = True

    print("Controls:")
    print("  N = Enable/Disable active filter")
    print("  F = Switch filters")
    print("  Q = Quit")

    # On-screen buttons that press the same keys as above
    screen = kiosk.Kiosk("Face Filters", buttons=[("On / Off", "n"), ("Switch", "f")])

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)

        if filter_enabled and faces is not None:

            for face in faces:

                if current_filter == "clown":

                    nose_x = int(face[8])
                    nose_y = int(face[9])

                    face_width = int(face[2])
                    nose_size = max(40, face_width // 4)

                    frame = overlay_clown_nose(
                        frame,
                        clown_nose,
                        nose_x,
                        nose_y,
                        nose_size
                    )

                elif current_filter == "eyes":

                    left_eye = (
                        int(face[4]),
                        int(face[5])
                    )

                    right_eye = (
                        int(face[6]),
                        int(face[7])
                    )

                    eye_radius = max(
                        25,
                        int(face[2] * 0.12)
                    )

                    frame = enlarge_eyes(
                        frame,
                        left_eye,
                        right_eye,
                        eye_radius,
                        1.8
                    )

        status = "ON" if filter_enabled else "OFF"

        cv2.putText(
            frame,
            f"Filter: {current_filter.upper()} ({status})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Shows the frame (like cv2.imshow) and returns the key pressed or tapped
        key = screen.show(frame, person_seen=faces is not None)

        if key == ord('n'):
            filter_enabled = not filter_enabled
            print(
                f"Filter {'enabled' if filter_enabled else 'disabled'}"
            )

        elif key == ord('f'):

            if current_filter == "clown":
                current_filter = "eyes"
            else:
                current_filter = "clown"

            print(
                f"Switched to {current_filter} filter"
            )

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()