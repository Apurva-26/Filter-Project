"""
Hallway Display Filter: changes the iris of every opted-in face to
blood maroon, bright pink, or gold (picked at random per person).
Uses MediaPipe FaceLandmarker (478 points: exact iris & 16-point eyelid contours).
"""
import math
import os
import random
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# 16-point eyelid outlines + iris center/rim indices in MediaPipe's 478-point mesh
LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS_CENTER = 468
LEFT_IRIS_RIM = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RIM = [474, 475, 476, 477]

# Filter colors (BGR) -- used when an iris picture is missing
COLORS = {
    "blood_maroon": (25, 10, 150),
    "bright_pink": (170, 40, 255),
    "gold": (20, 170, 230),
}
COLOR_KEYS = list(COLORS.keys())

# Tuning knobs
IRIS_RADIUS_RATIO = 0.10   # fallback ratio if iris rim points aren't used
IRIS_OPACITY = 0.85        # how strongly the new iris covers the real one
SMOOTHING = 0.45           # 0 = no smoothing, closer to 1 = steadier
MAX_JUMP_RATIO = 0.6       # ignore an eye point that jumps more than this * eye distance

# Opt-in zone
MAX_DIST_M = 1.33          # farther than this = bystander (blurred)
MAX_HORIZ_OFFSET_M = 0.5   # must be within 0.5 m left/right of the camera center
HOLD_SECONDS = 2.0         # hold still for 2 seconds to opt in
FORGET_AFTER_S = 0.75      # forget person once they step out

CALIBRATION_K = 45.0
REAL_EYE_DIST_M = 0.063


def load_model() -> None:
    """Download the MediaPipe FaceLandmarker model the first time the script runs."""
    if os.path.exists(TASK_PATH) and os.path.getsize(TASK_PATH) > 1_000_000:
        return
    print("Downloading MediaPipe FaceLandmarker model (one time only)...")
    urllib.request.urlretrieve(TASK_URL, TASK_PATH)
    print("Done.")


def make_iris_texture(bgr_color, size=256, seed=0) -> np.ndarray:
    """Draw a realistic-looking iris (fibers, dark rim, pupil, catchlight)."""
    rng = np.random.default_rng(seed)
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - c, yy - c) / c
    theta = np.arctan2(yy - c, xx - c)

    fibers = sum(np.sin(theta * f + rng.uniform(0, 2 * np.pi) + r * rng.uniform(2, 6))
                 for f in (40, 73, 121, 190)) / 4.0
    bright = 0.85 + 0.25 * fibers
    bright *= 1.0 + 0.35 * np.exp(-((r - 0.45) ** 2) / 0.01)
    bright *= np.clip(1.15 - 0.9 * np.clip((r - 0.75) / 0.25, 0, 1), 0, None)

    tex = np.clip(np.array(bgr_color, np.float32) * bright[..., None], 0, 255)
    pupil = np.clip((r - 0.28) / 0.03, 0, 1)[..., None]
    tex = tex * pupil + 12 * (1 - pupil)
    glint = np.exp(-(((xx - c * 0.72) ** 2 + (yy - c * 0.72) ** 2) / (2 * (size * 0.035) ** 2)))
    tex = tex + (255 - tex) * glint[..., None] * 0.9
    return tex.astype(np.uint8)


def load_iris_textures() -> dict:
    """Load iris_<color>.png for each color, or draw one if it's missing."""
    textures = {}
    for i, key in enumerate(COLOR_KEYS):
        img = None
        for folder in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "images")):
            path = os.path.join(folder, f"iris_{key}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                break
        if img is None:
            print(f"iris_{key}.png not found -- using a drawn {key} iris.")
            img = make_iris_texture(COLORS[key], seed=i)
        else:
            h, w = img.shape[:2]
            s = min(h, w)
            img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        textures[key] = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return textures


class Person:
    def __init__(self, left_eye, right_eye, now):
        self.left = np.array(left_eye, np.float32)
        self.right = np.array(right_eye, np.float32)
        self.last_seen = now
        self.in_zone_since = None
        self.color = None

    def center(self):
        return (self.left + self.right) / 2

    def eye_dist(self):
        return float(np.linalg.norm(self.right - self.left))

    def update(self, left_eye, right_eye, now):
        max_jump = MAX_JUMP_RATIO * max(self.eye_dist(), 1.0)
        for name, new in (("left", left_eye), ("right", right_eye)):
            new = np.array(new, np.float32)
            old = getattr(self, name)
            if np.linalg.norm(new - old) <= max_jump:
                setattr(self, name, SMOOTHING * old + (1 - SMOOTHING) * new)
        self.last_seen = now


def match_people(people, eye_pairs, now):
    people[:] = [p for p in people if now - p.last_seen <= FORGET_AFTER_S]
    unused = list(people)
    matched = []
    for left_eye, right_eye in eye_pairs:
        center = (np.array(left_eye) + np.array(right_eye)) / 2
        best, best_d = None, math.dist(left_eye, right_eye) * 1.5
        for p in unused:
            d = float(np.linalg.norm(p.center() - center))
            if d < best_d:
                best, best_d = p, d
        if best is None:
            best = Person(left_eye, right_eye, now)
            people.append(best)
        else:
            unused.remove(best)
            best.update(left_eye, right_eye, now)
        matched.append(best)
    return matched


def overlay_iris(frame, texture, eye_center, radius, angle_deg, eyelid_pts) -> None:
    """Draws the iris clipped inside the dynamic eyelid polygon so blinking shrinks/hides it."""
    f_h, f_w = frame.shape[:2]
    t = texture.shape[0]
    radius = max(3.0, radius)

    x1 = max(0, int(min(eye_center[0] - radius, eyelid_pts[:, 0].min())) - 2)
    y1 = max(0, int(min(eye_center[1] - radius, eyelid_pts[:, 1].min())) - 2)
    x2 = min(f_w, int(max(eye_center[0] + radius, eyelid_pts[:, 0].max())) + 3)
    y2 = min(f_h, int(max(eye_center[1] + radius, eyelid_pts[:, 1].max())) + 3)
    if x2 <= x1 or y2 <= y1:
        return

    # 1. Warp the iris texture into the eye ROI
    M = cv2.getRotationMatrix2D((t / 2, t / 2), -angle_deg, (2 * radius) / t)
    M[0, 2] += (eye_center[0] - x1) - t / 2
    M[1, 2] += (eye_center[1] - y1) - t / 2
    warped = cv2.warpAffine(
        texture, M, (x2 - x1, y2 - y1),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    # 2. Circular iris mask (keeps eye whites natural)
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    dist = np.hypot(xx - eye_center[0], yy - eye_center[1])
    circle_alpha = np.clip((radius - dist) / max(1.0, radius * 0.15), 0, 1)

    # 3. Dynamic eyelid polygon mask (clips the iris as the person blinks)
    lid_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    local_pts = eyelid_pts - np.array([[x1, y1]], dtype=np.int32)
    cv2.fillPoly(lid_mask, [local_pts], 255)
    lid_alpha = cv2.GaussianBlur(lid_mask, (3, 3), 0).astype(np.float32) / 255.0

    # 4. Combine masks so only the visible portion of the iris inside the eyelids is colored
    alpha = (circle_alpha * lid_alpha * IRIS_OPACITY)[..., None]
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (warped * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def blur_face(frame, bbox) -> None:
    x, y, w, h = bbox
    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
    f_h, f_w = frame.shape[:2]
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(f_w, x + w + pad_w), min(f_h, y + h + pad_h)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (51, 51), 30)


def draw_hold_ring(frame, person, progress) -> None:
    cx, cy = person.center()
    r = max(8, int(person.eye_dist() * 0.25))
    center = (int(cx), int(cy - person.eye_dist() * 1.1))
    cv2.circle(frame, center, r, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.ellipse(frame, center, (r, r), -90, 0, int(360 * progress), (255, 255, 255), 3, cv2.LINE_AA)


def extract_face_data(landmarks_list, w, h):
    """Convert MediaPipe landmarks into pixel coordinates for irises, eyelids, and face boxes."""
    parsed = []
    for lm in landmarks_list:
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        left_iris = pts[LEFT_IRIS_CENTER]
        right_iris = pts[RIGHT_IRIS_CENTER]
        left_r = float(np.mean(np.linalg.norm(pts[LEFT_IRIS_RIM] - left_iris, axis=1)))
        right_r = float(np.mean(np.linalg.norm(pts[RIGHT_IRIS_RIM] - right_iris, axis=1)))
        left_lid = pts[LEFT_EYE_OUTLINE].astype(np.int32)
        right_lid = pts[RIGHT_EYE_OUTLINE].astype(np.int32)

        x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
        x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        parsed.append((left_iris, right_iris, left_r, right_r, left_lid, right_lid, bbox))
    return parsed


def apply_filter(frame, face_data, people, textures, now) -> None:
    if not face_data:
        match_people(people, [], now)
        return

    f_w = frame.shape[1]
    eye_pairs = [(d[0], d[1]) for d in face_data]
    matched = match_people(people, eye_pairs, now)

    for data, person in zip(face_data, matched):
        _, _, left_r, right_r, left_lid, right_lid, bbox = data
        eye_dist = person.eye_dist()
        if eye_dist < 5:
            continue

        distance_m = CALIBRATION_K / eye_dist
        offset_m = abs(person.center()[0] - f_w / 2) * REAL_EYE_DIST_M / eye_dist

        if distance_m > MAX_DIST_M or offset_m > MAX_HORIZ_OFFSET_M:
            person.in_zone_since, person.color = None, None
            blur_face(frame, bbox)
            continue

        if person.in_zone_since is None:
            person.in_zone_since = now
        held = now - person.in_zone_since
        if held < HOLD_SECONDS:
            draw_hold_ring(frame, person, held / HOLD_SECONDS)
            continue

        if person.color is None:
            person.color = random.choice(COLOR_KEYS)

        dx, dy = person.right - person.left
        angle = math.degrees(math.atan2(dy, dx))
        fallback_r = eye_dist * IRIS_RADIUS_RATIO
        r_left = left_r if left_r > 2.0 else fallback_r
        r_right = right_r if right_r > 2.0 else fallback_r

        overlay_iris(frame, textures[person.color], person.left, r_left, angle, left_lid)
        overlay_iris(frame, textures[person.color], person.right, r_right, angle, right_lid)


def main():
    load_model()
    textures = load_iris_textures()

    with open(TASK_PATH, "rb") as f:
        model_buffer = f.read()

    base_options = mp_python.BaseOptions(model_asset_buffer=model_buffer)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    people = []
    print("Running. Press 'q' or Esc to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        face_data = extract_face_data(result.face_landmarks, w, h) if result.face_landmarks else []
        apply_filter(frame, face_data, people, textures, time.monotonic())

        cv2.imshow("Hallway Iris Filter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if cv2.getWindowProperty("Hallway Iris Filter", cv2.WND_PROP_VISIBLE) < 1:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()