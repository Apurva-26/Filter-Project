"""
features.py - shared landmark + feature code for the Gun Gun Shoot game.

EVERY other script (collect.py, train.py, live_test.py, game.py) imports this
file. That way the training data and live play are processed in exactly the
same way. If you want to change how features are computed, change it HERE only.

What's in this file, in the order the data flows through it:
  1. LandmarkExtractor.process(frame)
        - runs MediaPipe Pose on the whole camera frame
        - crops a box around each wrist and runs MediaPipe Hand on each crop
        - maps the hand points back to full-frame pixel coordinates
  2. to_raw_row(...)  -> flattens those landmarks into one row of numbers (this is what goes in the CSV)
  3. normalize(row)   -> turns a raw row into the numbers the MLP actually sees
  4. draw(...)        -> draws the skeleton on a frame so you can see what's being detected

Setup:
  pip install mediapipe opencv-python numpy
  Download these two model files into a folder called "models/" next to this file:
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Coordinate conventions used everywhere in this file:
  - x, y are in PIXELS of the full camera frame. (0, 0) is the top-left corner.
    y gets BIGGER as you go DOWN the image.
  - z is MediaPipe's rough depth estimate, converted to roughly the same pixel
    scale as x. Smaller z = closer to the camera. It's noisy, so we use it sparingly.
  - "left" / "right" always mean the PERSON's left/right, not the image's.
    (On an unflipped camera image, the person's left hand appears on the right side.)
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

# =============================================================================
# Classes (what the MLP predicts) and the game move each one counts as
# =============================================================================

CLASSES = [
    "reload",
    "shoot_one",
    "shoot_two",
    "shield",
    "mirror_flat",
    "mirror_fingers",
    "none"
]

CLASS_TO_MOVE = {
    "reload": "reload",
    "shoot_one": "shoot",
    "shoot_two": "shoot",
    "shield": "shield",
    "mirror_flat": "mirror",
    "mirror_fingers": "mirror",
    "none": None,  # ignored by the game
}

# =============================================================================
# Which landmarks we keep
# =============================================================================

# MediaPipe Pose gives 33 points. We only keep these (name -> MediaPipe index).
# Face points (1-10) and legs (25-32) are thrown away. Hips are kept in the raw
# CSV (useful to check framing later) but are NOT fed to the MLP, because the
# camera may not always see them.
POSE_POINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

# The pose points that actually become MLP features.
POSE_FEATURE_POINTS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

# Extra pose points that are kept in FrameLandmarks.pose but NOT saved to the CSV and
# NOT fed to the MLP (so adding them doesn't change the training data at all).
# The eyes are used by distance.py to estimate how far away the player is.
EXTRA_POSE_POINTS = {
    "left_eye": 2,
    "right_eye": 5,
}

HAND_SIDES = ("left", "right")
NUM_HAND_POINTS = 21  # MediaPipe Hand: 0 = wrist, then 4 points per finger

# Pairs of hand points to connect with lines when drawing (thumb, index, middle, ring, pinky, palm).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# =============================================================================
# Tuning knobs for the hand crop
# =============================================================================

# Size of the square box around each hand, as a multiple of shoulder width.
# Bigger = less likely to cut off fingers, but the hand is smaller inside the crop.
CROP_SCALE = 1.1

# The hand sits PAST the wrist (further from the elbow), so we shift the box
# center along the elbow->wrist direction by this fraction of the forearm length.
CROP_SHIFT = 0.35

# If the hand found in a crop has its wrist further than this (x shoulder width)
# from where the pose model thinks the wrist is, it's probably the OTHER hand,
# so we reject it.
MAX_WRIST_MISMATCH = 0.5

# If both crops end up returning the same physical hand (possible during mirror,
# when the hands touch), their wrists will be almost on top of each other.
DUPLICATE_HAND_DIST = 0.1  # x shoulder width

# Ignore people whose shoulders are less than this many pixels apart (too far away).
MIN_SHOULDER_WIDTH_PX = 40

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POSE_MODEL = os.path.join(HERE, "models", "pose_landmarker_full.task")
DEFAULT_HAND_MODEL = os.path.join(HERE, "models", "hand_landmarker.task")

# =============================================================================
# What LandmarkExtractor.process() returns
# =============================================================================


@dataclass
class FrameLandmarks:
    # name -> (x, y, z, visibility). None if no person was found.
    pose: dict | None = None
    # "left"/"right" -> numpy array of shape (21, 3) with (x, y, z) per point, or None if not found.
    hands: dict = field(default_factory=lambda: {"left": None, "right": None})
    # "left"/"right" -> (x0, y0, x1, y1) crop box in pixels, or None. Only used for drawing.
    hand_boxes: dict = field(default_factory=lambda: {"left": None, "right": None})

    @property
    def person_found(self) -> bool:
        return self.pose is not None


# =============================================================================
# 1. Running the MediaPipe models
# =============================================================================


class LandmarkExtractor:
    """
    Wraps both MediaPipe models. Create ONE of these and call process() on every frame.

        with LandmarkExtractor() as extractor:
            landmarks = extractor.process(frame)
    """

    def __init__(self, pose_model_path: str = DEFAULT_POSE_MODEL, hand_model_path: str = DEFAULT_HAND_MODEL):
        for path in (pose_model_path, hand_model_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Download it using the links at the top of features.py and put it in the models/ folder."
                )

        vision = mp.tasks.vision
        BaseOptions = mp.tasks.BaseOptions

        # Pose runs in VIDEO mode: it uses the previous frame to track the
        # person more smoothly, but needs an increasing timestamp each call.
        # num_poses=3 so we can see several people and then pick the closest one.
        self.pose_model = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=3,
            )
        )

        # Hand runs in IMAGE mode, because each crop is a new, separate little
        # image (the box moves every frame). num_hands=2 because during mirror
        # BOTH hands end up inside each crop; we pick the right one ourselves.
        self.hand_model = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
            )
        )

        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        """Find the person and their hands in one camera frame (as OpenCV gives it: BGR, unflipped)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB, OpenCV gives BGR

        # --- Step 1: pose on the full frame ---
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        pose_result = self.pose_model.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )

        pose = self._pick_closest_person(pose_result.pose_landmarks, w, h)
        if pose is None:
            return FrameLandmarks()

        shoulder_width = _shoulder_width(pose)

        # --- Steps 2 + 3: crop around each wrist, run the hand model, map back ---
        result = FrameLandmarks(pose=pose)
        for side in HAND_SIDES:
            box = self._hand_box(pose, side, shoulder_width, w, h)
            result.hand_boxes[side] = box
            if box is not None:
                result.hands[side] = self._find_hand_in_box(rgb, box, pose[f"{side}_wrist"], shoulder_width)

        self._remove_duplicate_hand(result.hands, pose, shoulder_width)
        return result

    # --- helpers for process() ---

    def _pick_closest_person(self, all_people, w: int, h: int) -> dict | None:
        """Out of everyone detected, keep the person with the widest shoulders (= closest to the camera)."""
        best_pose, best_width = None, 0.0
        for person in all_people:
            # Convert MediaPipe's 0-1 values into pixels. z uses the same scale as x.
            pose = {
                name: (person[i].x * w, person[i].y * h, person[i].z * w, person[i].visibility)
                for name, i in {**POSE_POINTS, **EXTRA_POSE_POINTS}.items()
            }
            width = _shoulder_width(pose)
            if width > best_width:
                best_pose, best_width = pose, width

        if best_width < MIN_SHOULDER_WIDTH_PX:
            return None
        return best_pose

    def _hand_box(self, pose: dict, side: str, shoulder_width: float, w: int, h: int):
        """Square crop box around where this side's hand should be. Returns (x0, y0, x1, y1) or None."""
        wx, wy = pose[f"{side}_wrist"][:2]
        ex, ey = pose[f"{side}_elbow"][:2]

        # Push the center past the wrist, away from the elbow.
        cx = wx + CROP_SHIFT * (wx - ex)
        cy = wy + CROP_SHIFT * (wy - ey)
        half = CROP_SCALE * shoulder_width / 2

        # Clamp to the frame edges.
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
        if x1 - x0 < 20 or y1 - y0 < 20:  # box is (almost) entirely off-screen
            return None
        return (x0, y0, x1, y1)

    def _find_hand_in_box(self, rgb: np.ndarray, box, pose_wrist, shoulder_width: float):
        """Run the hand model on one crop. Returns a (21, 3) array in full-frame pixels, or None."""
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])  # MediaPipe needs a contiguous array
        hand_result = self.hand_model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))

        crop_w, crop_h = x1 - x0, y1 - y0
        best_points, best_dist = None, math.inf
        for hand in hand_result.hand_landmarks:
            # The hand model gives 0-1 values relative to the CROP.
            # Convert back to full-frame pixels: offset by the crop's corner.
            points = np.array(
                [[x0 + p.x * crop_w, y0 + p.y * crop_h, p.z * crop_w] for p in hand],
                dtype=np.float32,
            )
            # How far is this hand's wrist from where the pose model put the wrist?
            dist = math.hypot(points[0, 0] - pose_wrist[0], points[0, 1] - pose_wrist[1])
            if dist < best_dist:
                best_points, best_dist = points, dist

        if best_dist > MAX_WRIST_MISMATCH * shoulder_width:
            return None  # the closest hand is too far away; probably the other hand
        return best_points

    def _remove_duplicate_hand(self, hands: dict, pose: dict, shoulder_width: float):
        """If both sides grabbed the same physical hand, keep it only on the side it fits better."""
        left, right = hands["left"], hands["right"]
        if left is None or right is None:
            return
        wrist_gap = math.hypot(left[0, 0] - right[0, 0], left[0, 1] - right[0, 1])
        if wrist_gap > DUPLICATE_HAND_DIST * shoulder_width:
            return  # two genuinely different hands

        def mismatch(side, points):
            px, py = pose[f"{side}_wrist"][:2]
            return math.hypot(points[0, 0] - px, points[0, 1] - py)

        if mismatch("left", left) > mismatch("right", right):
            hands["left"] = None
        else:
            hands["right"] = None

    def close(self):
        self.pose_model.close()
        self.hand_model.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _shoulder_width(pose: dict) -> float:
    lx, ly = pose["left_shoulder"][:2]
    rx, ry = pose["right_shoulder"][:2]
    return math.hypot(lx - rx, ly - ry)


# =============================================================================
# 2. Raw rows (what's saved in the CSV)
# =============================================================================


def _build_raw_columns() -> list[str]:
    cols = ["frame_w", "frame_h"]
    for name in POSE_POINTS:
        cols += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]
    for side in HAND_SIDES:
        cols.append(f"{side}_hand_found")
        for i in range(NUM_HAND_POINTS):
            cols += [f"{side}_h{i}_x", f"{side}_h{i}_y", f"{side}_h{i}_z"]
    return cols


# Column names for a raw row, in order. collect.py writes these as the CSV header.
RAW_COLUMNS = _build_raw_columns()
_COL = {name: i for i, name in enumerate(RAW_COLUMNS)}  # column name -> position


def to_raw_row(landmarks: FrameLandmarks, frame_w: int, frame_h: int) -> list[float]:
    """Flatten one frame's landmarks into a list of numbers matching RAW_COLUMNS.
    Only call this when landmarks.person_found is True."""
    row = [float(frame_w), float(frame_h)]
    for name in POSE_POINTS:
        row += list(landmarks.pose[name])  # x, y, z, visibility
    for side in HAND_SIDES:
        points = landmarks.hands[side]
        if points is None:
            row.append(0.0)  # hand_found = 0
            row += [0.0] * (NUM_HAND_POINTS * 3)  # placeholder zeros so every row is the same length
        else:
            row.append(1.0)
            row += points.flatten().tolist()
    return row


# =============================================================================
# 3. Normalization: raw row -> MLP features
# =============================================================================


def _build_feature_names() -> list[str]:
    names = []
    for name in POSE_FEATURE_POINTS:
        names += [f"{name}_bx", f"{name}_by"]  # "b" = body frame
    for side in HAND_SIDES:
        names += [f"{side}_found", f"{side}_wrist_bx", f"{side}_wrist_by"]
        for i in range(1, NUM_HAND_POINTS):  # skip point 0 (wrist): it's always (0, 0, 0) in the hand frame
            names += [f"{side}_h{i}_lx", f"{side}_h{i}_ly", f"{side}_h{i}_lz"]  # "l" = hand-local frame
    return names


# Names of the numbers normalize() returns, in order (useful for debugging the MLP).
FEATURE_NAMES = _build_feature_names()


def normalize(raw_row) -> np.ndarray:
    """
    Turn one raw row (list/array in RAW_COLUMNS order) into the MLP's input.

    Two kinds of normalization:

    BODY FRAME (pose points and each hand's wrist):
        Move the origin to the midpoint between the shoulders, then divide by
        shoulder width. Example: right_wrist_by = -1.3 means "the right wrist is
        1.3 shoulder-widths ABOVE the shoulder line" (negative = up, since y points down).
        This removes where the person stands, how tall they are, and how far
        they are from the camera.

    HAND-LOCAL FRAME (finger points):
        For each hand, move the origin to that hand's wrist, then divide by hand
        size (wrist -> base of the middle finger). This describes the SHAPE of the
        hand (gun / fist / flat) no matter where the hand is or how big it looks.
        We use the 3D distance for hand size so a hand pointing at the camera
        (which looks short in 2D) doesn't get blown up.

    A hand that wasn't found becomes all zeros, with its "found" feature = 0.
    """
    r = np.asarray(raw_row, dtype=np.float32)

    def col(name):
        return r[_COL[name]]

    # Body frame origin + scale
    lsx, lsy = col("left_shoulder_x"), col("left_shoulder_y")
    rsx, rsy = col("right_shoulder_x"), col("right_shoulder_y")
    origin_x, origin_y = (lsx + rsx) / 2, (lsy + rsy) / 2
    shoulder_width = max(math.hypot(lsx - rsx, lsy - rsy), 1e-6)

    features = []

    # Pose points in the body frame (x, y only; pose z is too noisy)
    for name in POSE_FEATURE_POINTS:
        features.append((col(f"{name}_x") - origin_x) / shoulder_width)
        features.append((col(f"{name}_y") - origin_y) / shoulder_width)

    # Hands
    per_hand = 3 + (NUM_HAND_POINTS - 1) * 3  # found + wrist(x, y) + 20 points * (x, y, z)
    for side in HAND_SIDES:
        start = _COL[f"{side}_h0_x"]
        points = r[start:start + NUM_HAND_POINTS * 3].reshape(NUM_HAND_POINTS, 3)
        wrist = points[0]
        hand_size = float(np.linalg.norm(points[9] - wrist))  # point 9 = base of the middle finger

        if col(f"{side}_hand_found") < 0.5 or hand_size < 1e-6:
            features += [0.0] * per_hand
            continue

        features.append(1.0)
        features.append((wrist[0] - origin_x) / shoulder_width)
        features.append((wrist[1] - origin_y) / shoulder_width)
        local = (points[1:] - wrist) / hand_size  # shape (20, 3)
        features += local.flatten().tolist()

    return np.array(features, dtype=np.float32)


def features_from_landmarks(landmarks: FrameLandmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """Shortcut for live play: landmarks -> raw row -> features, in one call."""
    return normalize(to_raw_row(landmarks, frame_w, frame_h))


# =============================================================================
# 4. Drawing (for display only; never affects the data)
# =============================================================================

POSE_LINES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]


def draw(frame_bgr: np.ndarray, landmarks: FrameLandmarks, show_boxes: bool = True) -> None:
    """Draw the arm skeleton, hand points and crop boxes onto the frame (in place).
    Draw on the UNFLIPPED frame, then flip it for display if you want a mirror view."""
    if not landmarks.person_found:
        return

    def pt(xy):
        return int(xy[0]), int(xy[1])

    # Arms: white lines, yellow joints
    for a, b in POSE_LINES:
        cv2.line(frame_bgr, pt(landmarks.pose[a]), pt(landmarks.pose[b]), (255, 255, 255), 2)
    for name in POSE_FEATURE_POINTS:
        cv2.circle(frame_bgr, pt(landmarks.pose[name]), 5, (0, 255, 255), -1)

    # Hands: green for left, magenta for right
    colors = {"left": (0, 255, 0), "right": (255, 0, 255)}
    for side in HAND_SIDES:
        if show_boxes and landmarks.hand_boxes[side] is not None:
            x0, y0, x1, y1 = landmarks.hand_boxes[side]
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), colors[side], 1)
        points = landmarks.hands[side]
        if points is None:
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pt(points[a]), pt(points[b]), colors[side], 2)
        for p in points:
            cv2.circle(frame_bgr, pt(p), 3, colors[side], -1)
