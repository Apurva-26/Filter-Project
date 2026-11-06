"""
distance.py - estimates how far the player is from the camera, using their eyes.

The idea: things look smaller the farther away they are. Twice as far = half as big.
The distance between a person's eyes is almost the same for everyone (about 6.3 cm),
so if we measure it in PIXELS we can work out the distance:

    distance (m) = EYE_PX_AT_1M / eye distance in pixels

EYE_PX_AT_1M = how many pixels apart the eyes are when someone stands exactly 1 m away.
That number depends on the CAMERA (its lens / zoom), so it's measured once with a
calibration: stand at a known distance, and we do the maths backwards:

    EYE_PX_AT_1M = eye distance in pixels x distance you stood at

Calibrate ONCE PER CAMERA (once for the laptop webcam, once for the Pi's USB camera after
it's mounted). Not once per player. You only need to redo it if you switch cameras or
change the camera's zoom. Changing the resolution (e.g. 1280x720 -> 640x360) is handled
automatically. Moving the camera to a different spot doesn't matter either.

To calibrate:
  py distance.py                 stand 1.24 m away (default) and press SPACE
  py distance.py --distance 1.0  if you'd rather stand somewhere else
Measure from the camera lens to your eyes, straight out. After saving, the screen shows
your estimated distance live, so walk to 1.0 m and 1.5 m (tape marks on the floor) and
check it says roughly the right number.

Accuracy: about +/-10%. Turning your head makes the eyes look closer together, which
reads as "farther away", so it's best when the player faces the screen (which they do).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import deque

import cv2

import features

CALIBRATION_PATH = os.path.join(features.HERE, "data", "distance_calibration.json")

# =============================================================================
# Settings
# =============================================================================

MIN_DISTANCE_M = 1.0     # closer than this -> "Step back"
MAX_DISTANCE_M = 1.5     # farther than this -> "Come closer"
ZONE_MARGIN_M = 0.05     # during a game, allow this much extra each way so it doesn't
                         # flicker in and out when someone stands right on the edge
SMOOTHING_FRAMES = 7     # use the median of this many frames (removes jitter)
MIN_EYE_VISIBILITY = 0.5 # ignore the eyes if MediaPipe isn't confident it can see them

# Used ONLY if you haven't calibrated yet: a guess based on a typical webcam
# (about 70 degrees wide). Calibrating is much more accurate.
FALLBACK_HFOV_DEG = 70.0
EYE_SPACING_M = 0.063


def eye_distance_px(landmarks) -> float | None:
    """Pixel distance between the two eyes, or None if they can't be seen."""
    if not landmarks.person_found:
        return None
    lx, ly, _, lv = landmarks.pose["left_eye"]
    rx, ry, _, rv = landmarks.pose["right_eye"]
    if min(lv, rv) < MIN_EYE_VISIBILITY:
        return None
    return math.hypot(lx - rx, ly - ry)


def load_calibration():
    """The saved calibration ({"eye_px_at_1m": ..., "frame_width": ...}) or None."""
    try:
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class DistanceEstimator:
    """Call update() once per frame; it returns the smoothed distance in metres (or None)."""

    def __init__(self, frame_width: int):
        cal = load_calibration()
        self.calibrated = cal is not None
        if cal:
            # Calibrated at a different resolution? Pixels scale with the frame width.
            self.eye_px_at_1m = cal["eye_px_at_1m"] * frame_width / cal["frame_width"]
        else:
            # Pinhole camera maths: focal length in pixels = (width / 2) / tan(field of view / 2)
            focal_px = (frame_width / 2) / math.tan(math.radians(FALLBACK_HFOV_DEG / 2))
            self.eye_px_at_1m = focal_px * EYE_SPACING_M
            print("distance.py: no calibration found, distances are a rough guess. Run: py distance.py")
        self.recent = deque(maxlen=SMOOTHING_FRAMES)

    def update(self, landmarks) -> float | None:
        px = eye_distance_px(landmarks)
        if px is None or px < 1:
            self.recent.clear()   # lost the person: start fresh
            return None
        self.recent.append(self.eye_px_at_1m / px)
        return statistics.median(self.recent)


def zone(distance_m: float | None, margin: float = 0.0) -> str:
    """'too_close', 'too_far', 'ok', or 'unknown' if there's no distance."""
    if distance_m is None:
        return "unknown"
    if distance_m < MIN_DISTANCE_M - margin:
        return "too_close"
    if distance_m > MAX_DISTANCE_M + margin:
        return "too_far"
    return "ok"


# =============================================================================
# Calibration screen:  py distance.py
# =============================================================================

CALIBRATION_RECORD_S = 3.0   # measure for this long and take the median


def calibrate():
    from collect import GREEN, RED, WHITE, YELLOW, put_centered, put_text

    parser = argparse.ArgumentParser(description="Calibrate the distance check for this camera.")
    parser.add_argument("--distance", type=float, default=1.24, help="how far you'll stand, in metres")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera gave no frame.")
    frame_w = frame.shape[1]

    window = "Distance calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    estimator = DistanceEstimator(frame_w)
    recording_since, samples, message = None, [], ""

    with features.LandmarkExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            landmarks = extractor.process(frame)
            px = eye_distance_px(landmarks)
            dist = estimator.update(landmarks)

            if recording_since is not None:
                if px:
                    samples.append(px)
                if time.monotonic() - recording_since >= CALIBRATION_RECORD_S:
                    recording_since = None
                    if len(samples) < 10:
                        message = "Couldn't see your eyes well enough. Try again."
                    else:
                        eye_px = statistics.median(samples)
                        cal = {"eye_px_at_1m": eye_px * args.distance, "frame_width": frame_w,
                               "measured_at_m": args.distance, "eye_px": eye_px}
                        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
                        with open(CALIBRATION_PATH, "w") as f:
                            json.dump(cal, f, indent=1)
                        estimator = DistanceEstimator(frame_w)  # use the new calibration right away
                        message = "Saved! Now walk around and check the distance below."
                        print(f"Saved {CALIBRATION_PATH}: {cal}")

            features.draw(frame, landmarks, show_boxes=False)
            display = cv2.flip(frame, 1)
            h = display.shape[0]
            put_text(display, f"Stand {args.distance:.2f} m from the camera (lens to eyes), face it,",
                     (20, 40), 0.8, WHITE)
            put_text(display, "then press SPACE.   Q = quit", (20, 75), 0.8, WHITE)
            put_text(display, "calibrated" if estimator.calibrated else "NOT calibrated (rough guess)",
                     (20, 110), 0.7, GREEN if estimator.calibrated else YELLOW)
            if recording_since is not None:
                put_centered(display, "Measuring... hold still", h // 2, 1.5, YELLOW, 3)
            elif message:
                put_centered(display, message, h // 2, 1.0, YELLOW, 2)
            if dist is None:
                put_centered(display, "Can't see your eyes", h - 40, 1.2, RED, 3)
            else:
                z = zone(dist)
                color = GREEN if z == "ok" else YELLOW
                put_centered(display, f"{dist:.2f} m  ({z})   eyes {px:.0f} px apart", h - 40, 1.2, color, 3)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and recording_since is None:
                recording_since, samples, message = time.monotonic(), [], ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate()
