"""
collect.py - guided training-data collection for the pose classifier.

How a session works:
  1. You (the operator) run this script and enter a person ID like "p07" (no real names).
  2. The participant stands in front of the camera. Press SPACE to start.
  3. For every pose, the screen shows "Next: <POSE>" with a 3-2-1 countdown, then
     records for a few seconds, then says "Relax!" and repeats. Each pose is done
     TAKES_PER_POSE times, then the "none" prompts run at the end.
  4. After every finished take, everything recorded so far is saved to
     data/raw/<person_id>_<date-time>.csv, so quitting early never loses data.

Each CSV row is ONE camera frame:
  person_id, label, take, t_ms, then all the raw landmark columns from features.RAW_COLUMNS.
  Only RAW landmarks are saved. train.py normalizes them with features.normalize(),
  so if the normalization changes later, nobody has to be re-recorded.

Usage:
  python collect.py                  (asks for the person ID)
  python collect.py --person p07
  python collect.py --person p07 --camera 1

Controls (click the video window first so it gets the key presses):
  SPACE  start / pause / resume (pausing throws away the take in progress)
  R      redo the last finished take
  Q      quit
"""

import argparse
import csv
import math
import os
import re
import time
from datetime import datetime

import cv2

import features

# =============================================================================
# Session settings (change these to make sessions longer/shorter)
# =============================================================================

TAKES_PER_POSE = 3      # how many times each pose is recorded
COUNTDOWN_S = 3.0       # "Next: RELOAD  3.. 2.. 1.."
RECORD_S = 3.0          # how long each take records for
REST_S = 1.5            # "Relax!" between takes, so they drop their hands and re-form the pose
MIN_ROWS_PER_TAKE = 15  # if fewer frames than this had a person in them, the take is redone automatically

# (class label, text shown on screen). Order here = order in the session.
POSE_STEPS = [
    ("reload", "RELOAD"),
    ("shoot_one", "SHOOT (one hand)"),
    ("shoot_two", "SHOOT (two hands)"),
    ("shield", "SHIELD"),
    ("mirror_flat", "MIRROR (flat hands)"),
    ("mirror_fingers", "MIRROR (fingers)"),
]

# The "none" class gets its own prompts, each recorded once, at the end of the session.
NONE_STEPS = [
    ("none", "Stand normally"),
    ("none", "Slap your thighs"),
    ("none", "Move slowly between poses"),
    ("none", "Move your hands around"),
]

WINDOW = "Gun Gun Shoot - data collection"
OUTPUT_DIR = os.path.join(features.HERE, "data", "raw")
CSV_HEADER = ["person_id", "label", "take", "t_ms"] + features.RAW_COLUMNS

# Colors (OpenCV uses BGR, not RGB)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 230), (0, 220, 255)


def build_steps() -> list[dict]:
    """The full list of takes for one session, in order."""
    steps = []
    for label, text in POSE_STEPS:
        for take in range(1, TAKES_PER_POSE + 1):
            steps.append({"label": label, "text": text, "take": take})
    for take, (label, text) in enumerate(NONE_STEPS, start=1):
        steps.append({"label": label, "text": text, "take": take})
    return steps


def save_csv(path: str, finished_takes: list[list[list]]) -> int:
    """Rewrite the whole CSV with every finished take. Returns the number of rows written.
    (Rewriting is simplest and fine at this size, and it makes 'redo' easy.)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [row for take in finished_takes for row in take]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    return len(rows)


# =============================================================================
# Drawing helpers for the on-screen text
# =============================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, org, scale=1.0, color=WHITE, thickness=2):
    """Text with a black outline so it's readable on any background."""
    cv2.putText(img, text, org, FONT, scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def put_centered(img, text, y, scale=1.0, color=WHITE, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    put_text(img, text, ((img.shape[1] - tw) // 2, y), scale, color, thickness)


def draw_overlay(img, state, step, step_i, n_steps, elapsed, person_id, landmarks, fps, notice):
    """Everything drawn on top of the (already flipped) camera image."""
    h, w = img.shape[:2]

    # --- Top bar: who, which step, and how fast the pipeline is running ---
    put_text(img, f"Person: {person_id}", (15, 35), 0.8)
    put_text(img, f"Step {min(step_i + 1, n_steps)}/{n_steps}", (w - 190, 35), 0.8)
    put_text(img, f"{fps:4.1f} FPS", (15, 70), 0.6, YELLOW, 1)

    # --- Middle: depends on the state ---
    if state == "waiting":
        put_centered(img, "Press SPACE to start", h // 2, 1.4)
    elif state == "paused":
        put_centered(img, "PAUSED - press SPACE to resume", h // 2, 1.2, YELLOW)
    elif state == "countdown":
        put_centered(img, f"Next: {step['text']}", h // 2 - 60, 1.5)
        put_centered(img, str(math.ceil(COUNTDOWN_S - elapsed)), h // 2 + 60, 3.0, YELLOW, 5)
    elif state == "recording":
        cv2.circle(img, (40, 115), 14, RED, -1)
        put_text(img, f"RECORDING: {step['text']}", (65, 127), 1.0, RED)
        # Progress bar showing how much of the take is left
        bar_w = int((w - 30) * min(elapsed / RECORD_S, 1.0))
        cv2.rectangle(img, (15, 145), (15 + bar_w, 160), RED, -1)
    elif state == "rest":
        put_centered(img, "Relax!", h // 2, 2.0, GREEN, 4)
    elif state == "done":
        put_centered(img, "DONE! Thank you!", h // 2 - 20, 1.6, GREEN, 4)
        put_centered(img, "Press Q to quit (or R to redo the last take)", h // 2 + 40, 0.8)

    if notice:
        put_centered(img, notice, h // 2 + 140, 0.9, YELLOW)

    # --- Bottom: are both hands being found? (person's left hand is on the left in the mirrored view) ---
    if not landmarks.person_found:
        put_centered(img, "NO PERSON DETECTED", h - 25, 1.0, RED)
    else:
        for side, x in (("left", 15), ("right", w - 280)):
            found = landmarks.hands[side] is not None
            put_text(img, f"{side.upper()} HAND: {'OK' if found else '--'}", (x, h - 25), 0.9, GREEN if found else RED)


# =============================================================================
# Main loop
# =============================================================================


def main(steps: list[dict] | None = None, file_suffix: str = ""):
    """steps: the takes to record (defaults to build_steps(), the full pose session).
    file_suffix: added to the CSV name, e.g. "_none" -> p07_none_<date-time>.csv.
    Other scripts (like collect_none.py) call this with their own steps."""
    parser = argparse.ArgumentParser(description="Record labeled pose data for the Gun Gun Shoot classifier.")
    parser.add_argument("--person", help='person ID, e.g. "p07" (letters, numbers, _ and - only)')
    parser.add_argument("--camera", type=int, default=0, help="camera index (try 1 if 0 is the wrong camera)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    person_id = args.person or input('Person ID (e.g. "p07"): ').strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", person_id):
        raise SystemExit("Person ID can only contain letters, numbers, _ and -")

    out_path = os.path.join(OUTPUT_DIR, f"{person_id}{file_suffix}_{datetime.now():%Y%m%d-%H%M%S}.csv")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    steps = steps or build_steps()

    # --- Session state ---
    # state is one of: waiting -> countdown -> recording -> rest -> countdown -> ... -> done
    # (and "paused" whenever SPACE is pressed mid-session)
    state = "waiting"
    state_start = time.monotonic()
    step_i = 0                # which take we're on (index into steps)
    current_rows = []         # rows of the take currently being recorded
    finished_takes = []       # one list of rows per finished take
    notice, notice_until = "", 0.0
    fps, last_frame_time = 0.0, time.monotonic()

    def go(new_state):
        nonlocal state, state_start
        state, state_start = new_state, time.monotonic()

    def flash(text, seconds=2.5):
        nonlocal notice, notice_until
        notice, notice_until = text, time.monotonic() + seconds

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with features.LandmarkExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped giving frames.")
                break
            frame_h, frame_w = frame.shape[:2]

            # 1. Detect landmarks on the UNFLIPPED frame (this is what gets saved)
            landmarks = extractor.process(frame)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_frame_time, 1e-6))  # smoothed
            last_frame_time = now
            elapsed = now - state_start
            step = steps[min(step_i, len(steps) - 1)]

            # 2. Advance the session
            if state == "countdown" and elapsed >= COUNTDOWN_S:
                current_rows = []
                go("recording")

            elif state == "recording":
                if landmarks.person_found:
                    row = [person_id, step["label"], step["take"], int(elapsed * 1000)]
                    row += features.to_raw_row(landmarks, frame_w, frame_h)
                    current_rows.append(row)

                if elapsed >= RECORD_S:
                    if len(current_rows) < MIN_ROWS_PER_TAKE:
                        flash("Couldn't see you well enough - redoing that take")
                        go("countdown")
                    else:
                        finished_takes.append(current_rows)
                        n = save_csv(out_path, finished_takes)
                        print(f"Saved take: {step['label']} #{step['take']} ({len(current_rows)} frames, {n} total)")
                        step_i += 1
                        go("done" if step_i >= len(steps) else "rest")

            elif state == "rest" and elapsed >= REST_S:
                go("countdown")

            # 3. Draw: skeleton first (on the unflipped frame), then flip, then text
            features.draw(frame, landmarks)
            display = cv2.flip(frame, 1)  # mirror view so it feels natural to the participant
            draw_overlay(
                display, state, step, step_i, len(steps), time.monotonic() - state_start,
                person_id, landmarks, fps, notice if time.monotonic() < notice_until else "",
            )
            cv2.imshow(WINDOW, display)

            # 4. Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if state in ("waiting", "paused"):
                    go("countdown")
                elif state in ("countdown", "recording", "rest"):
                    current_rows = []  # throw away the unfinished take
                    go("paused")
            elif key == ord("r") and finished_takes:
                dropped = finished_takes.pop()
                save_csv(out_path, finished_takes)
                step_i -= 1
                flash(f"Redoing: {steps[step_i]['text']}")
                print(f"Removed last take ({len(dropped)} frames); redoing it.")
                current_rows = []
                go("countdown")

    cap.release()
    cv2.destroyAllWindows()

    if finished_takes:
        print(f"\n{len(finished_takes)}/{len(steps)} takes saved to {out_path}")
    else:
        print("\nNothing was recorded.")


if __name__ == "__main__":
    main()
