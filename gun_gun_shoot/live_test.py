"""
live_test.py - try the trained classifier live, with the same lock-in logic the game will use.

Two things happen on screen:
  1. ALWAYS: live predictions. Bars on the right show how confident the model is
     in each game move, and the bottom says "I see: SHOOT 91%".
  2. When you press SPACE: a practice round. "GUN... GUN... SHOOT!" appears, the
     capture window opens, and the MoveLocker decides what you did. The result
     ("LOCKED: SHOOT" or "NO MOVE DETECTED") stays on screen for a moment.

Use this to check the classifier BEFORE building the game: try every pose,
try switching poses mid-window, try standing still, try a friend.

Usage:
  py live_test.py
  py live_test.py --camera 1

Controls: SPACE = start a practice round, Q = quit
"""

import argparse
import time

import cv2

import features
from classifier import (
    CAPTURE_WINDOW_S, CONF_THRESHOLD, GRACE_S, GUN_GAP_S, MOVES, SHOOT_AT_S, MoveLocker, PoseClassifier,
    confident_move,
)
from collect import GREEN, RED, WHITE, YELLOW, put_centered, put_text

# The classifier, lock-in logic and their settings live in classifier.py (shared with game.py).
RESULT_S = 2.0  # how long the result stays on screen
WINDOW = "Gun Gun Shoot - live test"


# =============================================================================
# Drawing
# =============================================================================


def draw_move_bars(img, move_probs: dict):
    """Confidence bar for each move on the right side. The threshold is the yellow line."""
    w = img.shape[1]
    x0, bar_max = w - 330, 200
    for i, move in enumerate(MOVES):
        y = 100 + i * 40
        p = move_probs.get(move, 0.0) if move_probs else 0.0
        color = GREEN if p >= CONF_THRESHOLD and move != "none" else WHITE
        put_text(img, move, (x0, y + 18), 0.6, color, 1)
        cv2.rectangle(img, (x0 + 100, y), (x0 + 100 + bar_max, y + 22), (60, 60, 60), -1)
        cv2.rectangle(img, (x0 + 100, y), (x0 + 100 + int(bar_max * p), y + 22), color, -1)
    tx = x0 + 100 + int(bar_max * CONF_THRESHOLD)
    cv2.line(img, (tx, 95), (tx, 100 + len(MOVES) * 40 - 15), YELLOW, 2)


# =============================================================================
# Main loop
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Live-test the Gun Gun Shoot pose classifier.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    classifier = PoseClassifier()
    locker = MoveLocker()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    # Round state: idle -> beat ("GUN.. GUN..") -> window ("SHOOT!") -> result -> idle
    state, state_start = "idle", time.monotonic()
    result_text, result_color = "", WHITE
    fps, last_frame_time = 0.0, time.monotonic()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with features.LandmarkExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped giving frames.")
                break
            frame_h, frame_w = frame.shape[:2]

            # 1. Landmarks + prediction for this frame
            landmarks = extractor.process(frame)
            if landmarks.person_found:
                move_probs = classifier.move_probabilities(landmarks, frame_w, frame_h)
                move, top, conf = confident_move(move_probs)
            else:
                move_probs, move, top, conf = None, None, None, 0.0

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_frame_time, 1e-6))
            last_frame_time = now
            elapsed = now - state_start

            # 2. Advance the practice round
            if state == "beat" and elapsed >= SHOOT_AT_S:
                locker.reset()
                state, state_start = "window", now
            elif state == "window":
                # The first GRACE_S seconds after SHOOT! are ignored: hands are still moving into place
                locked = locker.update(move, now) if elapsed >= GRACE_S else None
                if locked or elapsed >= CAPTURE_WINDOW_S:
                    if locked:
                        result_text, result_color = f"LOCKED: {locked.upper()}  ({elapsed:.2f}s)", GREEN
                    else:
                        result_text, result_color = "NO MOVE DETECTED", RED
                    print(result_text)
                    state, state_start = "result", now
            elif state == "result" and elapsed >= RESULT_S:
                state, state_start = "idle", now

            # 3. Draw
            features.draw(frame, landmarks)
            display = cv2.flip(frame, 1)
            h = display.shape[0]
            elapsed = time.monotonic() - state_start

            put_text(display, f"{fps:4.1f} FPS", (15, 35), 0.6, YELLOW, 1)
            put_text(display, "SPACE = practice round   Q = quit", (15, 65), 0.6, WHITE, 1)
            draw_move_bars(display, move_probs)

            if state == "beat":
                put_centered(display, "GUN..." if elapsed < GUN_GAP_S else "GUN... GUN...", h // 2, 2.0, WHITE, 4)
            elif state == "window":
                put_centered(display, "SHOOT!", h // 2 - 40, 2.5, RED, 6)
                # Hold progress bar: fills up as the current move is held
                bar_w = int(400 * locker.progress())
                x0 = display.shape[1] // 2 - 200
                cv2.rectangle(display, (x0, h // 2 + 10), (x0 + 400, h // 2 + 35), (60, 60, 60), -1)
                cv2.rectangle(display, (x0, h // 2 + 10), (x0 + bar_w, h // 2 + 35), GREEN, -1)
            elif state == "result":
                put_centered(display, result_text, h // 2, 1.5, result_color, 4)

            # Bottom feedback line: what the model thinks right now
            if not landmarks.person_found:
                put_centered(display, "NO PERSON DETECTED", h - 25, 1.0, RED)
            elif move:
                put_centered(display, f"I see: {move.upper()}  {conf:.0%}", h - 25, 1.1, GREEN)
            else:
                put_centered(display, f"I see: ---   (best guess {top} {conf:.0%})", h - 25, 0.9, WHITE)

            cv2.imshow(WINDOW, display)

            # 4. Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and state in ("idle", "result"):
                state, state_start = "beat", time.monotonic()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
