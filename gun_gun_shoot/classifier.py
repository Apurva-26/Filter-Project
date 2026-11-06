"""
classifier.py - turns landmarks into a game move. Shared by live_test.py and game.py.

  PoseClassifier   loads the trained MLP and gives a confidence for each game move
  confident_move() picks the move the player is doing right now (or None if unsure)
  MoveLocker       decides what the player did during the capture window (hold-to-lock)

The timing/threshold settings live here too, so the test tool and the real game
always behave the same way.
"""

import os

import joblib

import gun_gun_shoot.features as features

# =============================================================================
# Settings
# =============================================================================

MODEL_PATH = os.path.join(features.HERE, "models", "pose_classifier.joblib")

CONF_THRESHOLD = 0.75   # a frame only counts as a move if the model is at least this sure

# --- The beat. Timeline of one round (seconds since the first "GUN"):
#
#   0.0          GUN_GAP_S           GUN_GAP_S + SHOOT_GAP_S
#   GUN ........ GUN ............... SHOOT! --[GRACE_S]-- counting frames ... window closes
#                                           ^ hands moving into position, ignored
#
GUN_GAP_S = 0.4         # time between the two "GUN"s
SHOOT_GAP_S = 0.6       # time between the second "GUN" and "SHOOT!"
GRACE_S = 0.3           # after "SHOOT!", ignore the camera this long (hands are still moving)
CAPTURE_WINDOW_S = 1.6  # after "SHOOT!", how long the player has in total to lock a move
                        # (includes GRACE_S, so the real time to lock is CAPTURE_WINDOW_S - GRACE_S)
SHOOT_AT_S = GUN_GAP_S + SHOOT_GAP_S  # when "SHOOT!" happens (don't change this line)

# --- The hold-to-lock meter (see MoveLocker below)
HOLD_S = 0.4            # a move locks once its meter has HOLD_S seconds of "seen it" in it
SWITCH_DRAIN = 2.0      # a frame of a DIFFERENT move empties the other meters this many times faster
UNSURE_DRAIN = 0.25     # an unsure / "none" frame empties the meters at quarter speed
MAX_FRAME_GAP_S = 0.25  # a frame never counts for more than this (in case the camera stutters)

MOVES = ["reload", "shoot", "shield", "mirror", "none"]


# =============================================================================
# The classifier: landmarks -> confidence per game move
# =============================================================================


class PoseClassifier:
    def __init__(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"No model at {path}. Run train.py first.")
        saved = joblib.load(path)
        if saved["feature_names"] != features.FEATURE_NAMES:
            raise SystemExit("features.py has changed since this model was trained. Re-run train.py.")
        self.model = saved["model"]

    def move_probabilities(self, landmarks, frame_w: int, frame_h: int) -> dict:
        """Returns {move: confidence} for every game move.
        Variants of the same move are ADDED together: if the model says
        mirror_flat 50% + mirror_fingers 45%, it's clearly 95% "mirror"."""
        x = features.features_from_landmarks(landmarks, frame_w, frame_h).reshape(1, -1)
        class_probs = self.model.predict_proba(x)[0]
        moves = {m: 0.0 for m in MOVES}
        for cls, p in zip(self.model.classes_, class_probs):
            moves[features.CLASS_TO_MOVE[cls] or "none"] += p
        return moves


def confident_move(move_probs: dict):
    """The move the player is doing right now, or None if the model isn't sure
    (or the top answer is 'none'). Also returns the top move + its confidence for display."""
    top = max(move_probs, key=move_probs.get)
    conf = move_probs[top]
    if top == "none" or conf < CONF_THRESHOLD:
        return None, top, conf
    return top, top, conf


# =============================================================================
# The lock-in logic
# =============================================================================


class MoveLocker:
    """
    Decides what move the player did during the capture window.

    Every move has a "meter" (measured in seconds) that fills up while the player
    holds that move. Feed it one prediction per frame with update():

      frame shows move X (confident)   X's meter      +frame time
                                       other meters   -frame time x SWITCH_DRAIN
      frame is unsure or "none"        every meter    -frame time x UNSURE_DRAIN

    A move LOCKS as soon as its meter reaches HOLD_S. Meters never go below 0.

    "Frame time" is the real time since the previous frame, so this works the same
    at 5 FPS or 30 FPS: it's about SECONDS of holding, not a number of frames.
    So one or two bad frames only cost a little (the meter dips, it doesn't reset),
    while passing through "shoot" on the way to "reload" is wiped out quickly,
    because reload frames drain the shoot meter twice as fast as it filled.

    Once locked, the result never changes (switching poses afterwards is ignored).
    """

    def __init__(self, hold_s: float = HOLD_S):
        self.hold_s = hold_s
        self.reset()

    def reset(self):
        self.locked = None                         # the final answer, once there is one
        self.meters = {m: 0.0 for m in MOVES if m != "none"}
        self.last_time = None                      # time of the previous frame

    def update(self, move, now: float):
        """move = this frame's confident move (or None). Returns the locked move or None."""
        if self.locked:
            return self.locked

        # How much time this frame stands for. The very first frame only starts the clock.
        dt = 0.0 if self.last_time is None else min(now - self.last_time, MAX_FRAME_GAP_S)
        self.last_time = now

        for m in self.meters:
            if move is None:
                self.meters[m] -= dt * UNSURE_DRAIN
            elif m == move:
                self.meters[m] += dt
            else:
                self.meters[m] -= dt * SWITCH_DRAIN
            self.meters[m] = max(self.meters[m], 0.0)

        if move is not None and self.meters[move] >= self.hold_s - 1e-9:  # (1e-9: rounding errors)
            self.locked = move
        return self.locked

    @property
    def leader(self):
        """The move with the fullest meter (or None if all are empty)."""
        if self.locked:
            return self.locked
        best = max(self.meters, key=self.meters.get)
        return best if self.meters[best] > 0 else None

    def progress(self, now: float = None) -> float:
        """0 -> 1: how full the leading move's meter is (for the progress bar)."""
        if self.locked:
            return 1.0
        return min(max(self.meters.values()) / self.hold_s, 1.0)
