"""
collect_shoot.py - record ONLY extra "shoot" data (about 17 seconds in total).

Why this exists: the live game struggles to detect shoots, so this adds more
examples of just that move.

This runs the exact same guided session as collect.py (same screen, same keys,
same CSV format), with five short takes instead of the full list of poses.
Splitting the time into takes makes you drop your hands and form the pose
again, which gives the model more variety than holding one pose the whole time.

Files are saved as data/raw/<person_id>_shoot_<date-time>.csv, and train.py
picks them up automatically along with everything else.

Usage:
  py collect_shoot.py --person 1       (use the SAME ID you recorded under before)
  py collect_shoot.py --person 1 --camera 1

Controls: same as collect.py (SPACE start/pause, R redo last take, Q quit)

Afterwards, run train.py so the model actually learns from the new data.
"""

import collect

SECONDS_PER_TAKE = 3.5  # how long each take records for
TWO_HAND_TAKES = 3      # two hands gets more takes than right or left (one each)

# (class label, text shown on screen). Left and right hand are both "shoot_one".
SHOOT_PROMPTS = [
    ("shoot_one", "SHOOT (right hand)"),
    ("shoot_one", "SHOOT (left hand)"),
] + [("shoot_two", "SHOOT (two hands)")] * TWO_HAND_TAKES


def build_shoot_steps() -> list[dict]:
    return [
        {"label": label, "text": text, "take": take}
        for take, (label, text) in enumerate(SHOOT_PROMPTS, start=1)
    ]


if __name__ == "__main__":
    collect.RECORD_S = SECONDS_PER_TAKE
    collect.main(steps=build_shoot_steps(), file_suffix="_shoot")
