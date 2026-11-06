"""
collect_none.py - record ONLY "none" data: things people do that are NOT a game move.

Why this exists: in the first training run, "none" was the weakest class.
Frames of people just standing or moving got mistaken for shoot or reload.
More (and more varied) "none" data is the fix.

This runs the exact same guided session as collect.py (same screen, same keys,
same CSV format), just with a list of everyday non-moves instead of the game poses.
Files are saved as data/raw/<person_id>_none_<date-time>.csv, and train.py
picks them up automatically along with everything else.

Usage:
  py collect_none.py --person p07
  py collect_none.py --person p07 --camera 1

Controls: same as collect.py (SPACE start/pause, R redo last take, Q quit)

Tip: anyone can record this, even people who don't know the game, because
there are no poses to learn. Great for getting lots of different people quickly.
"""

import gun_gun_shoot.collect as collect

TAKES_PER_PROMPT = 2  # each prompt is recorded this many times

# Things people naturally do in front of the screen that are NOT moves.
# Several of these are deliberately CLOSE to real poses (hands raised, hands at
# chest height, hands together). Those "near misses" teach the model the most.
# Don't add anything that IS a real move: e.g. crossing your arms is a shield.
NONE_PROMPTS = [
    "Stand normally",
    "Slap your thighs",
    "Hands on your hips",
    "Hands in your pockets",
    "Hold your hands together in front",
    "Pretend to look at your phone",
    "Wave at the screen",
    "Clap your hands",
    "Touch your face / fix your hair",
    "Stretch your arms up",
    "Step side to side",
    "Talk with your hands",
]


def build_none_steps() -> list[dict]:
    return [
        {"label": "none", "text": text, "take": take}
        for text in NONE_PROMPTS
        for take in range(1, TAKES_PER_PROMPT + 1)
    ]


if __name__ == "__main__":
    collect.main(steps=build_none_steps(), file_suffix="_none")
