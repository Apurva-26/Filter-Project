"""
opponent.py - the computer player: decides what move to make each round.

How it decides (see DESIGN.md section 6), every round, BEFORE seeing the player's move:

  1. BASE STRATEGY (strategy.py): the "perfect player" mix for this situation,
     worked out from the rules alone. Unbeatable in the long run, but it doesn't
     take advantage of anyone's habits. e.g. right after reloading vs a player with
     0 bullets: reload 72%, shoot 28%.

  2. PREDICT the player's move from their habits.
       It keeps count tables of what players did in each situation:
           situation = (player bullets, AI bullets, player's last move)
       plus more general habits (what they do after each move, and overall);
       see LEVEL_WEIGHTS below.
       - GLOBAL table: every player ever. Saved to data/opponent_memory.json.
       - SESSION table: just this game's player. Reset every new game.
       Moves the player can't legally make (e.g. shoot with 0 bullets) get probability 0.

  3. EXPLOIT: score each AI move against that prediction. Each (AI move, player move)
     pair is worth +1 (AI wins), -1 (AI loses), or how good the next situation is
     (from strategy.py, so it thinks ahead, not just one round). Better-scoring moves
     get much higher chances (softmax, TEMPERATURE).

  4. BLEND 1 and 3. The more evidence it has about how players act in this situation,
     the more it trusts the exploit plan (up to MAX_EXPLOIT). With no evidence it just
     plays the base strategy. Then it picks at random using the blended chances, so it's
     never fully predictable.

Try it in the terminal (no camera needed):
  py opponent.py
"""

from __future__ import annotations

import json
import math
import os
import random

import gun_gun_shoot.rules as rules
from gun_gun_shoot.strategy import Strategy

HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(HERE, "data", "opponent_memory.json")

# =============================================================================
# Settings
# =============================================================================

# Difficulty: MAX_EXPLOIT is the main knob.
#   0.0 = only the base strategy (never adapts to the player)
#   0.6 = medium: mostly exploits habits once it has seen enough of them
#   0.9 = hard: almost fully exploits habits (can get predictable vs a player who changes up)
MAX_EXPLOIT = 0.6
EXPLOIT_RAMP = 8.0     # evidence needed to get halfway to MAX_EXPLOIT (in "rounds seen")
TEMPERATURE = 0.1      # randomness of the exploit plan: low = always the best-scoring move

SESSION_WEIGHT = 3.0   # one round from THIS player counts as much as this many global rounds
PRIOR_COUNT = 1.0      # pretend every legal move was seen this many times (so nothing is ever "impossible")
MAX_BULLET_BUCKET = 3  # bullets are grouped as 0, 1, 2, 3+ so situations repeat often enough to learn from


# The AI looks at the player's habits at three levels of detail, from exact to general:
#   exact situation   "with 1 bullet vs my 0, right after reloading, they usually..."
#   last move only    "right after reloading, they usually..."
#   overall           "this player just reloads a LOT"
# The exact level is the most useful but needs many rounds to fill up; the general
# levels kick in after just 2-3 rounds. Each level's counts are multiplied by its weight.
LEVEL_WEIGHTS = {"exact": 1.0, "last": 0.6, "overall": 0.3}


def situation_keys(state: rules.GameState) -> dict:
    """The count-table keys for this state at each level, e.g. {"exact": "p1|a0|reload", ...}."""
    pb = min(state.player_bullets, MAX_BULLET_BUCKET)
    ab = min(state.ai_bullets, MAX_BULLET_BUCKET)
    last = state.player_last or "start"
    return {"exact": f"p{pb}|a{ab}|{last}", "last": f"last|{last}", "overall": "overall"}


class Opponent:
    def __init__(self, max_exploit: float = MAX_EXPLOIT, memory_path: str = MEMORY_PATH):
        self.max_exploit = max_exploit
        self.memory_path = memory_path
        self.strategy = Strategy()          # the base strategy (takes a second or two to work out)
        self.global_counts = self._load()   # {situation: {move: count}}
        self.session_counts = {}
        self.last_prediction = {}           # for debugging / showing on screen
        self.last_chances = {}

    # --- memory ---

    def _load(self) -> dict:
        try:
            with open(self.memory_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self):
        """Write the global table to disk. Call at the end of every game."""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump(self.global_counts, f, indent=1)

    def new_game(self):
        """A new person is playing: forget the last player's habits (keep the global ones)."""
        self.session_counts = {}

    # --- step 2: predict ---

    def predict_player(self, state: rules.GameState) -> tuple[dict, float]:
        """({move: probability} for the player's next move, how much evidence that's based on).
        Evidence = how many rounds were seen in this exact situation or after this last move
        (weighted, so the session player counts more)."""
        legal = rules.legal_moves(state, "player")
        weights = {m: PRIOR_COUNT for m in legal}
        evidence = 0.0
        for level, key in situation_keys(state).items():
            g = self.global_counts.get(key, {})
            s = self.session_counts.get(key, {})
            for m in legal:
                seen = LEVEL_WEIGHTS[level] * (g.get(m, 0) + SESSION_WEIGHT * s.get(m, 0))
                weights[m] += seen
                if level != "overall":  # "they reload a lot overall" is too vague to count as evidence
                    evidence += seen
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()}, evidence

    # --- step 3: score ---

    def score_moves(self, state: rules.GameState, prediction: dict) -> dict:
        """{ai move: expected value} for every legal AI move, against the predicted player."""
        return {
            ai_move: sum(p * self.strategy.outcome_value(state, ai_move, player_move)
                         for player_move, p in prediction.items())
            for ai_move in rules.legal_moves(state, "ai")
        }

    # --- step 4: blend and pick ---

    def chances(self, state: rules.GameState) -> dict:
        """{ai move: chance of playing it} for this round."""
        base = self.strategy.mix(state)
        prediction, evidence = self.predict_player(state)
        scores = self.score_moves(state, prediction)
        self.last_prediction = prediction

        # Softmax: turn scores into chances. Dividing by the temperature
        # stretches the differences (low temp) or flattens them (high temp).
        best = max(scores.values())
        exploit = {m: math.exp((v - best) / TEMPERATURE) for m, v in scores.items()}
        total = sum(exploit.values())
        exploit = {m: c / total for m, c in exploit.items()}

        # How much to trust the exploit plan: 0 with no evidence, -> max_exploit with lots
        trust = self.max_exploit * evidence / (evidence + EXPLOIT_RAMP)
        return {m: (1 - trust) * base.get(m, 0.0) + trust * exploit[m] for m in scores}

    def choose(self, state: rules.GameState) -> str:
        """Pick the AI's move for this round."""
        chances = self.chances(state)
        self.last_chances = chances
        moves = list(chances)
        return random.choices(moves, weights=[chances[m] for m in moves])[0]

    # --- learning ---

    def observe(self, state_before: rules.GameState, player_move: str):
        """After a round: remember what the player did in that situation (at every level)."""
        for key in situation_keys(state_before).values():
            for table in (self.global_counts, self.session_counts):
                counts = table.setdefault(key, {})
                counts[player_move] = counts.get(player_move, 0) + 1


# =============================================================================
# Terminal mode: play against the AI by typing moves
# =============================================================================

KEYS = {"r": "reload", "s": "shoot", "h": "shield", "m": "mirror"}


def play_in_terminal():
    print("Gun Gun Shoot vs the AI. Type r = reload, s = shoot, h = shield, m = mirror, q = quit.\n")
    ai = Opponent()
    ai.new_game()
    state = rules.GameState()

    while True:
        print(f"Round {state.round_num}   you: {state.player_bullets} bullets   AI: {state.ai_bullets} bullets"
              f"   (your defensive streak: {state.player_streak}/{rules.MAX_DEFENSIVE})")
        ai_move = ai.choose(state)  # the AI commits BEFORE you type
        key = input("  your move: ").strip().lower()
        if key == "q":
            break
        if key not in KEYS:
            print("  ? use r / s / h / m")
            continue

        player_move = KEYS[key]
        guess = ", ".join(f"{m} {p:.0%}" for m, p in sorted(ai.last_prediction.items(), key=lambda x: -x[1]))
        plan = ", ".join(f"{m} {p:.0%}" for m, p in sorted(ai.last_chances.items(), key=lambda x: -x[1]))
        print(f"  AI predicted you: {guess}")
        print(f"  AI's chances were: {plan}")
        print(f"  you: {player_move.upper()}   AI: {ai_move.upper()}")

        result = rules.resolve(state, player_move, ai_move)
        ai.observe(state, player_move)
        state = result.state
        print(f"  -> {result.event}\n")

        if result.winner:
            print("YOU WIN!" if result.winner == "player" else "THE AI WINS!")
            break

    ai.save()
    print(f"(AI memory saved to {ai.memory_path})")


if __name__ == "__main__":
    play_in_terminal()
