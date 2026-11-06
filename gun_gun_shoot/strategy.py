"""
strategy.py - the "perfect player" strategy, worked out from the rules alone.

opponent.py uses this as the AI's BASE strategy: what to do when it knows nothing
about the person it's playing. On top of it, opponent.py adds what it has learned
about the player's habits.

The idea (game theory: a "Nash equilibrium" / "minimax" strategy)
------------------------------------------------------------------
In a game like rock-paper-scissors, ANY fixed move can be beaten, so the safest
plan is a MIX of moves (1/3 each). Gun Gun Shoot is the same, except the best mix
depends on the situation: bullets and defense streaks. This file works out, for
every situation:

    mix(state)   how often to play each move, e.g. {"reload": 0.72, "shoot": 0.28}
                 Nobody can beat this mix in the long run, even if they knew it.
    value(state) how good the situation is for the AI: +1 = sure win, -1 = sure loss.

Some things it figures out on its own, just from the rules:
  - the player has 0 bullets: never shield (pointless), mostly reload, sometimes shoot
  - right after the AI reloads, it does NOT always shoot (that would be predictable)
  - the player is on 5 defenses in a row: they MUST reload or shoot -> attack!

How it's calculated ("value iteration")
---------------------------------------
A situation's value depends on the values of the situations it can lead to, which
depend on THEIR next situations... So we start by guessing "every situation = 0" and
repeat:
    for every situation:
        build a 4x4 table: for each (AI move, player move) pair, what happens?
            AI wins -> +1, AI loses -> -1,
            game goes on -> FUTURE_DISCOUNT x value of the next situation
        solve that table: find the AI's best mix and the table's value
until the values stop changing (about 30 rounds of this, around a second).
FUTURE_DISCOUNT (< 1) means "a win now is better than a win later", so the AI
doesn't stall forever.

Solving one table: we try every possible "support" (subset of moves the mix uses),
solve the small linear system that makes the opponent's choices equally good, and
keep the best valid answer. Tables are at most 4x4, so this is quick. All the
situations are solved at the same time with numpy.

Try it:  py strategy.py      (prints the mix for some example situations)
"""

from __future__ import annotations

import itertools

import numpy as np

import gun_gun_shoot.rules as rules

# =============================================================================
# Settings
# =============================================================================

FUTURE_DISCOUNT = 0.95  # a win next round is worth 0.95 of a win right now
MAX_BULLETS = 4         # more bullets than this counts as this many (keeps the table small)
TOLERANCE = 1e-6        # stop when values change less than this

MOVES = list(rules.MOVES)  # reload, shoot, shield, mirror
N = len(MOVES)


def _key(state: rules.GameState) -> tuple:
    """The parts of the game state the strategy depends on."""
    return (min(state.player_bullets, MAX_BULLETS), min(state.ai_bullets, MAX_BULLETS),
            min(state.player_streak, rules.MAX_DEFENSIVE), min(state.ai_streak, rules.MAX_DEFENSIVE))


class Strategy:
    def __init__(self):
        # Every situation: (player bullets, AI bullets, player streak, AI streak)
        b, d = range(MAX_BULLETS + 1), range(rules.MAX_DEFENSIVE + 1)
        self.keys = list(itertools.product(b, b, d, d))
        self.index = {k: i for i, k in enumerate(self.keys)}
        S = len(self.keys)

        # For every situation and every (AI move, player move) pair, play the round once
        # with rules.resolve() and remember what happens. Index order: [situation, ai, player].
        self._win = np.zeros((S, N, N))              # +1 / -1 if the round ends the game
        self._ends = np.zeros((S, N, N), bool)       # does the round end the game?
        self._next = np.zeros((S, N, N), int)        # otherwise: which situation comes next
        self._ai_illegal = np.zeros((S, N), bool)
        self._player_illegal = np.zeros((S, N), bool)
        for k, i in self.index.items():
            state = rules.GameState(player_bullets=k[0], ai_bullets=k[1], player_streak=k[2], ai_streak=k[3])
            for m, move in enumerate(MOVES):
                self._ai_illegal[i, m] = rules.rule_break(state, "ai", move) is not None
                self._player_illegal[i, m] = rules.rule_break(state, "player", move) is not None
            for a, ai_move in enumerate(MOVES):
                for p, player_move in enumerate(MOVES):
                    result = rules.resolve(state, player_move, ai_move)
                    if result.winner:
                        self._ends[i, a, p] = True
                        self._win[i, a, p] = 1.0 if result.winner == "ai" else -1.0
                    else:
                        self._next[i, a, p] = self.index[_key(result.state)]

        # Value iteration (see the top of the file)
        self.values = np.zeros(S)
        for _ in range(500):
            table = self._tables()
            self.mixes, new_values = _solve_all(table)
            done = np.abs(new_values - self.values).max() < TOLERANCE
            self.values = new_values
            if done:
                break
        self._final_tables = self._tables()

    def _tables(self) -> np.ndarray:
        """The 4x4 outcome table of every situation, using the current value guesses."""
        table = np.where(self._ends, self._win, FUTURE_DISCOUNT * self.values[self._next])
        # Illegal moves: the AI never wants one (-10), and the player never makes one (+10).
        table[self._ai_illegal] = -10.0
        table = np.where(self._player_illegal[:, None, :], 10.0, table)
        return table

    # --- what opponent.py uses ---

    def mix(self, state: rules.GameState) -> dict:
        """The perfect-player mix for this situation: {move: probability}, legal moves only."""
        probs = self.mixes[self.index[_key(state)]]
        return {m: float(p) for m, p in zip(MOVES, probs) if rules.rule_break(state, "ai", m) is None}

    def outcome_value(self, state: rules.GameState, ai_move: str, player_move: str) -> float:
        """How good this pair of moves is for the AI, counting what it leads to later.
        (+1 = AI wins now, -1 = AI loses now, in between = how good the next situation is.)"""
        return float(self._final_tables[self.index[_key(state)], MOVES.index(ai_move), MOVES.index(player_move)])

    def value(self, state: rules.GameState) -> float:
        return float(self.values[self.index[_key(state)]])


def _solve_all(table: np.ndarray):
    """Solve every situation's table at once. Returns (best AI mix per situation, value per situation).

    For a mix x over the AI's moves, the player's best reply gives the AI min_j sum_i x_i * table[i, j].
    We want the x that makes that as big as possible. The best x always uses some k AI moves
    against k player moves that come out exactly equal, so we try every such pair of subsets,
    solve the equations, and keep the best answer that is a real mix (no negative chances)."""
    S = table.shape[0]
    best_value = np.full(S, -np.inf)
    best_mix = np.zeros((S, N))
    for k in range(1, N + 1):
        rhs = np.zeros((S, k + 1, 1))
        rhs[:, k] = 1.0
        for rows in itertools.combinations(range(N), k):
            for cols in itertools.combinations(range(N), k):
                # Equations: for every chosen player move j:  sum_i x_i * table[i, j] - v = 0
                #            and the chances add up to 1:       sum_i x_i = 1
                eq = np.zeros((S, k + 1, k + 1))
                eq[:, :k, :k] = table[:, list(rows)][:, :, list(cols)].transpose(0, 2, 1)
                eq[:, :k, k] = -1.0
                eq[:, k, :k] = 1.0
                solvable = np.abs(np.linalg.det(eq)) > 1e-9
                eq[~solvable] = np.eye(k + 1)  # placeholder so the batch solve doesn't crash
                sol = np.linalg.solve(eq, rhs)[:, :, 0]
                x_sub, v = sol[:, :k], sol[:, k]

                x = np.zeros((S, N))
                x[:, list(rows)] = np.clip(x_sub, 0, None)
                worst_reply = np.einsum("si,sij->sj", x, table).min(axis=1)
                good = (solvable & (x_sub >= -1e-9).all(axis=1)
                        & (worst_reply >= v - 1e-7) & (v > best_value))
                best_value[good] = v[good]
                best_mix[good] = x[good]
    return best_mix, best_value


if __name__ == "__main__":
    import time

    t = time.time()
    strategy = Strategy()
    print(f"Solved {len(strategy.keys)} situations in {time.time() - t:.1f}s\n")
    examples = [
        (0, 0, 0, 0, "start of the game"),
        (0, 1, 0, 0, "AI just reloaded, player has nothing"),
        (1, 0, 0, 0, "player has a bullet, AI has none"),
        (1, 1, 0, 0, "both have 1 bullet"),
        (3, 1, 0, 0, "player has lots of bullets"),
        (1, 1, 5, 0, "player can't defend any more"),
        (1, 1, 0, 4, "AI has 1 defense left"),
    ]
    for pb, ab, ps, as_, what in examples:
        state = rules.GameState(player_bullets=pb, ai_bullets=ab, player_streak=ps, ai_streak=as_)
        mix = ", ".join(f"{m} {p:.0%}" for m, p in strategy.mix(state).items() if p >= 0.005)
        print(f"{what:40s} value {strategy.value(state):+.2f}   AI plays: {mix}")
