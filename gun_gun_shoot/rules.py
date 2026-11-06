"""
rules.py - the rules of Gun Gun Shoot, in one place.

Both the game (game.py) and the AI (opponent.py) use this file, so there is
exactly one definition of what each move does. If a rule changes, change it HERE.

The rules (see DESIGN.md section 3):
  reload  +1 bullet
  shoot   -1 bullet. Hits a reloading opponent (shooter wins).
          Blocked by shield. Reflected by mirror (MIRROR side wins).
          Shoot vs shoot = the bullets collide, nobody wins, the game goes on.
  shield  blocks a shot. Free.
  mirror  reflects a shot back at the shooter. Costs 1 bullet EVERY time it's used.

  Rule breaks (the other side wins immediately):
    - shooting with 0 bullets
    - mirroring with 0 bullets
    - a defensive move (shield/mirror) when you've already done MAX_DEFENSIVE in a row
"""

from __future__ import annotations

from dataclasses import dataclass, replace

MOVES = ("reload", "shoot", "shield", "mirror")
DEFENSIVE = ("shield", "mirror")

STARTING_BULLETS = 0
MAX_DEFENSIVE = 5  # max shield/mirror moves in a row; one more is a rule break

SIDES = ("player", "ai")


@dataclass(frozen=True)
class GameState:
    """Everything about the game between rounds. Frozen = it can't be changed by accident;
    resolve() returns a NEW state instead."""
    player_bullets: int = STARTING_BULLETS
    ai_bullets: int = STARTING_BULLETS
    player_streak: int = 0           # defensive moves in a row
    ai_streak: int = 0
    player_last: str | None = None   # last move (None before round 1)
    ai_last: str | None = None
    round_num: int = 1

    def bullets(self, side: str) -> int:
        return self.player_bullets if side == "player" else self.ai_bullets

    def streak(self, side: str) -> int:
        return self.player_streak if side == "player" else self.ai_streak


@dataclass(frozen=True)
class RoundResult:
    state: GameState        # the state AFTER this round
    winner: str | None      # "player", "ai", or None (game continues)
    event: str              # what happened (see EVENTS below); game.py turns this into a message
    actor: str | None       # which side the event is about (e.g. who got shot, who broke a rule)


# Possible values of RoundResult.event:
#   "rule_shoot_empty"   actor shot with 0 bullets          -> actor loses
#   "rule_mirror_empty"  actor mirrored with 0 bullets      -> actor loses
#   "rule_defensive"     actor did too many defensives      -> actor loses
#   "shot_reloading"     actor shot someone mid-reload      -> actor wins
#   "reflected"          actor mirrored a shot back         -> actor wins
#   "blocked"            actor's shot was blocked by a shield
#   "collided"           both shot; nobody wins
#   "quiet"              nothing dangerous happened


def other(side: str) -> str:
    return "ai" if side == "player" else "player"


def rule_break(state: GameState, side: str, move: str) -> str | None:
    """If this move would break a rule for this side, return the event name; else None."""
    if move == "shoot" and state.bullets(side) <= 0:
        return "rule_shoot_empty"
    if move == "mirror" and state.bullets(side) <= 0:
        return "rule_mirror_empty"
    if move in DEFENSIVE and state.streak(side) >= MAX_DEFENSIVE:
        return "rule_defensive"
    return None


def legal_moves(state: GameState, side: str) -> list[str]:
    """Every move this side can make without breaking a rule."""
    return [m for m in MOVES if rule_break(state, side, m) is None]


def resolve(state: GameState, player_move: str, ai_move: str) -> RoundResult:
    """Play one round. Returns the new state, the winner (if any) and what happened."""
    moves = {"player": player_move, "ai": ai_move}

    # --- Bullets and streaks update first (they show on screen even if the game ends) ---
    bullets = {s: state.bullets(s) for s in SIDES}
    streaks = {s: state.streak(s) for s in SIDES}
    for side in SIDES:
        m = moves[side]
        if m == "reload":
            bullets[side] += 1
        elif m in ("shoot", "mirror"):
            bullets[side] = max(0, bullets[side] - 1)
        streaks[side] = streaks[side] + 1 if m in DEFENSIVE else 0

    new_state = replace(
        state,
        player_bullets=bullets["player"], ai_bullets=bullets["ai"],
        player_streak=streaks["player"], ai_streak=streaks["ai"],
        player_last=player_move, ai_last=ai_move,
        round_num=state.round_num + 1,
    )

    def result(winner, event, actor):
        return RoundResult(new_state, winner, event, actor)

    # --- 1. Rule breaks (checked against the state BEFORE the round) ---
    breaks = {s: rule_break(state, s, moves[s]) for s in SIDES}
    if breaks["player"] and breaks["ai"]:
        return result(None, "quiet", None)  # both broke a rule: call it even (can't happen vs the AI)
    for side in SIDES:
        if breaks[side]:
            return result(other(side), breaks[side], side)

    # --- 2. Shots ---
    for side in SIDES:
        if moves[side] != "shoot":
            continue
        target_move = moves[other(side)]
        if target_move == "shoot":
            return result(None, "collided", None)
        if target_move == "reload":
            return result(side, "shot_reloading", side)
        if target_move == "mirror":
            return result(other(side), "reflected", other(side))
        if target_move == "shield":
            return result(None, "blocked", side)

    # --- 3. Nobody shot ---
    return result(None, "quiet", None)
