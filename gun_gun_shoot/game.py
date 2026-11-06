"""
game.py - the full Gun Gun Shoot game: what players actually use.

Screens (the "state machine"), in the order a player sees them:

  attract    title screen, waiting for someone to tap / press SPACE
  position   "stand between 1.0 and 1.5 m away" (distance.py; must hold for a moment)
  ready      "Get ready!"
  beat       "GUN... GUN..." with sound. The AI picks its move at the START of this.
  window     "SHOOT!" -> the capture window: the player's pose is locked in (MoveLocker).
             The first GRACE_S seconds are ignored (hands still moving into place).
             All the beat / window timings are in classifier.py.
  reveal     both moves are shown, plus what happened; bullets update
  missed     no move was detected -> the round is replayed
  game_over  "YOU WIN!" / "YOU LOSE!" -> tap to play again, or back to attract after a while
  left       the player walked away (or stopped moving) -> back to attract

How the pieces connect:
  Vision (background thread): camera -> features.py landmarks -> classifier.py move probabilities
  Game (main thread):         screens, timing, sound, rules.py, opponent.py

The camera + MediaPipe run in their own thread so the screen and the beat timing
stay smooth even if pose detection is slow (e.g. on the Pi).

Usage:
  py game.py                        laptop: windowed, SPACE to start
  py game.py --fullscreen --touch   Raspberry Pi: fullscreen, tap the screen to start
  ESC quits.

Setup:
  py -m pip install pygame
  Optional: put your own recorded sounds in a "sounds/" folder (see Sounds below).
  Any sound file that's missing is replaced by a simple beep.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

import distance
import features
import rules
from classifier import (
    CAPTURE_WINDOW_S, GRACE_S, GUN_GAP_S, SHOOT_AT_S, MoveLocker, PoseClassifier, confident_move,
)
from opponent import Opponent

# =============================================================================
# >>> TOUCHSCREEN SWITCH <<<
# =============================================================================
# False: only SPACE starts a game (for testing on the laptop).
# True:  tapping the screen ALSO starts a game (for the Pi's touchscreen).
#        A touchscreen acts like a mouse, so a tap arrives as a mouse click.
# You can also leave this False and run `py game.py --touch`.
ENABLE_TOUCH = False


def is_tap(event) -> bool:
    """Is this event a 'start / play again' tap?"""
    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        return True
    if ENABLE_TOUCH and event.type == pygame.MOUSEBUTTONDOWN:
        return True
    return False


# =============================================================================
# Settings
# =============================================================================

WINDOW_SIZE = (1280, 720)     # window size when not fullscreen

# Timings (seconds). The GUN / GUN / SHOOT! beat timings are in classifier.py.
POSITION_HOLD_S = 1.0         # must be framed correctly this long before the game starts
POSITION_TIMEOUT_S = 30.0     # give up and go back to the title screen
READY_S = 2.0                 # "Get ready!"
REVEAL_S = 2.5                # how long both moves are shown
MISSED_S = 1.8                # "Didn't catch that!"
MAX_MISSES_IN_A_ROW = 3       # this many missed rounds in a row -> assume they stopped playing
LEAVE_TIMEOUT_S = 5.0         # nobody in front of the camera this long -> game ends
LEFT_S = 3.0                  # "Game ended" message
GAME_OVER_S = 15.0            # win/lose screen, then back to the title screen

# Launcher (../launcher.py) support. The launcher sets FILTER_KIOSK=1 when it starts the game.
KIOSK_MODE = os.environ.get("FILTER_KIOSK") == "1"
AUTO_BACK_S = 20.0            # kiosk mode: nobody in front of the camera this long -> back to the menu
BACK_BUTTON = (-150, 20, 130, 60)  # x (negative = from the right edge), y, width, height

# Framing check. The distance zone (1.0 - 1.5 m) is set in distance.py.
MIN_NOSE_Y_FRAC = 0.18        # nose higher than this (fraction of the screen height from the top)
                              # = no room above the head for reload

# Colors
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (70, 70, 70)
GREEN, RED, YELLOW = (60, 220, 90), (235, 60, 60), (255, 210, 60)
MOVE_COLORS = {
    "reload": (255, 200, 40),
    "shoot": (235, 60, 60),
    "shield": (70, 150, 255),
    "mirror": (190, 100, 255),
}


# =============================================================================
# Vision: camera + pose detection in a background thread
# =============================================================================


@dataclass
class VisionResult:
    seq: int                  # increases by 1 every processed frame (so we know when there's a new one)
    time: float               # when the frame was captured (time.monotonic())
    frame: np.ndarray         # RGB, mirrored, skeleton drawn on: ready to display
    person: bool              # was a person found?
    distance_m: float | None  # estimated distance from the camera (None if the eyes can't be seen)
    nose_y_frac: float | None # how far down the screen the nose is (0 = top, 1 = bottom)
    move_probs: dict | None   # {move: confidence}, or None if no person
    fps: float                # how many frames per second detection is managing


def check_framing(vis: VisionResult | None, margin: float = 0.0) -> str:
    """Is the player standing in the right place?
    Returns "none" (nobody), "no_face" (can't see the eyes), "too_close", "too_far",
    "too_high" (head too near the top of the screen) or "ok".
    margin: extra leeway on the distance zone (used during a game, see distance.py)."""
    if vis is None or not vis.person:
        return "none"
    z = distance.zone(vis.distance_m, margin)
    if z == "unknown":
        return "no_face"
    if z != "ok":
        return z
    if vis.nose_y_frac < MIN_NOSE_Y_FRAC:
        return "too_high"
    return "ok"


class Vision:
    """Runs the camera and the models nonstop in a background thread.
    The game just calls latest() whenever it wants the newest result."""

    def __init__(self, camera: int, width: int, height: int):
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera {camera}")

        # Load the models here (not in the thread) so any error shows up immediately.
        self.classifier = PoseClassifier()
        self.extractor = features.LandmarkExtractor()
        ok, frame = self.cap.read()
        self.distance = distance.DistanceEstimator(frame.shape[1] if ok else width)

        self.error = None
        self._latest = None
        self._lock = threading.Lock()  # stops the two threads touching _latest at the same moment
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        seq, fps, last = 0, 0.0, time.monotonic()
        try:
            while self._running:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                captured = time.monotonic()
                h, w = frame.shape[:2]

                landmarks = self.extractor.process(frame)
                probs = self.classifier.move_probabilities(landmarks, w, h) if landmarks.person_found else None
                dist = self.distance.update(landmarks)
                nose_y = landmarks.pose["nose"][1] / h if landmarks.person_found else None

                features.draw(frame, landmarks, show_boxes=False)
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # mirror view, RGB for pygame

                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                seq += 1
                with self._lock:
                    self._latest = VisionResult(seq, captured, rgb, landmarks.person_found, dist, nose_y, probs, fps)
        except Exception as e:  # hand the error to the main thread so the game stops with a message
            self.error = e

    def latest(self) -> VisionResult | None:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()
        self.extractor.close()


# =============================================================================
# Sounds
# =============================================================================


class Sounds:
    """
    Plays the game's sounds. To use your OWN recordings, put .wav files in a folder
    called "sounds/" next to this file, with these names:
        gun.wav     the "gun" (+ thigh slap) beat, played twice per round
        shoot.wav   "SHOOT!"
        reveal.wav  when the moves are revealed
        hit.wav     someone got shot / a rule was broken
        win.wav     player wins the game
        lose.wav    player loses the game
    Anything missing is replaced with a beep, so the game always works.
    """

    # The placeholder beeps: name -> (pitch in Hz, length in ms).
    # Higher Hz = higher note. 330 -> 660 is exactly one octave up, so SHOOT clearly
    # sounds higher than the two GUNs. Change these to taste.
    BEEPS = {
        "gun": (330, 150), "shoot": (660, 250), "reveal": (440, 120),
        "hit": (150, 450), "win": (880, 600), "lose": (110, 700),
    }
    VOLUME = 0.5  # 0.0 - 1.0
    FOLDER = os.path.join(features.HERE, "sounds")

    def __init__(self):
        self.sounds = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except pygame.error:
            print("No audio device found: playing without sound.")
            return
        rate, _, channels = pygame.mixer.get_init()
        for name, (pitch, ms) in self.BEEPS.items():
            path = os.path.join(self.FOLDER, f"{name}.wav")
            if os.path.exists(path):
                self.sounds[name] = pygame.mixer.Sound(path)
            else:
                self.sounds[name] = self._beep(pitch, ms, rate, channels)

    @staticmethod
    def _beep(pitch: float, ms: int, rate: int, channels: int):
        t = np.arange(int(rate * ms / 1000)) / rate
        # A steady tone with a very short fade in/out at the ends (stops it from "clicking")
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
        wave = np.sin(2 * np.pi * pitch * t) * fade
        samples = (wave * Sounds.VOLUME * 32767).astype(np.int16)
        if channels > 1:
            samples = np.repeat(samples[:, None], channels, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(samples))

    def play(self, name: str):
        if name in self.sounds:
            self.sounds[name].play()


# =============================================================================
# Text for what happened in a round
# =============================================================================


def describe(result: rules.RoundResult) -> str:
    you = result.actor == "player"
    e = result.event
    if e == "rule_shoot_empty":
        return "You tried to shoot with no bullets!" if you else "The CPU tried to shoot with no bullets!"
    if e == "rule_mirror_empty":
        return "You can't mirror with no bullets!" if you else "The CPU mirrored with no bullets!"
    if e == "rule_defensive":
        return (f"More than {rules.MAX_DEFENSIVE} defensive moves in a row!" if you
                else f"The CPU defended more than {rules.MAX_DEFENSIVE} times in a row!")
    if e == "shot_reloading":
        return "You shot the CPU while it was reloading!" if you else "The CPU shot you while you were reloading!"
    if e == "reflected":
        return "You reflected the CPU's bullet!" if you else "The CPU reflected your bullet!"
    if e == "blocked":
        return "The CPU blocked your shot." if you else "You blocked the CPU's shot."
    if e == "collided":
        return "Both shots collided!"
    return ""


# =============================================================================
# Drawing helpers
# =============================================================================


class Painter:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.w, self.h = screen.get_size()
        self._fonts = {}

    def font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)  # None = pygame's built-in font
        return self._fonts[size]

    def text(self, s: str, pos, size: int, color=WHITE, anchor: str = "center"):
        """Text with a black outline so it's readable on top of the camera image."""
        f = self.font(size)
        outline = f.render(s, True, BLACK)
        main = f.render(s, True, color)
        rect = main.get_rect(**{anchor: pos})
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
            self.screen.blit(outline, rect.move(dx, dy))
        self.screen.blit(main, rect)

    def dim(self, alpha: int = 160):
        """Darken the whole screen (for menus on top of the camera image)."""
        shade = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, alpha))
        self.screen.blit(shade, (0, 0))

    def bar(self, center, width: int, height: int, fraction: float, color=GREEN):
        x, y = center[0] - width // 2, center[1] - height // 2
        pygame.draw.rect(self.screen, GREY, (x, y, width, height), border_radius=height // 2)
        if fraction > 0:
            pygame.draw.rect(self.screen, color, (x, y, int(width * min(fraction, 1.0)), height),
                             border_radius=height // 2)

    def panel(self, rect, color=(0, 0, 0), alpha: int = 170):
        surf = pygame.Surface(rect[2:], pygame.SRCALPHA)
        pygame.draw.rect(surf, (*color, alpha), (0, 0, *rect[2:]), border_radius=18)
        self.screen.blit(surf, rect[:2])


# =============================================================================
# The game
# =============================================================================


class Game:
    def __init__(self, screen: pygame.Surface, vision: Vision, sounds: Sounds):
        self.paint = Painter(screen)
        self.screen = screen
        self.vision = vision
        self.sounds = sounds
        self.opponent = Opponent()
        self.locker = MoveLocker()

        self.state = "attract"
        self.state_start = time.monotonic()

        self.gs = rules.GameState()     # bullets, streaks, round number
        self.ai_move = None             # the AI's move for the current round (chosen before the beat)
        self.player_move = None
        self.result = None              # the last RoundResult
        self.end_message = ""           # shown on the "left" screen
        self.misses_in_a_row = 0
        self.beats_played = 0
        self.position_ok_since = None

        self.vis = None                 # newest VisionResult
        self.last_seq = -1
        self.last_person_time = time.monotonic()
        self.quit_requested = False     # set by the Back button / auto-back; main() then exits
        self._camera_surface = None     # the camera frame, converted for pygame

    # --- state helpers ---

    def go(self, state: str):
        self.state, self.state_start = state, time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.state_start

    def new_game(self):
        self.gs = rules.GameState()
        self.opponent.new_game()
        self.misses_in_a_row = 0
        self.position_ok_since = None
        self.last_person_time = time.monotonic()
        self.go("position")

    def start_round(self):
        self.ai_move = self.opponent.choose(self.gs)  # the AI commits BEFORE the player moves
        self.beats_played = 0
        self.go("beat")

    def finish_round(self, player_move: str):
        self.player_move = player_move
        self.result = rules.resolve(self.gs, player_move, self.ai_move)
        self.opponent.observe(self.gs, player_move)  # the AI learns from what the player did
        self.gs = self.result.state
        self.misses_in_a_row = 0
        self.sounds.play("hit" if self.result.winner else "reveal")
        self.go("reveal")

    def end_early(self, message: str):
        self.end_message = message
        self.opponent.save()
        self.go("left")

    # --- input ---

    def back_button_rect(self) -> pygame.Rect:
        x, y, w, h = BACK_BUTTON
        return pygame.Rect(x + self.paint.w if x < 0 else x, y, w, h)

    def on_click(self, pos) -> bool:
        """A click/tap somewhere on the screen. Returns True if it hit the Back button."""
        if self.back_button_rect().collidepoint(pos):
            self.opponent.save()
            self.quit_requested = True
            return True
        return False

    def draw_back_button(self):
        r = self.back_button_rect()
        self.paint.panel((r.x, r.y, r.w, r.h), GREY, alpha=210)
        pygame.draw.rect(self.screen, WHITE, r, 2, border_radius=18)
        self.paint.text("< Back", r.center, 40)

    def on_tap(self):
        if self.state in ("attract", "game_over"):
            self.new_game()

    # --- every frame: advance the game ---

    def update(self):
        now = time.monotonic()

        # Kiosk mode: nobody in front of the camera for a while -> back to the launcher menu
        if KIOSK_MODE and now - self.last_person_time > AUTO_BACK_S:
            self.opponent.save()
            self.quit_requested = True
            return

        # Pick up the newest camera result
        self.vis = self.vision.latest()
        new_frame = self.vis is not None and self.vis.seq != self.last_seq
        if new_frame:
            self.last_seq = self.vis.seq
            self._camera_surface = None  # convert the new frame next time we draw
            if self.vis.person:
                self.last_person_time = self.vis.time

        # Did the player walk away mid-game?
        if self.state in ("ready", "beat", "window", "reveal", "missed"):
            if now - self.last_person_time > LEAVE_TIMEOUT_S:
                self.end_early("Nobody's there - game ended.")
                return

        s, e = self.state, self.elapsed

        if s == "position":
            if check_framing(self.vis) == "ok":
                if self.position_ok_since is None:
                    self.position_ok_since = now
                elif now - self.position_ok_since >= POSITION_HOLD_S:
                    self.go("ready")
            else:
                self.position_ok_since = None
            if self.state == "position" and e > POSITION_TIMEOUT_S:
                self.go("attract")

        elif s == "ready":
            if e >= READY_S:
                self.start_round()

        elif s == "beat":
            # "gun" at 0s, "gun" at GUN_GAP_S; "SHOOT!" at SHOOT_AT_S opens the window
            while self.beats_played < 2 and e >= self.beats_played * GUN_GAP_S:
                self.sounds.play("gun")
                self.beats_played += 1
            if e >= SHOOT_AT_S:
                self.sounds.play("shoot")
                self.locker.reset()
                self.go("window")

        elif s == "window":
            # Only use frames captured at least GRACE_S after "SHOOT!" (hands are still moving
            # before that). Frames where the player is outside the distance zone count as "unsure".
            if new_frame and self.vis.time >= self.state_start + GRACE_S:
                move = None
                if self.vis.move_probs and check_framing(self.vis, distance.ZONE_MARGIN_M) == "ok":
                    move = confident_move(self.vis.move_probs)[0]
                self.locker.update(move, self.vis.time)
            if self.locker.locked:
                self.finish_round(self.locker.locked)
            elif e >= CAPTURE_WINDOW_S:
                self.misses_in_a_row += 1
                if self.misses_in_a_row >= MAX_MISSES_IN_A_ROW:
                    self.end_early("No moves detected - game ended.")
                else:
                    self.go("missed")

        elif s == "reveal":
            if e >= REVEAL_S:
                if self.result.winner:
                    self.sounds.play("win" if self.result.winner == "player" else "lose")
                    self.opponent.save()
                    self.go("game_over")
                else:
                    self.start_round()

        elif s == "missed":
            if e >= MISSED_S:
                self.start_round()

        elif s == "game_over":
            if e >= GAME_OVER_S:
                self.go("attract")

        elif s == "left":
            if e >= LEFT_S:
                self.go("attract")

    # --- every frame: draw ---

    def draw(self):
        p = self.paint
        self._draw_camera()

        s, e = self.state, self.elapsed
        cx, cy = p.w // 2, p.h // 2
        tap_word = "Tap the screen" if ENABLE_TOUCH else "Press SPACE"

        if s == "attract":
            p.dim(170)
            # No game title on screen (on purpose): just the tagline and the pose guide.
            p.text("Can you beat the computer?", (cx, int(p.h * 0.2)), 64, YELLOW)
            guide = [
                ("RELOAD", "both finger guns pointing up", "reload"),
                ("SHOOT", "finger gun pointed at the screen", "shoot"),
                ("SHIELD", "arms crossed in an X", "shield"),
                ("MIRROR", "hands make a diamond (costs a bullet)", "mirror"),
            ]
            for i, (name, how, move) in enumerate(guide):
                y = int(p.h * 0.36) + i * 60
                p.text(name, (cx - 30, y), 46, MOVE_COLORS[move], anchor="midright")
                p.text(how, (cx, y), 38, WHITE, anchor="midleft")
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4)
            p.text(f"{tap_word} to play!", (cx, int(p.h * 0.86)), 64,
                   tuple(int(160 + 95 * pulse) for _ in range(3)))
            return

        if s == "position":
            p.dim(120)
            p.text(f"Stand {distance.MIN_DISTANCE_M:g} - {distance.MAX_DISTANCE_M:g} m from the camera",
                   (cx, int(p.h * 0.18)), 60)
            p.text("and face the screen", (cx, int(p.h * 0.26)), 44)
            framing = check_framing(self.vis)
            status = {
                "none": ("I can't see you yet...", WHITE),
                "no_face": ("Look at the screen!", YELLOW),
                "too_close": ("Too close! Step BACK", YELLOW),
                "too_far": ("Too far! Come CLOSER", YELLOW),
                "too_high": ("Crouch a little: I can't see above your head", YELLOW),
                "ok": ("Perfect! Hold still...", GREEN),
            }[framing]
            p.text(status[0], (cx, int(p.h * 0.72)), 72, status[1])
            if self.vis and self.vis.distance_m:
                p.text(f"You're {self.vis.distance_m:.2f} m away", (cx, int(p.h * 0.64)), 40)
            if self.position_ok_since:
                p.bar((cx, int(p.h * 0.80)), 400, 22, (time.monotonic() - self.position_ok_since) / POSITION_HOLD_S)
            return

        if s == "left":
            p.dim(190)
            p.text(self.end_message, (cx, cy), 64, YELLOW)
            return

        if s == "game_over":
            p.dim(170)
            won = self.result.winner == "player"
            p.text("YOU WIN!" if won else "YOU LOSE!", (cx, int(p.h * 0.30)), 170, GREEN if won else RED)
            p.text(describe(self.result), (cx, int(p.h * 0.47)), 52)
            p.text(f"You lasted {self.gs.round_num - 1} rounds", (cx, int(p.h * 0.56)), 40)
            p.text(f"{tap_word} to play again", (cx, int(p.h * 0.75)), 56, YELLOW)
            p.text(f"(back to the start in {max(0, math.ceil(GAME_OVER_S - e))})", (cx, int(p.h * 0.82)), 32)
            return

        # --- In-game screens: scoreboard on top, feedback on the bottom ---
        self._draw_scoreboard()

        if s == "ready":
            p.text("Get ready!", (cx, cy - 30), 120, YELLOW)
            p.text("Slap your thighs on GUN... GUN... then do your move on SHOOT!", (cx, cy + 50), 40)
        elif s == "beat":
            words = "GUN..." if e < GUN_GAP_S else "GUN... GUN..."
            p.text(words, (cx, cy), 150)
        elif s == "window":
            p.text("SHOOT!", (cx, cy - 40), 190, RED)
            leader = self.locker.leader
            p.bar((cx, cy + 70), 420, 26, self.locker.progress(), MOVE_COLORS.get(leader, GREEN))
        elif s == "reveal":
            self._draw_reveal()
        elif s == "missed":
            p.text("Didn't catch that!", (cx, cy - 20), 100, YELLOW)
            p.text("Hold your move a little longer. Again!", (cx, cy + 55), 44)

        self._draw_feedback()

    def _draw_camera(self):
        if self.vis is None:
            self.screen.fill(BLACK)
            self.paint.text("Starting camera...", (self.paint.w // 2, self.paint.h // 2), 50)
            return
        if self._camera_surface is None:
            fh, fw = self.vis.frame.shape[:2]
            surf = pygame.image.frombuffer(self.vis.frame.tobytes(), (fw, fh), "RGB")
            self._camera_surface = pygame.transform.scale(surf, (self.paint.w, self.paint.h))
        self.screen.blit(self._camera_surface, (0, 0))

    def _draw_scoreboard(self):
        p = self.paint
        for side, label, x, anchor in (("player", "YOU", 20, "topleft"), ("ai", "CPU", p.w - 460, "topleft")):
            p.panel((x, 20, 280, 120))
            p.text(label, (x + 20, 32), 52, YELLOW, anchor=anchor)
            p.text(f"Bullets: {self.gs.bullets(side)}", (x + 20, 78), 40, anchor=anchor)
            streak = self.gs.streak(side)
            p.text(f"Defense streak: {streak}/{rules.MAX_DEFENSIVE}", (x + 20, 110), 28,
                   RED if streak >= rules.MAX_DEFENSIVE else WHITE, anchor=anchor)
        p.text(f"Round {self.gs.round_num}", (p.w // 2, 45), 44)

    def _draw_reveal(self):
        p = self.paint
        cx, cy = p.w // 2, p.h // 2
        for label, move, x in (("YOU", self.player_move, cx - 260), ("CPU", self.ai_move, cx + 260)):
            p.panel((x - 210, cy - 130, 420, 200), MOVE_COLORS[move], alpha=200)
            p.text(label, (x, cy - 90), 50)
            p.text(move.upper(), (x, cy - 10), 100)
        p.text("vs", (cx, cy - 30), 60)
        message = describe(self.result)
        if message:
            p.text(message, (cx, cy + 120), 54, YELLOW)

    def _draw_feedback(self):
        """Bottom of the screen: what the camera thinks you're doing right now."""
        p = self.paint
        y = p.h - 40
        framing = check_framing(self.vis, distance.ZONE_MARGIN_M)
        if not self.vis or not self.vis.person:
            p.text("I can't see you!", (p.w // 2, y), 44, RED)
        elif framing == "too_close":
            p.text("Too close! Step back", (p.w // 2, y), 48, RED)
        elif framing == "too_far":
            p.text("Too far! Come closer", (p.w // 2, y), 48, RED)
        elif self.vis.move_probs:
            move, top, conf = confident_move(self.vis.move_probs)
            if move:
                p.text(f"I see: {move.upper()}  {conf:.0%}", (p.w // 2, y), 48, MOVE_COLORS[move])
            else:
                p.text("I see: ---", (p.w // 2, y), 44)
        if self.vis:
            p.text(f"{self.vis.fps:4.1f} FPS", (15, p.h - 15), 24, GREY, anchor="bottomleft")


# =============================================================================
# Main
# =============================================================================


def main():
    global ENABLE_TOUCH
    parser = argparse.ArgumentParser(description="Play Gun Gun Shoot.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280, help="camera width")
    parser.add_argument("--height", type=int, default=720, help="camera height")
    parser.add_argument("--fullscreen", action="store_true", help="fill the whole screen (use on the Pi)")
    parser.add_argument("--touch", action="store_true", help="tapping the screen starts a game")
    args = parser.parse_args()
    if args.touch:
        ENABLE_TOUCH = True

    pygame.init()
    if args.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)  # no cursor in the middle of a touchscreen
    else:
        screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("")  # no game name shown anywhere while playing

    sounds = Sounds()
    vision = Vision(args.camera, args.width, args.height)
    game = Game(screen, vision, sounds)
    clock = pygame.time.Clock()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and game.on_click(event.pos):
                    pass  # it was the Back button
                elif is_tap(event):
                    game.on_tap()
            if vision.error:
                raise vision.error
            game.update()
            game.draw()
            game.draw_back_button()
            if game.quit_requested:
                running = False
            pygame.display.flip()
            clock.tick(60)  # redraw the screen up to 60 times a second
    finally:
        game.opponent.save()
        vision.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
