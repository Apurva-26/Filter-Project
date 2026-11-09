"""
launcher.py - the home screen: an arcade with one machine per filter. Tap one to start it.

How it works:
  1. The menu shows one arcade machine per filter (the list is in filters.json).
     The drawing itself is in arcade_art.py.
  2. Tapping anywhere on a machine closes the menu window and runs that filter as its own program
     (exactly like typing `py saumyaa.py` in its folder).
  3. When the filter ends (its Back button, auto-back after 20 s with nobody there,
     or a crash), the menu comes straight back.

The menu never touches the camera, so each filter gets the camera to itself.

Usage:
  py launcher.py                laptop: windowed menu, filters open in normal windows
  py launcher.py --fullscreen   Raspberry Pi: fullscreen menu, filters run fullscreen
                                with auto-back ("kiosk mode")
  ESC quits the menu (needs a keyboard, so hallway visitors can't close it).

Adding a filter: add an entry to filters.json:
  "name"        text on top of the machine
  "script"      the .py file to run, relative to this folder
  "icon"        which animation its screen shows: clown / bigeyes / iris (see SCREEN_ART
                in arcade_art.py). Leave it out for a blank screen.
  "args"        (optional) extra command-line arguments for the script
  "kiosk_args"  (optional) extra arguments used only with --fullscreen
For the Back button and auto-back, the filter should use kiosk.py (OpenCV filters).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pygame

from arcade_art import Arcade

HERE = os.path.dirname(os.path.abspath(__file__))
FILTERS_PATH = os.path.join(HERE, "filters.json")

WINDOW_SIZE = (1280, 720)
TITLE = "Arcade"
# >>> The row of symbols under the title, in machine order. Type them in as text, e.g.
#     SUBTITLE_SYMBOLS = [":)", "<3", ":D", "\U0001F604"]   (or paste real emojis in)
SUBTITLE_SYMBOLS = [":3", "👍", "🌹", ":D"]
FOOTNOTE = "The camera only runs while you use a filter. Nothing is recorded or saved."


def load_filters() -> list:
    with open(FILTERS_PATH, encoding="utf-8") as f:
        filters = json.load(f)
    for flt in filters:
        flt["path"] = os.path.join(HERE, flt["script"])
        if not os.path.exists(flt["path"]):
            print(f"Warning: {flt['script']} not found (machine '{flt['name']}' won't work)")
    return filters


# =============================================================================
# Running a filter
# =============================================================================


def run_filter(flt: dict, kiosk: bool) -> int:
    """Run one filter and wait until it closes. Returns its exit code (0 = fine)."""
    args = [sys.executable, flt["path"], *flt.get("args", [])]
    if kiosk:
        args += flt.get("kiosk_args", [])
    env = dict(os.environ)
    if kiosk:
        env["FILTER_KIOSK"] = "1"   # tells kiosk.py / game.py: fullscreen + auto-back
    print(f"Starting {flt['name']}: {' '.join(args)}")
    # cwd = the script's own folder, so it finds its images/models like it does when run by hand
    return subprocess.run(args, cwd=os.path.dirname(flt["path"]), env=env).returncode


# =============================================================================
# Main
# =============================================================================


def open_window(fullscreen: bool):
    pygame.display.init()
    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)  # no cursor on a touchscreen
    else:
        screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("")
    return screen


def main():
    parser = argparse.ArgumentParser(description="Filter menu.")
    parser.add_argument("--fullscreen", action="store_true",
                        help="fullscreen menu + kiosk mode for the filters (use on the Pi)")
    args = parser.parse_args()

    filters = load_filters()
    pygame.init()
    screen = open_window(args.fullscreen)
    arcade = Arcade(filters, screen.get_size(), TITLE, SUBTITLE_SYMBOLS, FOOTNOTE)
    clock = pygame.time.Clock()
    message, message_until = "", 0.0

    running = True
    while running:
        chosen = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:  # a tap
                chosen = arcade.machine_at(event.pos)

        if time.monotonic() > message_until:
            message = ""
        arcade.draw(screen, chosen, message)
        pygame.display.flip()

        if chosen is not None:
            pygame.time.wait(150)  # let the "pressed" machine show for a moment
            # Close the menu window so the filter's window gets the whole screen
            pygame.display.quit()
            try:
                code = run_filter(filters[chosen], kiosk=args.fullscreen)
            except OSError as e:
                print(f"Couldn't start {filters[chosen]['name']}: {e}")
                code = -1
            screen = open_window(args.fullscreen)
            if screen.get_size() != arcade.size:
                arcade = Arcade(filters, screen.get_size(), TITLE, SUBTITLE_SYMBOLS, FOOTNOTE)
            pygame.event.clear()  # ignore taps made while the filter was closing
            if code != 0:
                message = f"Sorry, {filters[chosen]['name']} isn't working right now."
                message_until = time.monotonic() + 4

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
