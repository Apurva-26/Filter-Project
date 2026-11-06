"""
launcher.py - the home screen: shows a tile for every filter, tap one to start it.

How it works:
  1. The menu shows one big tile per filter (the list is in filters.json).
  2. Tapping a tile closes the menu window and runs that filter as its own program
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
  "name"        text on the tile
  "script"      the .py file to run, relative to this folder
  "icon"        clown / glasses / iris / mystery (a simple drawn picture), or leave it
                out to show the first letter of the name
  "color"       tile color [red, green, blue], 0-255
  "args"        (optional) extra command-line arguments for the script
  "kiosk_args"  (optional) extra arguments used only with --fullscreen
For the Back button and auto-back, the filter should use kiosk.py (OpenCV filters).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
FILTERS_PATH = os.path.join(HERE, "filters.json")

WINDOW_SIZE = (1280, 720)
HEADLINE = "Tap a filter to try it!"
FOOTNOTE = "The camera only runs while you use a filter. Nothing is recorded or saved."

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
BACKGROUND_TOP, BACKGROUND_BOTTOM = (25, 25, 45), (10, 10, 20)


def load_filters() -> list:
    with open(FILTERS_PATH, encoding="utf-8") as f:
        filters = json.load(f)
    for flt in filters:
        flt["path"] = os.path.join(HERE, flt["script"])
        if not os.path.exists(flt["path"]):
            print(f"Warning: {flt['script']} not found (tile '{flt['name']}' won't work)")
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
# Drawing
# =============================================================================


def draw_icon(screen, kind: str, center, size: int, name: str):
    """Simple drawn pictures for the tiles (no image files needed)."""
    cx, cy = center
    r = size // 2
    if kind == "clown":
        pygame.draw.circle(screen, WHITE, (cx, cy), r)                        # face
        pygame.draw.circle(screen, BLACK, (cx - r // 3, cy - r // 4), r // 10)  # eyes
        pygame.draw.circle(screen, BLACK, (cx + r // 3, cy - r // 4), r // 10)
        pygame.draw.circle(screen, (220, 30, 30), (cx, cy + r // 10), r // 4)  # red nose
        pygame.draw.arc(screen, BLACK, (cx - r // 2, cy, r, int(r / 1.6)), math.pi * 1.1, math.pi * 1.9, 4)
    elif kind == "glasses":
        lens = r // 2
        for dx in (-lens - 6, lens + 6):
            pygame.draw.circle(screen, WHITE, (cx + dx, cy), lens, 8)
        pygame.draw.line(screen, WHITE, (cx - 10, cy - 4), (cx + 10, cy - 4), 8)
    elif kind == "iris":
        eye = pygame.Rect(0, 0, size, int(size * 0.55))
        eye.center = (cx, cy)
        pygame.draw.ellipse(screen, WHITE, eye)
        pygame.draw.circle(screen, (130, 20, 60), (cx, cy), int(r * 0.42))     # maroon
        pygame.draw.circle(screen, (255, 60, 170), (cx, cy), int(r * 0.30))    # pink
        pygame.draw.circle(screen, BLACK, (cx, cy), int(r * 0.14))             # pupil
    else:
        text = "?" if kind == "mystery" else name[:1].upper()
        font = pygame.font.Font(None, int(size * 1.4))
        img = font.render(text, True, WHITE)
        screen.blit(img, img.get_rect(center=(cx, cy + size // 12)))


def tile_rects(n: int, w: int, h: int) -> list:
    """Lay out n tiles in one centered row (two rows if there are many)."""
    cols = n if n <= 4 else math.ceil(n / 2)
    rows = math.ceil(n / cols)
    gap = int(w * 0.03)
    tile_w = min(int((w * 0.9 - gap * (cols - 1)) / cols), int(h * 0.5))
    tile_h = int(tile_w * 1.25) if rows == 1 else int(tile_w * 1.05)
    total_h = rows * tile_h + (rows - 1) * gap
    top = (h - total_h) // 2 + int(h * 0.03)
    rects = []
    for i in range(n):
        row, col = divmod(i, cols)
        in_row = min(cols, n - row * cols)
        left = (w - (in_row * tile_w + (in_row - 1) * gap)) // 2
        rects.append(pygame.Rect(left + col * (tile_w + gap), top + row * (tile_h + gap), tile_w, tile_h))
    return rects


def draw_menu(screen, filters, rects, pressed: int | None, message: str):
    w, h = screen.get_size()
    # vertical gradient background
    for y in range(0, h, 4):
        t = y / h
        color = [int(a + (b - a) * t) for a, b in zip(BACKGROUND_TOP, BACKGROUND_BOTTOM)]
        pygame.draw.rect(screen, color, (0, y, w, 4))

    title = pygame.font.Font(None, int(h * 0.10)).render(HEADLINE, True, WHITE)
    screen.blit(title, title.get_rect(center=(w // 2, int(h * 0.11))))

    name_font = pygame.font.Font(None, int(h * 0.065))
    bob = math.sin(time.monotonic() * 2.5)
    for i, (flt, rect) in enumerate(zip(filters, rects)):
        r = rect.inflate(-10, -10) if i == pressed else rect
        color = flt.get("color", [90, 90, 90])
        pygame.draw.rect(screen, color, r, border_radius=28)
        pygame.draw.rect(screen, WHITE, r, 4, border_radius=28)
        icon_size = int(r.w * 0.55)
        icon_y = r.y + int(r.h * 0.40) + int(bob * 4)
        draw_icon(screen, flt.get("icon", ""), (r.centerx, icon_y), icon_size, flt["name"])
        label = name_font.render(flt["name"], True, WHITE)
        screen.blit(label, label.get_rect(center=(r.centerx, r.bottom - int(r.h * 0.13))))

    foot = pygame.font.Font(None, int(h * 0.04)).render(message or FOOTNOTE, True, (190, 190, 210))
    screen.blit(foot, foot.get_rect(center=(w // 2, int(h * 0.93))))


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
    clock = pygame.time.Clock()
    message, message_until = "", 0.0

    running = True
    while running:
        w, h = screen.get_size()
        rects = tile_rects(len(filters), w, h)
        chosen = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:  # a tap
                for i, rect in enumerate(rects):
                    if rect.collidepoint(event.pos):
                        chosen = i

        if time.monotonic() > message_until:
            message = ""
        draw_menu(screen, filters, rects, chosen, message)
        pygame.display.flip()

        if chosen is not None:
            pygame.time.wait(150)  # let the "pressed" tile show for a moment
            # Close the menu window so the filter's window gets the whole screen
            pygame.display.quit()
            try:
                code = run_filter(filters[chosen], kiosk=args.fullscreen)
            except OSError as e:
                print(f"Couldn't start {filters[chosen]['name']}: {e}")
                code = -1
            screen = open_window(args.fullscreen)
            pygame.event.clear()  # ignore taps made while the filter was closing
            if code != 0:
                message = f"Sorry, {filters[chosen]['name']} isn't working right now."
                message_until = time.monotonic() + 4

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
