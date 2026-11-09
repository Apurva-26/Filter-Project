"""
arcade_art.py - draws the arcade menu that launcher.py shows.

Everything here is drawn in code (no picture files), in black-and-white line art,
with color only on the animated parts.

How the drawing works:
  - The layout is planned on an imaginary "design canvas" (DESIGN_W x DESIGN_H), then
    scaled to fit the real screen. Shapes keep their proportions on any screen, and
    any extra space is just background. On a tall screen the machines wrap into rows.
  - pygame's shapes have jagged edges, so everything is drawn at 2x size and shrunk
    down (SUPERSAMPLE). That smooths the edges.
  - The room and machines never change, so they're drawn ONCE into a picture. Each
    frame only redraws the small animated machine screens, to keep the work low on the Pi.

Adding art for a new machine: write a function like draw_clown(surf, t), where surf
is the machine's screen and t is the time in seconds, and add it to SCREEN_ART below.
A machine whose "icon" isn't in SCREEN_ART just gets a blank screen.
"""

import math
import os

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE_FONT_PATH = os.path.join(HERE, "assets", "fonts", "Limelight-Regular.ttf")
EMOJI_FONT_PATH = os.path.join(HERE, "assets", "fonts", "NotoEmoji-Variable.ttf")

INK, PAPER = (0, 0, 0), (255, 255, 255)
SUPERSAMPLE = 2

# --- Layout, in design-canvas units ---
DESIGN_W, DESIGN_H = 1600, 900     # the room, for a wide screen with one row of machines
MACHINE_W, MACHINE_H = 300, 470    # one machine, including its feet
MACHINE_GAP = 70                   # space between machines
ROW_GAP = 40                       # space between rows, on tall screens
MACHINES_TOP = 275                 # top of the first row
TITLE_Y, SUBTITLE_Y = 105, 205     # center of the title and the symbol row
TITLE_SIZE, SUBTITLE_SIZE = 130, 58
FOOTNOTE_SIZE = 30
FLOOR_ABOVE_FEET = 100             # the floor starts this far above the machines' feet
FLOOR_DEPTH = 75                   # and ends this far below them
FOOTNOTE_GAP = 40                  # footnote center, below the floor

# --- Animation speeds, in seconds ---
NOSE_PULSE_S = 1.4        # clown nose: one slow pulse
EYE_SWELL_S = 3.2         # big eyes: one swell and pop
EYE_POP_FRACTION = 0.12   # the last part of that cycle is the quick pop back
BLINK_EVERY_S = 2.4       # iris: time from one blink to the next
BLINK_LENGTH_S = 0.3      # iris: how long a blink takes

# The three iris colors the Iris Colors filter really uses (RGB)
IRIS_COLORS = [(150, 10, 25), (255, 40, 170), (230, 170, 20)]
CLOWN_RED = (225, 25, 35)
BIG_EYE_BLUE = (70, 130, 210)


def load_font(path: str, size: int) -> pygame.font.Font:
    """A font from a file, or pygame's built-in font if the file is missing."""
    try:
        return pygame.font.Font(path, size)
    except (FileNotFoundError, pygame.error):
        print(f"Warning: font not found: {path}")
        return pygame.font.Font(None, size)


def render_fit(text: str, path: str, size: int, max_w: int, color=INK) -> pygame.Surface:
    """Render text, shrinking the font size if needed so it fits within max_w pixels."""
    while True:
        image = load_font(path, size).render(text, True, color)
        if image.get_width() <= max_w or size <= 8:
            return image
        size -= 2


def render_symbol(symbol: str, size: int) -> pygame.Surface:
    """One subtitle symbol. Plain keyboard characters like :) use the title font,
    and anything else (emojis) uses the black-and-white emoji font."""
    symbol = symbol.replace("️", "")  # an invisible "show as emoji" character fonts don't have
    path = TITLE_FONT_PATH if symbol.isascii() else EMOJI_FONT_PATH
    return load_font(path, size).render(symbol, True, INK)


# =============================================================================
# Animated machine screens. Each one draws onto surf (already filled white).
# =============================================================================

_static_layers = {}  # parts that never change, drawn once: {(name, size): surface}


def _sausage(surf, points, radius: int, line: int):
    """A thick outlined band that follows a curve (used for the clown's smile)."""
    for x, y in points:
        pygame.draw.circle(surf, INK, (x, y), radius)
    for x, y in points:
        pygame.draw.circle(surf, PAPER, (x, y), radius - line)


def _clown_static(size) -> pygame.Surface:
    """The glasses and the smile."""
    w, h = size
    layer = pygame.Surface(size)
    layer.fill(PAPER)
    cx, eyes_y = w // 2, int(h * 0.30)
    lens_r, lens_dx, line = int(h * 0.185), int(h * 0.225), max(2, int(h * 0.035))

    # Glasses: two round lenses, a bridge between them, and an arm on each side
    bar_h = line
    for side in (-1, 1):
        lens_x = cx + side * lens_dx
        pygame.draw.circle(layer, INK, (lens_x, eyes_y), lens_r, line)
        outer = lens_x + side * lens_r
        arm_end = outer + side * int(h * 0.09)
        pygame.draw.rect(layer, INK, (min(outer, arm_end), eyes_y - bar_h // 2, abs(arm_end - outer), bar_h))
    bridge = lens_dx - lens_r
    pygame.draw.rect(layer, INK, (cx - bridge, eyes_y - bar_h // 2, bridge * 2, bar_h))

    # Smile: a thick band with a thin line inside it, like a painted-on grin
    smile_y, a, b = int(h * 0.57), h * 0.31, h * 0.25
    def curve(start, end, steps=70):
        return [(int(cx + a * math.cos(math.radians(d))), int(smile_y + b * math.sin(math.radians(d))))
                for d in (start + (end - start) * i / steps for i in range(steps + 1))]
    _sausage(layer, curve(18, 162), int(h * 0.075), line)
    pygame.draw.lines(layer, INK, False, curve(38, 142), max(2, line // 2))
    return layer


def _glow_image(radius: int, color) -> pygame.Surface:
    """A soft round glow that fades out toward its edge."""
    image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        fade = 1 - r / radius
        pygame.draw.circle(image, (*color, int(150 * fade * fade)), (radius, radius), r)
    return image


def draw_clown(surf, t: float):
    """Clown face with glasses. The red nose slowly pulses and glows."""
    size = w, h = surf.get_size()
    key = ("clown", size)
    if key not in _static_layers:
        _static_layers[key] = _clown_static(size)
        _static_layers[("clown_glow", size)] = _glow_image(int(h * 0.30), CLOWN_RED)
    surf.blit(_static_layers[key], (0, 0))

    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi / NOSE_PULSE_S)   # goes 0 -> 1 -> 0
    center = (w // 2, int(h * 0.50))
    nose_r = int(h * 0.125 * (1 + 0.08 * pulse))

    glow_r = int(h * (0.22 + 0.08 * pulse))
    glow = pygame.transform.smoothscale(_static_layers[("clown_glow", size)], (glow_r * 2, glow_r * 2))
    surf.blit(glow, (center[0] - glow_r, center[1] - glow_r))

    pygame.draw.circle(surf, CLOWN_RED, center, nose_r)
    shine = pygame.Rect(0, 0, int(nose_r * 0.55), int(nose_r * 0.32))
    shine.center = (center[0] + int(nose_r * 0.32), center[1] - int(nose_r * 0.48))
    pygame.draw.ellipse(surf, PAPER, shine)


def draw_big_eyes(surf, t: float):
    """Two eyes that slowly swell up, then quickly pop back to normal."""
    w, h = surf.get_size()
    phase = (t % EYE_SWELL_S) / EYE_SWELL_S
    grow_part = 1 - EYE_POP_FRACTION
    if phase < grow_part:
        x = phase / grow_part
        swell = x * x * (3 - 2 * x)     # slow start, slow finish
    else:
        x = (phase - grow_part) / EYE_POP_FRACTION
        swell = (1 - x) ** 3            # snaps back quickly

    radius = int(h * 0.15 * (1 + 0.75 * swell))
    line = max(2, int(h * 0.03))
    for side in (-1, 1):
        center = (w // 2 + side * int(h * 0.27), h // 2)
        pygame.draw.circle(surf, PAPER, center, radius)
        pygame.draw.circle(surf, INK, center, radius, line)
        pygame.draw.circle(surf, BIG_EYE_BLUE, center, int(radius * 0.60))
        pygame.draw.circle(surf, INK, center, int(radius * 0.32))
        shine = (center[0] + int(radius * 0.18), center[1] - int(radius * 0.22))
        pygame.draw.circle(surf, PAPER, shine, max(2, int(radius * 0.11)))


def _eye_openness(t: float):
    """Returns (how open the eye is from 0 to 1, how many times it has changed color).
    A blink happens at the end of every BLINK_EVERY_S. The color count goes up at the
    exact moment the lid is fully shut, so you never see the color switch."""
    cycle = int(t // BLINK_EVERY_S)
    phase = t - cycle * BLINK_EVERY_S
    blink_start = BLINK_EVERY_S - BLINK_LENGTH_S
    colors_changed = cycle + (1 if phase >= blink_start + BLINK_LENGTH_S / 2 else 0)
    if phase < blink_start:
        return 1.0, colors_changed
    x = (phase - blink_start) / BLINK_LENGTH_S     # 0 -> 1 through the blink
    return abs(1 - 2 * x), colors_changed          # open -> shut -> open


def draw_iris(surf, t: float):
    """A simple eye. Every time it blinks, the iris comes back a different color."""
    w, h = surf.get_size()
    cx, cy = w // 2, h // 2
    openness, colors_changed = _eye_openness(t)
    color = IRIS_COLORS[colors_changed % len(IRIS_COLORS)]

    # The eye's outline: two curves that meet at the corners. Blinking flattens them.
    half_w, half_h = min(w * 0.42, h * 0.72), h * 0.34 * openness
    steps = 48
    xs = [-1 + 2 * i / steps for i in range(steps + 1)]
    top = [(int(cx + x * half_w), int(cy - half_h * (1 - x * x))) for x in xs]
    bottom = [(int(cx + x * half_w), int(cy + half_h * (1 - x * x))) for x in reversed(xs)]
    outline = top + bottom
    line = max(2, int(h * 0.03))

    if half_h >= 2:
        # Draw the iris on its own layer, then cut it to the shape of the open eye
        layer = pygame.Surface((w, h), pygame.SRCALPHA)
        iris_r = int(h * 0.27)
        pygame.draw.circle(layer, INK, (cx, cy), iris_r)
        pygame.draw.circle(layer, color, (cx, cy), iris_r - line)
        pygame.draw.circle(layer, INK, (cx, cy), int(iris_r * 0.42))
        pygame.draw.circle(layer, PAPER, (cx + int(iris_r * 0.3), cy - int(iris_r * 0.35)), max(2, int(iris_r * 0.16)))
        cutout = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.polygon(cutout, (255, 255, 255, 255), outline)
        layer.blit(cutout, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf.blit(layer, (0, 0))
    pygame.draw.lines(surf, INK, True, outline, line)


SCREEN_ART = {"clown": draw_clown, "bigeyes": draw_big_eyes, "iris": draw_iris}


# =============================================================================
# The whole menu
# =============================================================================


class Arcade:
    """Holds the finished room picture plus where each machine is on the screen."""

    def __init__(self, filters: list, size: tuple, title: str, symbols: list, footnote: str):
        self.filters = filters
        self.size = w, h = size
        self.time_start = pygame.time.get_ticks()

        # A wide screen gets one row. A tall one gets rows of two.
        n = len(filters)
        self.cols = n if w >= h else math.ceil(n / 2)
        self.rows = math.ceil(n / self.cols)
        row_w = self.cols * MACHINE_W + (self.cols - 1) * MACHINE_GAP
        self.design_w = max(DESIGN_W if w >= h else 900, row_w + 190)
        self.design_h = DESIGN_H + (self.rows - 1) * (MACHINE_H + ROW_GAP)

        # One scale for both directions, so shapes keep their proportions
        self.scale = min(w / self.design_w, h / self.design_h)
        self.offset_x = (w - self.design_w * self.scale) / 2
        self.offset_y = (h - self.design_h * self.scale) / 2

        self.machine_rects = [self._to_screen(self._machine_box(i)) for i in range(n)]
        self.screen_rects = [self._to_screen(self._screen_box(i)) for i in range(n)]
        self.art_surfaces = [pygame.Surface((r.w * SUPERSAMPLE, r.h * SUPERSAMPLE)) for r in self.screen_rects]

        self.footnote_text = footnote
        self.footnote_y = int(self.offset_y + self._footnote_design_y() * self.scale)
        self.background = self._paint_room(title, symbols)
        self._notes = {}  # rendered bottom-line text, so it isn't rebuilt every frame

    # --- layout helpers ---

    def _machine_box(self, i: int) -> tuple:
        """Where machine i sits on the design canvas: (x, y, w, h)."""
        row, col = divmod(i, self.cols)
        in_row = min(self.cols, len(self.filters) - row * self.cols)
        row_w = in_row * MACHINE_W + (in_row - 1) * MACHINE_GAP
        x = (self.design_w - row_w) / 2 + col * (MACHINE_W + MACHINE_GAP)
        y = MACHINES_TOP + row * (MACHINE_H + ROW_GAP)
        return x, y, MACHINE_W, MACHINE_H

    def _screen_box(self, i: int) -> tuple:
        """The area inside machine i's screen, where its animation goes."""
        x, y, w, _ = self._machine_box(i)
        return x + 42, y + 112, w - 84, 186

    def _feet_y(self) -> float:
        return MACHINES_TOP + self.rows * MACHINE_H + (self.rows - 1) * ROW_GAP

    def _footnote_design_y(self) -> float:
        return self._feet_y() + FLOOR_DEPTH + FOOTNOTE_GAP

    def _to_screen(self, box: tuple) -> pygame.Rect:
        x, y, w, h = box
        return pygame.Rect(round(self.offset_x + x * self.scale), round(self.offset_y + y * self.scale),
                           round(w * self.scale), round(h * self.scale))

    # --- drawing the parts that never change ---

    def _paint_room(self, title: str, symbols: list) -> pygame.Surface:
        """Draws the frame, title, symbols, floor and machines at 2x, then shrinks it."""
        w, h = self.size
        ss = SUPERSAMPLE
        s = self.scale * ss
        big = pygame.Surface((w * ss, h * ss))
        big.fill(PAPER)
        ox, oy = self.offset_x * ss, self.offset_y * ss

        def x_of(v): return int(ox + v * s)
        def y_of(v): return int(oy + v * s)
        def px(v): return max(1, int(v * s))
        def box(x, y, bw, bh): return pygame.Rect(x_of(x), y_of(y), int(bw * s), int(bh * s))

        def centered(image, center_x, center_y):
            big.blit(image, image.get_rect(center=(center_x, center_y)))

        # Rounded frame around the whole screen
        margin = max(4, int(min(w, h) * 0.012)) * ss
        frame = pygame.Rect(margin, margin, w * ss - 2 * margin, h * ss - 2 * margin)
        pygame.draw.rect(big, INK, frame, px(9), border_radius=px(34))

        # Title and the row of symbols under it
        center_x = w * ss // 2
        centered(render_fit(title, TITLE_FONT_PATH, px(TITLE_SIZE), frame.w - 8 * margin), center_x, y_of(TITLE_Y))
        if symbols:
            images = [render_symbol(sym, px(SUBTITLE_SIZE)) for sym in symbols]
            gap = px(46)
            x = center_x - (sum(im.get_width() for im in images) + gap * (len(images) - 1)) // 2
            for image in images:
                big.blit(image, image.get_rect(midleft=(x, y_of(SUBTITLE_Y))))
                x += image.get_width() + gap

        self._paint_floor(big, frame, y_of(self._feet_y() - FLOOR_ABOVE_FEET),
                          y_of(self._feet_y() + FLOOR_DEPTH), px(4))

        for i, flt in enumerate(self.filters):
            self._paint_machine(big, flt["name"], box, px, *self._machine_box(i))

        return pygame.transform.smoothscale(big, self.size)

    @staticmethod
    def _paint_floor(big, frame, top: int, bottom: int, line: int):
        """A tiled floor that gets wider toward the viewer."""
        center = frame.centerx
        big.set_clip(frame)                            # lines stop at the frame
        for i in range(6):                             # across: spaced wider near the viewer
            y = top + int((bottom - top) * (i / 5) ** 1.3)
            pygame.draw.line(big, INK, (frame.left, y), (frame.right, y), line)
        for i in range(-16, 17):                       # going away from the viewer
            far_x = center + i * frame.w // 20
            near_x = center + i * frame.w // 11
            pygame.draw.line(big, INK, (far_x, top), (near_x, bottom), line)
        big.set_clip(None)

    @staticmethod
    def _paint_machine(big, name: str, box, px, x, y, w, h):
        """One arcade cabinet. All numbers are design-canvas units."""
        thick, thin = px(9), px(6)

        def outlined(rect, radius, width=thick):
            pygame.draw.rect(big, PAPER, rect, border_radius=radius)
            pygame.draw.rect(big, INK, rect, width, border_radius=radius)

        body_h = h - 34
        for foot_x in (x + 34, x + w - 34 - 56):       # feet
            outlined(box(foot_x, y + body_h - 8, 56, 42), px(10))
        outlined(box(x, y, w, body_h), px(30))         # the cabinet

        # Name plate on top
        plate = box(x + 22, y + 20, w - 44, 66)
        outlined(plate, px(14), thin)
        label = render_fit(name.upper(), TITLE_FONT_PATH, px(40), plate.w - px(22))
        big.blit(label, label.get_rect(center=plate.center))

        # Screen and its frame
        outlined(box(x + 28, y + 98, w - 56, 214), px(18))

        # Control panel: a divider, a joystick, two buttons and a coin slot
        panel_y = y + 330
        pygame.draw.line(big, INK, box(x + 4, panel_y, 0, 0).topleft, box(x + w - 4, panel_y, 0, 0).topleft, thin)
        stick_x, stick_y = x + 82, panel_y + 56
        base = box(stick_x - 36, stick_y + 2, 72, 24)
        pygame.draw.ellipse(big, PAPER, base)
        pygame.draw.ellipse(big, INK, base, thin)
        pygame.draw.line(big, INK, box(stick_x, stick_y - 22, 0, 0).topleft, box(stick_x, stick_y + 12, 0, 0).topleft, thin)
        knob = box(stick_x - 15, stick_y - 40, 30, 30)
        pygame.draw.ellipse(big, PAPER, knob)
        pygame.draw.ellipse(big, INK, knob, thin)
        for bx, by in ((x + 178, panel_y + 26), (x + 226, panel_y + 40)):
            button = box(bx, by, 44, 44)
            pygame.draw.ellipse(big, PAPER, button)
            pygame.draw.ellipse(big, INK, button, thin)
        pygame.draw.rect(big, INK, box(x + w / 2 - 24, y + body_h - 22, 48, 11), border_radius=px(3))

    # --- every frame ---

    def machine_at(self, pos):
        """Which machine a tap landed on (anywhere on it counts), or None."""
        for i, rect in enumerate(self.machine_rects):
            if rect.collidepoint(pos):
                return i
        return None

    def draw(self, screen, pressed=None, message: str = ""):
        screen.blit(self.background, (0, 0))
        t = (pygame.time.get_ticks() - self.time_start) / 1000

        for flt, rect, surf in zip(self.filters, self.screen_rects, self.art_surfaces):
            art = SCREEN_ART.get(flt.get("icon"))
            if art is None:
                continue                               # a blank screen
            surf.fill(PAPER)
            art(surf, t)
            screen.blit(pygame.transform.smoothscale(surf, rect.size), rect)

        text = message or self.footnote_text
        if text not in self._notes:
            self._notes[text] = render_fit(text, TITLE_FONT_PATH, max(10, int(FOOTNOTE_SIZE * self.scale)),
                                           int(self.size[0] * 0.9))
        note = self._notes[text]
        screen.blit(note, note.get_rect(center=(self.size[0] // 2, self.footnote_y)))

        if pressed is not None:                        # darken the machine that was just tapped
            rect = self.machine_rects[pressed]
            shade = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shade, (0, 0, 0, 110), shade.get_rect(), border_radius=max(4, int(30 * self.scale)))
            screen.blit(shade, rect)
