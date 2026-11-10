"""
kiosk.py - shared touchscreen helper for the OpenCV filters.

Adds three things to a filter WITHOUT changing what the filter does:
  1. A "Back" button (top-right). Tapping it acts exactly like pressing 'q'.
  2. Optional on-screen buttons that act exactly like pressing a key
     (so keyboard controls also work on a touchscreen).
  3. Kiosk mode (only when started from launcher.py): fullscreen, and
     auto-back after AUTO_BACK_S seconds with nobody in front of the camera.

How a filter uses it (replaces its cv2.imshow + cv2.waitKey lines):

    import kiosk
    screen = kiosk.Kiosk("My Filter", buttons=[("Switch", "f")])
    ...
    key = screen.show(frame, person_seen=faces is not None)
    if key == ord('q'):
        break

show() draws the buttons on the frame, displays it, and returns the key that was
pressed, the same value as `cv2.waitKey(1) & 0xFF`. A tap on a button returns
that button's key. A tap on Back (or the auto-back timer) returns ord('q').

Run a filter directly (py saumyaa.py) and it behaves like before: a normal window,
plus the buttons, and no auto-back.
"""

import os
import time

import cv2

AUTO_BACK_S = 20.0  # kiosk mode: go back to the menu after this long with nobody there

# launcher.py sets this environment variable when it starts a filter
KIOSK_MODE = os.environ.get("FILTER_KIOSK") == "1"

BUTTON_COLOR = (60, 60, 60)        # BGR
BUTTON_ACTIVE = (40, 160, 40)
TEXT_COLOR = (255, 255, 255)


class Kiosk:
    def __init__(self, window_name: str, buttons=(), on_tap=None):
        """buttons: list of (label, key) pairs, e.g. [("Glasses", "2")].
        on_tap: optional function(x, y) for a filter that draws its own touch buttons.
                Taps that miss Back and the kiosk buttons are passed to it."""
        self.window = window_name
        self.buttons = [(label, ord(key)) for label, key in buttons]
        self.on_tap = on_tap
        self._rects = []            # [(x1, y1, x2, y2, key)] for the frame just drawn
        self._taps = []             # taps not handled yet, in image coordinates
        self._last_person = time.monotonic()
        self.active_key = None      # last button key used (highlighted)

        if KIOSK_MODE:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)  # same as a plain cv2.imshow window
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        # A touchscreen tap arrives as a left mouse click.
        if event == cv2.EVENT_LBUTTONDOWN:
            self._taps.append((x, y))

    # --- drawing ---

    def _draw_button(self, frame, x1, y1, x2, y2, label, color, scale):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), TEXT_COLOR, max(1, int(scale)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, int(2 * scale)))
        cv2.putText(frame, label, ((x1 + x2 - tw) // 2, (y1 + y2 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, TEXT_COLOR, max(1, int(2 * scale)), cv2.LINE_AA)

    def draw(self, frame):
        """Draw Back (top-right) and the extra buttons (bottom row). Sizes scale with the frame."""
        h, w = frame.shape[:2]
        s = w / 640                      # 1.0 for a 640-wide frame
        self._rects = []

        bw, bh, m = int(110 * s), int(46 * s), int(10 * s)
        x1, y1 = w - m - bw, m
        self._draw_button(frame, x1, y1, x1 + bw, y1 + bh, "< Back", BUTTON_COLOR, s)
        self._rects.append((x1, y1, x1 + bw, y1 + bh, ord("q")))

        if self.buttons:
            n = len(self.buttons)
            gap = int(8 * s)
            bw = min(int(150 * s), (w - 2 * m - (n - 1) * gap) // n)
            total = n * bw + (n - 1) * gap
            x = (w - total) // 2
            y1 = h - m - bh
            for label, key in self.buttons:
                color = BUTTON_ACTIVE if key == self.active_key else BUTTON_COLOR
                self._draw_button(frame, x, y1, x + bw, y1 + bh, label, color, s)
                self._rects.append((x, y1, x + bw, y1 + bh, key))
                x += bw + gap
        return frame

    # --- the main call ---

    def show(self, frame, person_seen: bool = True) -> int:
        """Draw the buttons, show the frame, and return the key pressed (or tapped).
        person_seen: is someone in front of the camera? (for the auto-back timer)"""
        now = time.monotonic()
        if person_seen:
            self._last_person = now

        self.draw(frame)
        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF   # taps are collected while this runs

        for tx, ty in self._taps:
            hit = False
            for x1, y1, x2, y2, button_key in self._rects:
                if x1 <= tx <= x2 and y1 <= ty <= y2:
                    key = button_key
                    hit = True
            if not hit and self.on_tap is not None:
                self.on_tap(tx, ty)   # the filter's own buttons
        self._taps.clear()

        if key in [k for _, k in self.buttons]:
            self.active_key = key
        if KIOSK_MODE and now - self._last_person > AUTO_BACK_S:
            print(f"Nobody there for {AUTO_BACK_S:.0f}s: going back to the menu.")
            key = ord("q")
        return key

    def ask(self, cap, message: str, yes_label: str = "Start", yes_key: str = "y",
            flip: bool = True) -> bool:
        """An on-screen yes/no question over the live camera, e.g. "Enable the filter?".
        Returns True if they tap the yes button (or press yes_key), False if they tap
        Back or press any other key."""
        start_time = time.monotonic()
        yes = ord(yes_key)
        while True:
            ok, frame = cap.read()
            if not ok:
                return False
            if flip:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            s = w / 640
            (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, max(1, int(2 * s)))
            cv2.putText(frame, message, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9 * s, TEXT_COLOR, max(1, int(2 * s)), cv2.LINE_AA)

            saved = self.buttons
            self.buttons = [(yes_label, yes)]
            key = self.show(frame, person_seen=True)
            self.buttons = saved
            self.active_key = None

            if key == yes:
                return True
            if key != 255:  # 255 = no key pressed
                return False
            if KIOSK_MODE and time.monotonic() - start_time > AUTO_BACK_S:
                return False
