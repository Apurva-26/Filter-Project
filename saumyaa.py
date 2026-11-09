"""
Hallway Display Filter: recolors the iris of every opted-in face.

Two live rendering modes (switch any time from the video window):
  1 = Bright -> Dynamic Pigment Engine: a procedural iris re-lit every frame from the real
                eye (exposure, color cast, light direction, eyelid shadow, catchlights).
  2 = Dark   -> the real iris cut out of the "* Iris.png" photos is warped onto the
                measured iris ellipse and clipped by the eyelids.

Distance zone (calibrated from two photos: 33 cm and 1.5 m from the camera):
  closer than 33 cm -> face blurred + "Too close! Step back."
  33 cm .. 1.5 m    -> hold still 2 s to opt in, then the iris filter is applied
  farther than 1.5 m -> face blurred only
Once opted in, a person keeps the filter and is never blurred while they move around,
turn or look away -- until they walk out of the webcam's view.  (The too-close warning
still shows for them.)  People who have not opted in follow the zone rules above.

Touchscreen (or mouse): tap the < > arrows on the top bar to switch Bright / Dark and the
       < > arrows on the bottom bar to pick the iris color (shown with a live swatch).
Keys:  1 / 2 = switch mode    C = next iris color    Q / Esc / window X = quit
The selected color is used for everyone being filtered and for each new person who opts in.

Uses MediaPipe FaceLandmarker (478 points: exact iris & 16-point eyelid contours) plus the
YuNet face detector to catch small, far-away faces FaceLandmarker misses (so they get blurred).
"""
import glob
import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
YUNET_PATH = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
WINDOW = "Hallway Iris Filter"

# 16-point eyelid outlines + iris center/rim indices in MediaPipe's 478-point mesh.
# Outline order: indices 0..8 run along the lower lid, 8..15 + 0 along the upper lid.
LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
UPPER_LID_IDX = [8, 9, 10, 11, 12, 13, 14, 15, 0]
LEFT_IRIS_CENTER = 468
LEFT_IRIS_RIM = [469, 470, 471, 472]       # [right, top, left, bottom]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RIM = [474, 475, 476, 477]      # [right, top, left, bottom]

# ---------------------------------------------------------------------------------------------
# Dynamic Pigment Engine palette (BGR).  dark = limbal ring / crypts, base = stroma,
# light = collarette / fiber highlights, hue_shift = OpenCV hue units added in shadows.
# ---------------------------------------------------------------------------------------------
PIGMENTS = {
    "Blue":        {"dark": (95, 50, 20),   "base": (195, 125, 60),  "light": (240, 195, 135), "hue_shift": 5},
    "Dark Brown":  {"dark": (8, 14, 26),    "base": (25, 45, 78),    "light": (50, 90, 135),   "hue_shift": -3},
    "Emerald":     {"dark": (30, 65, 10),   "base": (95, 155, 35),   "light": (150, 215, 100), "hue_shift": 6},
    "Gold":        {"dark": (10, 70, 115),  "base": (25, 165, 220),  "light": (95, 225, 255),  "hue_shift": -5},
    "Grey":        {"dark": (68, 66, 63),   "base": (150, 146, 140), "light": (215, 212, 206), "hue_shift": 0},
    "Light Brown": {"dark": (15, 40, 72),   "base": (40, 98, 165),   "light": (90, 160, 220),  "hue_shift": -3},
    "Pink":        {"dark": (95, 20, 135),  "base": (175, 65, 245),  "light": (225, 160, 255), "hue_shift": -5},
    "Red":         {"dark": (10, 5, 72),    "base": (28, 18, 175),   "light": (70, 60, 235),   "hue_shift": 3},
}
PUPIL_BGR = (14, 11, 11)
TEX_SIZE = 256                             # iris textures: disk centred at 127.5, radius 127.5
TEX_C = (TEX_SIZE - 1) / 2.0
TEX_R = TEX_SIZE / 2.0 - 0.5

MODE_LABELS = {
    1: "Mode 1: Bright",
    2: "Mode 2: Dark",
}

# Tuning knobs
IRIS_RADIUS_RATIO = 0.10   # fallback ratio if iris rim points aren't usable
SMOOTHING = 0.45           # eye-center smoothing for tracking (0 = none, ->1 = steadier)
IRIS_SMOOTHING = 0.5       # iris ellipse (cx, cy, rx, ry, angle) smoothing
MAX_JUMP_RATIO = 0.6       # ignore an eye point that jumps more than this * eye distance
NATURAL_OPACITY = 0.90
PNG_OPACITY = 0.94
PNG_FEATHER_PX = 1.75      # anti-aliased limbal edge width in Mode 2
LID_SHADOW_NATURAL = 0.38  # strength of the upper-eyelid shadow
LID_SHADOW_PNG = 0.22

# Opt-in zone
MIN_DIST_M = 0.33          # closer than this = blurred + "Too close! Step back."
MAX_DIST_M = 1.5           # farther than this = bystander (blurred)
MAX_HORIZ_OFFSET_M = 0.5   # must be within 0.5 m left/right of the camera center
HOLD_SECONDS = 2.0         # hold still for 2 seconds to opt in
FORGET_AFTER_S = 0.75      # forget person once they step out

# Distance calibration, measured on this webcam at 1280x720:
#   33 cm  -> iris centers 149.8 px apart (0.1170 of frame width)
#   1.5 m  -> iris centers  38.1 px apart (0.0298 of frame width)
# Two-point fit: distance_m = DIST_A / (eye_px / frame_width) + DIST_B
DIST_A = 0.046707
DIST_B = -0.0691
REAL_EYE_DIST_M = 0.063
YUNET_WIDTH = 640          # YuNet runs on a copy this wide (fast, still finds far faces)
YUNET_EYE_SCALE = 1.08     # YuNet eye-corner spacing -> iris-center spacing (measured)

# Turning the head shrinks the on-screen eye spacing, which would look like walking away.
# Face height (forehead to chin) barely changes with a head turn, so the distance uses
# scale = max(eye spacing, ratio * face height).  Ratios measured on the calibration photos.
FACE_HEIGHT_RATIO = 0.43   # iris spacing / landmark face height (10 -> 152), facing camera
YUNET_BOX_RATIO = 0.37     # iris spacing / YuNet face-box height, facing camera

# "Staring at the camera" test used for opting in (passersby who look away get blurred).
FACING_MAX_YAW = 0.30      # head turn: nose offset from face center / half face width
FACING_MAX_GAZE = 0.35     # eyes turned: iris offset from eye center / half eye width
FACING_GRACE_S = 0.3       # brief glances away are forgiven while holding to opt in
OPTED_IN_MARGIN = 0.10     # opted-in people get 10% extra zone before they lose the filter
HIDDEN_EYE_RATIO = 0.35    # skip drawing an eye that is this much narrower (head in profile)

# Once opted in, a person keeps the filter (never blurred, never asked to opt in again)
# wherever they move, until they leave the webcam's view.
OPTED_IN_MEMORY_S = 5.0    # face lost mid-frame (motion blur, turned away): remember this long
EDGE_MARGIN = 0.03         # face last seen this close to a frame edge = walked out of view
OPTED_IN_MATCH = 3.0       # an opted-in face may move this many face-scales between sightings


def distance_m(eye_px, frame_w) -> float:
    """Camera-to-face distance in meters from the pixel spacing of the two irises."""
    return DIST_A / max(eye_px / float(frame_w), 1e-4) + DIST_B


def load_model() -> None:
    """Download the MediaPipe FaceLandmarker model the first time the script runs."""
    if os.path.exists(TASK_PATH) and os.path.getsize(TASK_PATH) > 1_000_000:
        return
    print("Downloading MediaPipe FaceLandmarker model (one time only)...")
    urllib.request.urlretrieve(TASK_URL, TASK_PATH)
    print("Done.")


# =============================================================================================
# Mode 1 -- Dynamic Pigment Engine: procedural iris structure + pigment
# =============================================================================================
def make_iris_structure(size=TEX_SIZE, seed=0) -> dict:
    """Procedural iris anatomy as grey-level maps: radial stroma fibers, crypts, a wavy
    collarette, a dark limbal ring and an irregular pupil edge.  tone: 0 = dark, 0.5 = base,
    1 = light pigment.  pupil: 0 inside the pupil, 1 on the iris."""
    rng = np.random.default_rng(seed)
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - c, yy - c) / (size / 2.0)
    theta = np.arctan2(yy - c, xx - c)

    fibers = np.zeros_like(r)
    for f in (37, 61, 97, 151, 223):
        warp = rng.uniform(0.8, 2.0) * np.sin(theta * rng.integers(2, 5) + rng.uniform(0, 6.3)) * r
        fibers += np.sin(theta * f + warp + rng.uniform(0, 2 * np.pi)) / math.sqrt(f)
    fibers /= np.abs(fibers).max() + 1e-6

    crypts = np.zeros_like(r)
    for _ in range(22):
        rr, tt = rng.uniform(0.42, 0.82), rng.uniform(0, 2 * np.pi)
        px, py = c + rr * math.cos(tt) * size / 2, c + rr * math.sin(tt) * size / 2
        sig = rng.uniform(0.02, 0.045) * size
        crypts += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sig ** 2))
    crypts = np.clip(crypts, 0, 1)

    collar_r = 0.43 + 0.03 * np.sin(7 * theta + rng.uniform(0, 6.3)) + 0.015 * np.sin(13 * theta)
    collarette = np.exp(-((r - collar_r) ** 2) / (2 * 0.035 ** 2))
    limbal = np.clip((r - 0.78) / 0.22, 0, 1) ** 1.5
    pupil_ruff = np.exp(-((r - 0.31) ** 2) / (2 * 0.02 ** 2))

    tone = (0.55 + 0.22 * fibers + 0.26 * collarette - 0.30 * crypts
            - 0.55 * limbal - 0.22 * pupil_ruff)
    pupil_r = 0.27 + 0.008 * np.sin(9 * theta + rng.uniform(0, 6.3))
    pupil = np.clip((r - pupil_r) / 0.025, 0, 1)
    return {"tone": np.clip(tone, 0, 1).astype(np.float32), "pupil": pupil.astype(np.float32)}


def colorize_pigment(structure, pigment) -> np.ndarray:
    """Turn the grey iris structure into an evenly lit BGR iris for one pigment."""
    t = structure["tone"][..., None]
    dark, base, light = (np.array(pigment[k], np.float32) for k in ("dark", "base", "light"))
    lo = dark + (base - dark) * np.clip(t / 0.5, 0, 1)
    hi = base + (light - base) * np.clip((t - 0.5) / 0.5, 0, 1)
    col = np.where(t < 0.5, lo, hi)
    p = structure["pupil"][..., None]
    col = col * p + np.array(PUPIL_BGR, np.float32) * (1 - p)
    return np.clip(col, 0, 255).astype(np.uint8)


def pigment_from_texture(tex) -> dict:
    """Derive a pigment for a PNG-only color (e.g. iris_teal.png) from its iris ring."""
    yy, xx = np.mgrid[0:TEX_SIZE, 0:TEX_SIZE]
    rr = np.hypot(xx - TEX_C, yy - TEX_C) / TEX_R
    base = np.median(tex[(rr > 0.45) & (rr < 0.85)], 0).astype(np.float32)
    return {"dark": tuple(np.clip(base * 0.45, 0, 255)), "base": tuple(base),
            "light": tuple(np.clip(base * 1.35 + 20, 0, 255)), "hue_shift": 0}


def build_natural_textures(color_names) -> dict:
    return {name: colorize_pigment(make_iris_structure(seed=i + 1), PIGMENTS[name])
            for i, name in enumerate(color_names)}


# =============================================================================================
# Mode 2 -- automatic iris-only extraction from the close-up "* Iris.png" photos
# =============================================================================================
def _whiteness(bgr) -> np.ndarray:
    """Bright AND colorless = sclera.  Vivid irises (gold, pink) score low even when bright."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    return np.clip(lab[..., 0] - 1.5 * chroma, 0, 255)


def _limbus_scores(W, centers, radii) -> np.ndarray:
    """Daugman-style limbus score for many candidate centers at once -> (n_centers, n_radii).
    Rewards a sharp dark->white step on BOTH the left and right iris edges, bright sclera
    just outside, and a darker iris inside (top/bottom are ignored: eyelids live there)."""
    ar = np.radians(np.linspace(-35, 35, 24))
    al = np.radians(np.linspace(145, 215, 24))
    ang = np.concatenate([ar, al])
    mx = (centers[:, 0][:, None, None] + radii[None, None, :] * np.cos(ang)[None, :, None])
    my = (centers[:, 1][:, None, None] + radii[None, None, :] * np.sin(ang)[None, :, None])
    n, a, r = mx.shape
    mx = mx.reshape(n * a, r).astype(np.float32)
    my = my.reshape(n * a, r).astype(np.float32)
    step = 30000                                             # remap rows must stay < 32767
    s = np.concatenate([cv2.remap(W, mx[i:i + step], my[i:i + step], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
                        for i in range(0, n * a, step)]).reshape(n, a, r)
    k = len(ar)
    sr, sl = s[:, :k].mean(1), s[:, k:].mean(1)
    edge = np.zeros_like(sr)
    edge[:, 2:-2] = np.minimum(sr[:, 4:] - sr[:, :-4], sl[:, 4:] - sl[:, :-4])
    outside = np.zeros_like(sr)
    outside[:, :-4] = np.minimum(sr[:, 4:], sl[:, 4:])
    both = (sr + sl) / 2
    cs = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(both, 1)], 1)
    idx = np.arange(r)
    lo = idx // 2
    inner = (cs[:, idx + 1] - cs[:, lo]) / (idx + 1 - lo)[None]
    return edge + 0.5 * (outside - inner)


def _pick_radius(profile, radii, rmin, rmax):
    """Outermost strong peak (the limbus, not the pupil edge or a bright collarette)."""
    ok = (radii >= rmin) & (radii <= rmax)
    p = np.where(ok, profile, -1e9)
    best = p.max()
    peaks = [i for i in range(1, len(p) - 1)
             if ok[i] and p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] >= 0.35 * best]
    i = peaks[-1] if peaks else int(np.argmax(p))
    return float(p[i]), float(radii[i])


def detect_iris_cv(bgr, r_range=None):
    """Find the iris (cx, cy, r, score) in a close-up eye photo without a face detector.
    Coarse search on a 120px-high copy, then sub-pixel refinement at full resolution."""
    h, w = bgr.shape[:2]
    work_h = 120
    sc = work_h / h
    small = cv2.resize(bgr, (max(8, int(round(w * sc))), work_h), interpolation=cv2.INTER_AREA)
    W = cv2.GaussianBlur(_whiteness(small), (5, 5), 0)
    if r_range is None:
        rmin, rmax = 0.10 * work_h, 0.32 * work_h
    else:
        rmin, rmax = r_range[0] * sc, r_range[1] * sc
    m = max(2, int(rmin * 0.6))
    ys, xs = np.mgrid[m:work_h - m:2, m:small.shape[1] - m:2]
    ys, xs = ys.ravel(), xs.ravel()
    if len(xs) == 0:
        return None
    dark = cv2.blur(W, (7, 7))                     # the iris is never sclera-white
    keep = dark[ys, xs] <= np.percentile(dark, 40)
    centers = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
    if len(centers) == 0:
        return None
    radii = np.arange(max(3.0, rmin * 0.7), rmax * 1.1, 1.0, dtype=np.float32)
    prof = _limbus_scores(W, centers, radii)
    best = (-1e9, 0.0, 0.0, 0.0)
    for i in np.argsort(-prof.max(1))[:60]:
        s, r = _pick_radius(prof[i], radii, rmin, rmax)
        if s > best[0]:
            best = (s, float(centers[i, 0]), float(centers[i, 1]), r)
    _, x, y, r = best

    Wf = cv2.GaussianBlur(_whiteness(bgr), (3, 3), 0)
    X, Y, R = x / sc, y / sc, r / sc
    g = np.arange(-3.0, 3.01, 0.5, dtype=np.float32)
    gy, gx = np.meshgrid(g + Y, g + X, indexing="ij")
    cands = np.stack([gx.ravel(), gy.ravel()], 1)
    radii = np.arange(R * 0.8, R * 1.2, 0.5, dtype=np.float32)
    prof = _limbus_scores(Wf, cands, radii)
    i = int(np.argmax(prof.max(1)))
    s2, r2 = _pick_radius(prof[i], radii, R * 0.85, R * 1.15)
    return float(cands[i, 0]), float(cands[i, 1]), r2, s2


def _sclera_iris_refs(W, cx, cy, r):
    """Median whiteness of the sclera beside the iris and of the iris itself."""
    h, w = W.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    side = np.abs(yy - cy) < 0.4 * r
    scl = side & (dist > 1.12 * r) & (dist < 1.45 * r)
    iri = side & (dist > 0.5 * r) & (dist < 0.9 * r)
    s_ref = float(np.median(W[scl])) if scl.any() else 200.0
    i_ref = float(np.median(W[iri])) if iri.any() else 60.0
    return s_ref, i_ref


def fit_eyelids(bgr, cx, cy, r):
    """Fit y = a*x^2 + b*x + c to the upper and lower eyelid margins, traced along the sclera
    on both sides of the iris (top edge of the sclera = upper lid, bottom edge = lower lid)."""
    h, w = bgr.shape[:2]
    W = _whiteness(bgr)
    s_ref, i_ref = _sclera_iris_refs(W, cx, cy, r)
    sclera = (W > i_ref + 0.5 * (s_ref - i_ref)).astype(np.uint8)
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, lbl = cv2.connectedComponents(sclera)
    keep = set()
    for sx in (cx - 1.25 * r, cx + 1.25 * r):
        for dy in (-0.15 * r, 0.0, 0.15 * r):
            x, y = int(round(sx)), int(round(cy + dy))
            if 0 <= x < w and 0 <= y < h and lbl[y, x] > 0:
                keep.add(int(lbl[y, x]))
    if not keep:
        return None, None
    mask = np.isin(lbl, list(keep))
    y0 = max(0, int(cy - 1.6 * r))
    up_pts, lo_pts, sides = [], [], set()
    for x in range(max(0, int(cx - 2.4 * r)), min(w, int(cx + 2.4 * r) + 1)):
        if abs(x - cx) < 1.08 * r:
            continue
        ys = np.flatnonzero(mask[y0:int(cy + 1.6 * r), x]) + y0
        if len(ys) < 2:
            continue
        up_pts.append((x, ys[0]))
        lo_pts.append((x, ys[-1]))
        sides.add(x > cx)

    def fit(pts):
        if len(pts) < 4:
            return None
        p = np.array(pts, np.float64)
        deg = 2 if len(sides) == 2 else 1
        coef = np.polyfit(p[:, 0], p[:, 1], deg)
        res = np.abs(np.polyval(coef, p[:, 0]) - p[:, 1])
        good = res <= max(2.0, 2.5 * np.median(res))
        if good.sum() >= deg + 2:
            coef = np.polyfit(p[good, 0], p[good, 1], deg)
        return np.pad(coef, (3 - len(coef), 0))

    return fit(up_pts), fit(lo_pts)


def build_iris_texture(bgr, cx, cy, r):
    """Cut the iris disk (cx, cy, r) out of a photo into a 256x256 texture that contains ONLY
    iris + pupil.  Catchlights are inpainted, eyelid/lash-covered parts are rebuilt from the
    mirrored visible iris, and baked-in lighting is flattened (Mode 2 re-lights it live).
    Returns (texture, visible_fraction, sclera_fraction)."""
    n_ang, n_rad = 360, 128
    ih, iw = bgr.shape[:2]

    # 1. Remove the photo's corneal catchlights (small bright spots) before unwrapping.
    x1, y1 = max(0, int(cx - r * 1.2)), max(0, int(cy - r * 1.2))
    x2, y2 = min(iw, int(cx + r * 1.2) + 1), min(ih, int(cy + r * 1.2) + 1)
    crop = bgr[y1:y2, x1:x2].copy()
    Lc = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)[..., 0]
    k = max(5, int(r * 0.5) | 1)
    tophat = Lc.astype(np.float32) - cv2.medianBlur(Lc, k).astype(np.float32)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    inside = np.hypot(xx - cx, yy - cy) < r
    glint = ((tophat > 30) & (Lc > 120) & inside).astype(np.uint8)
    glint = cv2.dilate(glint, np.ones((3, 3), np.uint8), iterations=2)
    if glint.any():
        crop = cv2.inpaint(crop, glint * 255, 3, cv2.INPAINT_TELEA)

    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rad = (np.arange(n_rad) + 0.5) / n_rad * r
    px = cx + rad[None, :] * np.cos(ang)[:, None]
    py = cy + rad[None, :] * np.sin(ang)[:, None]
    polar = cv2.remap(crop, (px - x1).astype(np.float32), (py - y1).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2. Eyelid occlusion from parabolas fitted to the real lid margins.
    upper, lower = fit_eyelids(bgr, cx, cy, r)
    margin = 0.16 * r                       # lashes + lid shadow hang just below the margin
    occluded = np.zeros((n_ang, n_rad), bool)
    if upper is not None:
        occluded |= py < np.polyval(upper, px) + margin
    if lower is not None:
        occluded |= py > np.polyval(lower, px) - margin * 0.5
    occluded[:, : int(0.30 * n_rad)] = False          # never touch the pupil
    visible = 1.0 - float(occluded.mean())

    # 3. Flatten the photo's lighting (lid shadow, side light) while keeping the iris's own
    #    radial profile (pupil, collarette, limbal ring).
    valid = (~occluded).astype(np.float32)
    lum = cv2.cvtColor(polar, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0

    def tiled(m):
        return np.concatenate([m[-60:], m, m[:60]])

    num = cv2.GaussianBlur(tiled(lum * valid), (15, 61), 0)[60:-60]
    den = cv2.GaussianBlur(tiled(valid), (15, 61), 0)[60:-60]
    illum = num / np.maximum(den, 1e-3)
    prof = np.array([np.median(lum[valid[:, j] > 0, j]) if (valid[:, j] > 0).any()
                     else lum[:, j].mean() for j in range(n_rad)], np.float32)
    prof = cv2.GaussianBlur(prof[None], (15, 1), 0)[0]
    gain = np.clip(prof[None] / np.maximum(illum, 1.0), 0.6, 1.8)
    ramp = np.clip((np.arange(n_rad) / n_rad - 0.15) / 0.25, 0, 1)[None]
    gain = 1.0 + (gain - 1.0) * ramp
    polar = np.clip(polar.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)

    # 4. Rebuild hidden iris from the mirror angle (top <-> bottom), feathered at the lid line.
    a = np.arange(n_ang)
    wgt = cv2.GaussianBlur(occluded.astype(np.float32), (25, 13), 0)
    wgt = np.maximum(wgt * 2.5, occluded).clip(0, 1)
    for mirror in ((-a) % n_ang, (180 - a) % n_ang):
        src_ok = ~occluded[mirror]
        ww = np.where(src_ok, wgt, 0.0)[..., None]
        polar = (polar * (1 - ww) + polar[mirror] * ww).astype(np.uint8)
        occluded = occluded & ~src_ok
        wgt = np.where(src_ok, 0.0, wgt)
    if occluded.any():
        tmask = tiled(occluded).astype(np.uint8) * 255
        polar = cv2.inpaint(tiled(polar), tmask, 5, cv2.INPAINT_TELEA)[60:-60]

    # 5. Back to Cartesian: the iris disk fills radius 127.5 around (127.5, 127.5).
    ty, tx = np.mgrid[0:TEX_SIZE, 0:TEX_SIZE].astype(np.float32)
    rr = np.hypot(tx - TEX_C, ty - TEX_C) / TEX_R
    rho = np.clip(np.minimum(rr, 1.0) * n_rad - 0.5, 0, n_rad - 1)
    theta = (np.degrees(np.arctan2(ty - TEX_C, tx - TEX_C)) % 360.0) * n_ang / 360.0
    wrapped = np.concatenate([polar, polar[:1]])
    tex = cv2.remap(wrapped, rho.astype(np.float32), theta.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Corners outside the disk: plain limbal-ring color (never sclera or skin).
    rim = np.median(tex[(rr > 0.9) & (rr <= 1.0)], 0)
    tex[rr > 1.0] = rim.astype(np.uint8)

    # Self-check: how much of the disk still looks like eye-white?
    s_ref, i_ref = _sclera_iris_refs(_whiteness(bgr), cx, cy, r / 0.97)
    thr = i_ref + 0.6 * (s_ref - i_ref)
    sclera_frac = float((_whiteness(tex)[rr <= 1.0] > thr).mean()) if s_ref > i_ref + 20 else 0.0
    return tex, visible, sclera_frac


def _png_color_name(filename):
    base = os.path.splitext(os.path.basename(filename))[0].strip()
    if base.lower().endswith(" iris"):
        base = base[:-5]
    elif base.lower().startswith("iris_"):
        base = base[5:]
    else:
        return None
    return " ".join(part.capitalize() for part in base.replace("_", " ").split())


def _landmarker_iris_guesses(landmarker, img):
    """Stage 1: ask FaceLandmarker for both irises (raw image, then on padded canvases --
    a tight two-eye crop usually has too little face for the detector)."""
    h, w = img.shape[:2]
    fill = np.median(np.concatenate([img[0], img[-1]]), 0).tolist()
    variants = [(img, 0, 0)]
    for py_, px_ in ((1.0, 0.25), (1.5, 0.5)):
        t, l = int(h * py_), int(w * px_)
        variants.append((cv2.copyMakeBorder(img, t, t, l, l, cv2.BORDER_REPLICATE), l, t))
        variants.append((cv2.copyMakeBorder(img, t, t, l, l, cv2.BORDER_CONSTANT, value=fill), l, t))
    for canvas, ox, oy in variants:
        rgb = np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            continue
        lm = res.face_landmarks[0]
        ch, cw = canvas.shape[:2]
        pts = np.array([[p.x * cw - ox, p.y * ch - oy] for p in lm], np.float32)
        out = []
        for c_idx, rim_idx in ((LEFT_IRIS_CENTER, LEFT_IRIS_RIM), (RIGHT_IRIS_CENTER, RIGHT_IRIS_RIM)):
            c = pts[c_idx]
            r = float(np.mean(np.linalg.norm(pts[rim_idx] - c, axis=1)))
            if 0 <= c[0] < w and 0 <= c[1] < h and r > 3:
                out.append((float(c[0]), float(c[1]), r))
        if out:
            return out
    return []


def _iris_candidates(img, landmarker):
    """(cx, cy, r, score, source) candidates for every eye in the photo."""
    h, w = img.shape[:2]
    cands = []
    for gx, gy, gr in (_landmarker_iris_guesses(landmarker, img) if landmarker else []):
        x1, y1 = max(0, int(gx - 2.8 * gr)), max(0, int(gy - 2.0 * gr))
        x2, y2 = min(w, int(gx + 2.8 * gr)), min(h, int(gy + 2.0 * gr))
        det = detect_iris_cv(img[y1:y2, x1:x2], (0.75 * gr, 1.3 * gr))
        if det and math.hypot(det[0] + x1 - gx, det[1] + y1 - gy) < 0.6 * gr:
            cands.append((det[0] + x1, det[1] + y1, det[2], det[3], "FaceLandmarker+refine"))
        else:
            cands.append((gx, gy, gr, 0.0, "FaceLandmarker"))
    if cands:
        return cands

    # Stage 2: pure computer vision, one search per image half (one eye each).
    halves = [(0, img[:, : w // 2]), (w // 2, img[:, w // 2:])]
    dets = [detect_iris_cv(half) for _, half in halves]
    found = [d for d in dets if d]
    if len(found) == 2:                    # both eyes of one face share the same iris size
        rs = [d[2] for d in dets]
        if max(rs) > 1.2 * min(rs):
            i = int(np.argmin(rs))
            R = max(rs)
            dets[i] = detect_iris_cv(halves[i][1], (0.85 * R, 1.15 * R)) or dets[i]
    for (x0, _), d in zip(halves, dets):
        if d:
            cands.append((d[0] + x0, d[1], d[2], d[3], "limbus search"))
    return cands


def load_png_irises(model_buffer=None, verbose=True) -> dict:
    """Load every '* Iris.png' / 'iris_*.png' and return {color name: 256x256 iris texture}."""
    paths = []
    for folder in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "images")):
        paths += sorted(glob.glob(os.path.join(folder, "* Iris.png")))
        paths += sorted(glob.glob(os.path.join(folder, "iris_*.png")))
    landmarker = None
    if paths and model_buffer is not None:
        try:
            landmarker = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=model_buffer),
                running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
        except Exception as exc:              # still fine: the CV stage handles these crops
            print(f"  (FaceLandmarker unavailable for PNGs: {exc})")

    textures = {}
    for path in paths:
        name = _png_color_name(path)
        if not name or name in textures:
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 20:
            print(f"  {os.path.basename(path)}: could not read -- skipped")
            continue
        try:
            best = None
            for cx, cy, r, score, source in _iris_candidates(img, landmarker):
                r *= 0.97                       # stay just inside the limbus: no sclera
                tex, vis, scl = build_iris_texture(img, cx, cy, r)
                rank = (score + 1.0) * (0.5 + 0.5 * vis) * (1.0 - scl)
                if best is None or rank > best[0]:
                    best = (rank, tex, cx, cy, r, vis, scl, source)
            if best is None:
                print(f"  {os.path.basename(path)}: no iris found -- skipped")
                continue
            _, tex, cx, cy, r, vis, scl, source = best
            textures[name] = tex
            if verbose:
                print(f"  {os.path.basename(path):24s} -> {name:12s} iris c=({cx:5.1f},{cy:5.1f}) "
                      f"r={r:4.1f}px  visible={vis:4.0%}  sclera-left={scl:5.1%}  [{source}]")
        except Exception as exc:
            print(f"  {os.path.basename(path)}: extraction failed ({exc}) -- skipped")
    if landmarker:
        landmarker.close()
    return textures


# =============================================================================================
# Tracking
# =============================================================================================
class Person:
    def __init__(self, left_eye, right_eye, now, scale=None):
        self.left = np.array(left_eye, np.float32)
        self.right = np.array(right_eye, np.float32)
        self.scale = float(scale) if scale else self.eye_dist()   # head-turn-proof size (px)
        self.last_seen = now
        self.last_facing = None
        self.in_zone_since = None
        self.color = None                            # set = opted in
        self.bbox = None
        self.iris = {"left": None, "right": None}   # smoothed [cx, cy, rx, ry, angle_deg]

    @property
    def opted_in(self):
        return self.color is not None

    def center(self):
        return (self.left + self.right) / 2

    def eye_dist(self):
        return float(np.linalg.norm(self.right - self.left))

    def update(self, left_eye, right_eye, now, scale=None):
        """Smooth small jitter; follow big jumps immediately (the person is moving fast)."""
        max_jump = MAX_JUMP_RATIO * max(self.scale, self.eye_dist(), 1.0)
        for name, new in (("left", left_eye), ("right", right_eye)):
            new = np.array(new, np.float32)
            old = getattr(self, name)
            a = SMOOTHING if np.linalg.norm(new - old) <= max_jump else 0.0
            setattr(self, name, a * old + (1 - a) * new)
        if scale:
            self.scale = SMOOTHING * self.scale + (1 - SMOOTHING) * float(scale)
        self.last_seen = now

    def follow_box(self, bbox, scale, now):
        """Track the person from a face box alone (FaceLandmarker lost them for now)."""
        if self.bbox is not None:
            shift = np.array(_box_center(bbox)) - np.array(_box_center(self.bbox))
            self.left = self.left + shift.astype(np.float32)
            self.right = self.right + shift.astype(np.float32)
        self.bbox = bbox
        self.scale = SMOOTHING * self.scale + (1 - SMOOTHING) * float(scale)
        self.last_seen = now

    def gone(self, now, frame_shape):
        """True once the person has left the webcam's view."""
        age = now - self.last_seen
        if not self.opted_in or self.bbox is None:
            return age > FORGET_AFTER_S
        f_h, f_w = frame_shape[:2]
        x, y, w, h = self.bbox
        mx, my = EDGE_MARGIN * f_w, EDGE_MARGIN * f_h
        at_edge = x <= mx or y <= my or x + w >= f_w - mx or y + h >= f_h - my
        return age > (FORGET_AFTER_S if at_edge else OPTED_IN_MEMORY_S)

    def match_limit(self, scale, now):
        """How far (px) a new face may be from this person and still be them."""
        s = max(scale, self.scale)
        if not self.opted_in:
            return 2.0 * s
        gap = min(now - self.last_seen, 1.0)          # lost longer -> may have moved farther,
        return OPTED_IN_MATCH * s * (1.0 + gap)       # capped so a stranger can't inherit it

    def update_iris(self, side, params):
        """Adaptive smoothing: steady when still, snaps quickly on real movement."""
        new = np.array(params, np.float32)
        old = self.iris[side]
        if old is None:
            self.iris[side] = new
            return
        move = float(np.linalg.norm(new[:2] - old[:2])) / max(old[2], old[3], 1.0)
        a = IRIS_SMOOTHING if move < 0.35 else (0.2 if move < 1.0 else 0.0)
        out = a * old + (1 - a) * new
        d_ang = (new[4] - old[4] + 180.0) % 360.0 - 180.0
        out[4] = old[4] + (1 - a) * d_ang
        self.iris[side] = out


def _box_center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def forget_gone_people(people, now, frame_shape):
    people[:] = [p for p in people if not p.gone(now, frame_shape)]


def match_people(people, faces, now):
    """faces: [(left_eye, right_eye, scale_px)].  Closest pairs are matched first, opted-in
    people win ties, and opted-in people may have moved farther (they keep their opt-in)."""
    pairs = []
    for fi, (left_eye, right_eye, scale) in enumerate(faces):
        center = (np.array(left_eye) + np.array(right_eye)) / 2
        for p in people:
            d = float(np.linalg.norm(p.center() - center))
            limit = p.match_limit(scale, now)
            if d < limit:
                pairs.append((d / limit, not p.opted_in, fi, p))
    pairs.sort(key=lambda t: (t[0], t[1]))
    matched, used = [None] * len(faces), set()
    for _, _, fi, p in pairs:
        if matched[fi] is None and id(p) not in used:
            matched[fi] = p
            used.add(id(p))
    for fi, (left_eye, right_eye, scale) in enumerate(faces):
        if matched[fi] is None:
            matched[fi] = Person(left_eye, right_eye, now, scale)
            people.append(matched[fi])
        else:
            matched[fi].update(left_eye, right_eye, now, scale)
    return matched


def find_owner(people, bbox, scale, now, claimed=()):
    """The opted-in person a FaceLandmarker-less face box belongs to, if any."""
    c = np.array(_box_center(bbox))
    best, best_r = None, 1.0
    for p in people:
        if not p.opted_in or p.bbox is None or id(p) in claimed:
            continue
        r = float(np.linalg.norm(np.array(_box_center(p.bbox)) - c)) / p.match_limit(scale, now)
        if r < best_r:
            best, best_r = p, r
    return best


# =============================================================================================
# Shared rendering helpers (eye ROI, warps, masks, live light probes)
# =============================================================================================
def eye_roi(shape, cx, cy, rx, ry, lid_pts):
    f_h, f_w = shape[:2]
    rm = max(rx, ry)
    x1 = max(0, int(min(cx - rm, lid_pts[:, 0].min())) - 3)
    y1 = max(0, int(min(cy - rm, lid_pts[:, 1].min())) - 3)
    x2 = min(f_w, int(max(cx + rm, lid_pts[:, 0].max())) + 4)
    y2 = min(f_h, int(max(cy + rm, lid_pts[:, 1].max())) + 4)
    return x1, y1, x2, y2


def texture_affine(cx, cy, rx, ry, angle_deg, x1, y1):
    """2x3 affine mapping the normalized texture circle (127.5, 127.5, r=127.5) onto the
    measured iris ellipse (cx, cy, rx, ry, angle) in ROI coordinates."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    kx, ky = rx / TEX_R, ry / TEX_R
    A = np.array([[kx * ca, -ky * sa], [kx * sa, ky * ca]], np.float64)
    t = np.array([cx - x1, cy - y1]) - A @ np.array([TEX_C, TEX_C])
    return np.hstack([A, t[:, None]]).astype(np.float32)


def ellipse_frame(h, w, x1, y1, cx, cy, angle_deg):
    """Pixel coordinates of the ROI rotated into the iris ellipse's axes."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    yy, xx = np.mgrid[y1:y1 + h, x1:x1 + w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    return dx * ca + dy * sa, -dx * sa + dy * ca


def ellipse_alpha(u, v, rx, ry, feather_px):
    """Anti-aliased elliptical mask; the soft edge is feather_px wide in real pixels."""
    d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2) + 1e-6
    grad = np.sqrt((u / rx ** 2) ** 2 + (v / ry ** 2) ** 2) / d + 1e-6
    edge_px = (1.0 - d) / grad                  # signed distance to the ellipse, + inside
    return np.clip(edge_px / feather_px + 0.5, 0, 1)


def eyelid_alpha(h, w, x1, y1, lid_pts):
    """Sub-pixel rasterized 16-point eyelid polygon, lightly blurred."""
    mask = np.zeros((h, w), np.uint8)
    pts = np.round((lid_pts - np.array([x1, y1], np.float32)) * 4).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(mask, (5, 5), 0.9).astype(np.float32) / 255.0


def upper_lid_shadow(h, w, x1, y1, lid_pts, ry, strength):
    """Shadow multiplier that is darkest right under the upper eyelid and fades downward."""
    canvas = np.full((h, w), 255, np.uint8)
    upper = np.round(lid_pts[UPPER_LID_IDX] - np.array([x1, y1], np.float32)).astype(np.int32)
    cv2.polylines(canvas, [upper], False, 0, 1)
    dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
    return 1.0 - strength * np.exp(-dist / max(1.0, 0.45 * ry))


def scene_light(roi, gray, lid_a, iris_a, u):
    """Live probes from the real eye: exposure, RGB color cast of the light (from the sclera,
    which is ~white) and left/right light direction (-1..1)."""
    eye = lid_a > 0.5
    sclera = eye & (iris_a < 0.05)
    if sclera.sum() >= 12:
        ref = roi[sclera]
    elif eye.sum() >= 12:
        ref = roi[eye]
    else:
        ref = roi.reshape(-1, 3)
    mean_bgr = ref.mean(0) + 1.0
    cast = np.clip(mean_bgr / mean_bgr.mean(), 0.8, 1.2)
    exposure = float(np.clip((gray[eye].mean() if eye.any() else gray.mean()) / 115.0, 0.35, 1.5))
    light_dir = 0.0
    left, right = sclera & (u < 0), sclera & (u > 0)
    if left.sum() >= 6 and right.sum() >= 6:
        ml, mr = gray[left].mean(), gray[right].mean()
        light_dir = float(np.clip((mr - ml) / (mr + ml + 1.0), -0.5, 0.5)) * 2.0
    return exposure, cast, light_dir


def catchlight_alpha(gray, iris_a, lid_a, strongest_only):
    """Real corneal reflections inside the visible iris, as a soft alpha mask."""
    inside = (iris_a * lid_a) > 0.5
    n_in = int(inside.sum())
    if n_in < 10:
        return np.zeros_like(gray)
    vals = gray[inside]
    thr = max(float(np.median(vals)) + 55.0, 130.0)
    m = ((gray > thr) & inside).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= 0.15 * n_in]
    if strongest_only and comps:
        comps = [max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA] * gray[lbl == i].mean())]
    for i in comps:
        keep[lbl == i] = 1
    if not keep.any():
        return np.zeros_like(gray)
    keep = cv2.dilate(keep, np.ones((2, 2), np.uint8))
    return cv2.GaussianBlur(keep.astype(np.float32), (5, 5), 0.8)


def _odd(n):
    n = max(3, int(n))
    return n if n % 2 else n + 1


# =============================================================================================
# Mode 1 renderer
# =============================================================================================
def overlay_iris_natural(frame, texture, pigment, eye, lid_pts) -> None:
    """Procedural pigment iris, re-lit from the live eye, clipped inside the eyelids."""
    cx, cy, rx, ry, ang = (float(v) for v in eye)
    x1, y1, x2, y2 = eye_roi(frame.shape, cx, cy, rx, ry, lid_pts)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    h, w = y2 - y1, x2 - x1
    roi8 = frame[y1:y2, x1:x2]
    roi = roi8.astype(np.float32)
    gray = cv2.cvtColor(roi8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    M = texture_affine(cx, cy, rx, ry, ang, x1, y1)
    warped = cv2.warpAffine(texture, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    u, v = ellipse_frame(h, w, x1, y1, cx, cy, ang)
    iris_a = ellipse_alpha(u, v, rx, ry, max(1.5, 0.12 * min(rx, ry)))
    lid_a = eyelid_alpha(h, w, x1, y1, lid_pts)
    exposure, cast, light_dir = scene_light(roi, gray, lid_a, iris_a, u)

    # Live shading: soft local light from the real iris + directional light + eyelid shadow.
    blur_g = cv2.GaussianBlur(gray, (_odd(0.6 * min(rx, ry)),) * 2, 0)
    vis = iris_a * lid_a
    m_i = float((blur_g * vis).sum() / vis.sum()) if vis.sum() > 1 else float(blur_g.mean())
    local = np.clip(blur_g / max(m_i, 1.0), 0.6, 1.5) ** 0.6
    directional = 1.0 + 0.22 * light_dir * np.clip(u / rx, -1, 1)
    lid_shadow = upper_lid_shadow(h, w, x1, y1, lid_pts, ry, LID_SHADOW_NATURAL)
    shade = local * directional * lid_shadow

    # Pigment response: hue drifts and saturation rises in shadow, highlights desaturate.
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
    dark_amt = np.clip(1.0 - shade, 0, 1)
    bright_amt = np.clip(shade - 1.0, 0, 1)
    hsv[..., 0] = (hsv[..., 0] + pigment["hue_shift"] * dark_amt * 2.5) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.35 * dark_amt - 0.45 * bright_amt), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * shade * exposure, 0, 255)
    col = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) * cast

    alpha = (iris_a * lid_a * NATURAL_OPACITY)[..., None]
    out = col * alpha + roi * (1.0 - alpha)
    g = catchlight_alpha(gray, iris_a, lid_a, strongest_only=False)[..., None]
    out = out * (1 - g) + np.maximum(roi, out) * g        # real catchlights stay on top
    frame[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================================
# Mode 2 renderer
# =============================================================================================
def overlay_iris_png(frame, texture, eye, lid_pts) -> None:
    """Replace the live iris with the extracted PNG iris, warped onto the measured ellipse,
    feathered at the limbus and cut by the eyelids."""
    cx, cy, rx, ry, ang = (float(v) for v in eye)
    x1, y1, x2, y2 = eye_roi(frame.shape, cx, cy, rx, ry, lid_pts)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    h, w = y2 - y1, x2 - x1
    roi8 = frame[y1:y2, x1:x2]
    roi = roi8.astype(np.float32)
    gray = cv2.cvtColor(roi8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    M = texture_affine(cx, cy, rx, ry, ang, x1, y1)
    warped = cv2.warpAffine(texture, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    u, v = ellipse_frame(h, w, x1, y1, cx, cy, ang)
    iris_a = ellipse_alpha(u, v, rx, ry, PNG_FEATHER_PX)
    lid_a = eyelid_alpha(h, w, x1, y1, lid_pts)
    exposure, cast, _ = scene_light(roi, gray, lid_a, iris_a, u)

    # Keep the PNG's vivid color; only a touch of live exposure, cast and lid shadow.
    gain = math.sqrt(min(max(exposure, 0.55), 1.25))
    shadow = upper_lid_shadow(h, w, x1, y1, lid_pts, ry, LID_SHADOW_PNG)
    warped = warped * (shadow * gain)[..., None] * (1.0 + 0.35 * (cast - 1.0))

    alpha = (iris_a * lid_a * PNG_OPACITY)[..., None]
    out = warped * alpha + roi * (1.0 - alpha)
    g = catchlight_alpha(gray, iris_a, lid_a, strongest_only=True)[..., None]
    out = out * (1 - g) + np.maximum(roi, out) * g
    frame[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================================
# Face data, opt-in logic, UI
# =============================================================================================
def blur_face(frame, bbox) -> None:
    x, y, w, h = bbox
    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
    f_h, f_w = frame.shape[:2]
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(f_w, x + w + pad_w), min(f_h, y + h + pad_h)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (51, 51), 30)


def draw_hold_ring(frame, person, progress) -> None:
    cx, cy = person.center()
    r = max(8, int(person.eye_dist() * 0.25))
    center = (int(cx), int(cy - person.eye_dist() * 1.1))
    cv2.circle(frame, center, r, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.ellipse(frame, center, (r, r), -90, 0, int(360 * progress), (255, 255, 255), 3, cv2.LINE_AA)


# =============================================================================================
# Touchscreen UI: < Mode > bar at the top, < Color > bar at the bottom.
# A tap on a touchscreen reaches OpenCV as a left mouse click, so mouse clicks work too.
# =============================================================================================
FLASH_SECONDS = 0.15       # how long a tapped button stays highlighted


def _shade_box(frame, x1, y1, x2, y2, color, alpha):
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = (roi * (1.0 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def _draw_arrow_btn(frame, box, direction, flashing, s):
    """Touch-friendly button with an anti-aliased < or > triangle."""
    x1, y1, x2, y2 = box
    _shade_box(frame, x1, y1, x2, y2, (255, 220, 120) if flashing else (35, 35, 35),
               0.75 if flashing else 0.60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255) if flashing else (190, 190, 190),
                  max(1, int(2 * s)), cv2.LINE_AA)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sz = min(x2 - x1, y2 - y1) // 4
    if direction == "left":
        pts = np.array([[cx + sz // 2, cy - sz], [cx - sz, cy], [cx + sz // 2, cy + sz]], np.int32)
    else:
        pts = np.array([[cx - sz // 2, cy - sz], [cx + sz, cy], [cx - sz // 2, cy + sz]], np.int32)
    cv2.fillPoly(frame, [pts], (20, 20, 20) if flashing else (255, 255, 255), cv2.LINE_AA)


def _bar_boxes(x1, y1, width, height, btn_w, gap):
    x2, y2 = x1 + width, y1 + height
    return ((x1, y1, x1 + btn_w, y2),                     # < button
            (x1 + btn_w + gap, y1, x2 - btn_w - gap, y2),  # label in the middle
            (x2 - btn_w, y1, x2, y2))                      # > button


def draw_touch_ui(frame, ui, color_names, irises, active, now) -> None:
    """Draw the top Mode < Bright/Dark > bar and the bottom Color < swatch + name > bar,
    and record the button rectangles so taps can be matched to them."""
    h, w = frame.shape[:2]
    s = w / 640.0                                   # everything scales with the frame width
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale, text_thick = 0.58 * s, max(1, int(round(2 * s)))
    flash = ui["flash_btn"] if now - ui["flash_time"] < FLASH_SECONDS else None
    bw, bh, gap, margin = int(52 * s), int(44 * s), int(4 * s), int(10 * s)

    def label_box(box):
        _shade_box(frame, *box, (0, 0, 0), 0.60)
        cv2.rectangle(frame, box[:2], box[2:], (160, 160, 160), 1, cv2.LINE_AA)

    # --- Top bar: mode ------------------------------------------------------------------
    top_w = int(340 * s)
    m_prev, m_mid, m_next = _bar_boxes((w - top_w) // 2, margin, top_w, bh, bw, gap)
    label_box(m_mid)
    _draw_arrow_btn(frame, m_prev, "left", flash == "mode_prev", s)
    _draw_arrow_btn(frame, m_next, "right", flash == "mode_next", s)
    text = "Mode: " + ("Bright" if ui["mode"] == 1 else "Dark")
    (tw, th), _ = cv2.getTextSize(text, font, text_scale, text_thick)
    cv2.putText(frame, text, ((m_mid[0] + m_mid[2] - tw) // 2, (m_mid[1] + m_mid[3] + th) // 2),
                font, text_scale, (255, 255, 255), text_thick, cv2.LINE_AA)

    # --- Bottom bar: color with a live swatch of the iris that will be used ---------------
    bot_w = int(320 * s)
    c_prev, c_mid, c_next = _bar_boxes((w - bot_w) // 2, h - bh - margin - 2, bot_w, bh, bw, gap)
    label_box(c_mid)
    _draw_arrow_btn(frame, c_prev, "left", flash == "color_prev", s)
    _draw_arrow_btn(frame, c_next, "right", flash == "color_next", s)
    color = color_names[ui["color_idx"]]
    swatch = irises["png" if ui["mode"] == 2 else "natural"][color]
    r = int(14 * s)
    sx, sy = c_mid[0] + int(24 * s), (c_mid[1] + c_mid[3]) // 2
    patch = frame[sy - r:sy + r, sx - r:sx + r]
    if patch.shape[:2] == (2 * r, 2 * r):
        small = cv2.resize(swatch, (2 * r, 2 * r), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:2 * r, 0:2 * r]
        disk = (xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2 <= r ** 2
        patch[disk] = small[disk]
        cv2.circle(frame, (sx, sy), r, (230, 230, 230), 1, cv2.LINE_AA)
    (_, th), _ = cv2.getTextSize(color, font, text_scale, text_thick)
    cv2.putText(frame, color, (sx + r + int(12 * s), (c_mid[1] + c_mid[3] + th) // 2),
                font, text_scale, (255, 255, 255), text_thick, cv2.LINE_AA)

    if not active:
        hint = f"Stand still {HOLD_SECONDS:.0f}s to opt in"
        hs, ht = 0.45 * s, max(1, int(round(s)))
        (hw, _), _ = cv2.getTextSize(hint, font, hs, ht)
        cv2.putText(frame, hint, ((w - hw) // 2, c_prev[1] - int(8 * s)), font, hs,
                    (230, 230, 230), ht, cv2.LINE_AA)

    ui["buttons"] = {"mode_prev": m_prev, "mode_mid": m_mid, "mode_next": m_next,
                     "color_prev": c_prev, "color_mid": c_mid, "color_next": c_next}


def ui_set_mode(ui, mode) -> None:
    if mode != ui["mode"]:
        ui["mode"] = mode
        print(MODE_LABELS[mode])


def ui_step_color(ui, people, color_names, step) -> None:
    """Move the selected color and apply it to everyone currently filtered."""
    ui["color_idx"] = (ui["color_idx"] + step) % len(color_names)
    color = color_names[ui["color_idx"]]
    for p in people:
        if p.color is not None:
            p.color = color
    print("Color ->", color)


def handle_tap(ui, people, color_names, x, y) -> None:
    for name, (x1, y1, x2, y2) in ui["buttons"].items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            ui["flash_btn"], ui["flash_time"] = name, time.monotonic()
            if name.startswith("mode"):                 # only two modes: any mode tap toggles
                ui_set_mode(ui, 2 if ui["mode"] == 1 else 1)
            elif name == "color_prev":
                ui_step_color(ui, people, color_names, -1)
            else:                                       # color_next or the color label
                ui_step_color(ui, people, color_names, +1)
            return


def measure_iris(pts, center_idx, rim_idx):
    """Iris ellipse from the 5 iris landmarks: [cx, cy, rx, ry, angle_deg].
    rx from the left/right rim points, ry from top/bottom (captures foreshortening)."""
    c = pts[center_idx]
    rim = pts[rim_idx]                            # [right, top, left, bottom]
    horiz = rim[0] - rim[2]
    if horiz[0] < 0:
        horiz = -horiz
    vert = rim[3] - rim[1]
    rx = float(np.linalg.norm(horiz)) / 2
    ry = float(np.linalg.norm(vert)) / 2
    if rx > 1e-3:
        ry = rx * float(np.clip(ry / rx, 0.75, 1.35))
    angle = math.degrees(math.atan2(horiz[1], horiz[0]))
    center = 0.5 * c + 0.25 * (rim[0] + rim[2]) / 2 + 0.25 * (rim[1] + rim[3]) / 2
    return [float(center[0]), float(center[1]), rx, ry, angle]


def extract_face_data(landmarks_list, w, h):
    """Convert MediaPipe landmarks into pixel-space irises, eyelids and face boxes."""
    return [face_from_points(np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32))
            for lm in landmarks_list]


def landmark_far_faces(landmarker, frame, yunet_faces, face_data, people=(), now=0.0):
    """FaceLandmarker misses small faces (~1.3-1.5 m).  For each YuNet face it missed that is
    inside the zone -- or that belongs to someone already opted in, wherever they are --
    re-run FaceLandmarker on an upscaled crop and map the points back."""
    f_h, f_w = frame.shape[:2]
    found = []
    for bbox, scale in yunet_faces:
        x, y, bw, bh = bbox
        cx, cy = x + bw / 2, y + bh / 2
        if any(_inside(cx, cy, d["bbox"]) for d in face_data + found):
            continue
        in_zone = MIN_DIST_M <= distance_m(scale, f_w) <= MAX_DIST_M * (1 + OPTED_IN_MARGIN)
        if not in_zone and find_owner(people, bbox, scale, now) is None:
            continue
        side = max(bw, bh) * 2.0
        x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
        x2, y2 = min(f_w, int(cx + side / 2)), min(f_h, int(cy + side / 2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        s = max(1.0, 320.0 / (x2 - x1))
        crop = cv2.resize(frame[y1:y2, x1:x2], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        ch, cw = crop.shape[:2]
        for lm in res.face_landmarks[:1]:
            pts = np.array([[p.x * cw / s + x1, p.y * ch / s + y1] for p in lm], dtype=np.float32)
            found.append(face_from_points(pts))
    return found


def face_from_points(pts):
    """Pixel-space irises, eyelids and face box from the 478 landmark points."""
    x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
    x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())
    eye_px = float(np.linalg.norm(pts[RIGHT_IRIS_CENTER] - pts[LEFT_IRIS_CENTER]))
    face_h = float(np.linalg.norm(pts[10] - pts[152]))            # forehead -> chin

    # Head turn: how far the nose tip sits from the middle of the face (0 = facing camera).
    mid_x = (pts[234, 0] + pts[454, 0]) / 2
    yaw = (pts[1, 0] - mid_x) / max(abs(pts[454, 0] - pts[234, 0]) / 2, 1.0)

    # Eye turn: how far each iris sits from the middle of its eye opening.
    def gaze(c, a, b):
        return (pts[c, 0] - (pts[a, 0] + pts[b, 0]) / 2) / max(abs(pts[b, 0] - pts[a, 0]) / 2, 1.0)
    gaze_x = (gaze(LEFT_IRIS_CENTER, 33, 133) + gaze(RIGHT_IRIS_CENTER, 362, 263)) / 2
    return {
        "scale": max(eye_px, FACE_HEIGHT_RATIO * face_h),
        "facing": abs(yaw) < FACING_MAX_YAW and abs(gaze_x) < FACING_MAX_GAZE,
        "yaw": float(yaw),
        "gaze": float(gaze_x),
        "left_lid_w": float(np.linalg.norm(pts[33] - pts[133])),
        "right_lid_w": float(np.linalg.norm(pts[362] - pts[263])),
        "left_eye": pts[LEFT_IRIS_CENTER].copy(),
        "right_eye": pts[RIGHT_IRIS_CENTER].copy(),
        "left_iris": measure_iris(pts, LEFT_IRIS_CENTER, LEFT_IRIS_RIM),
        "right_iris": measure_iris(pts, RIGHT_IRIS_CENTER, RIGHT_IRIS_RIM),
        "left_lid": pts[LEFT_EYE_OUTLINE].copy(),
        "right_lid": pts[RIGHT_EYE_OUTLINE].copy(),
        "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
    }


def create_face_detector():
    """YuNet face detector for small/far faces (FaceLandmarker misses faces ~1.5 m away)."""
    if not os.path.exists(YUNET_PATH):
        print("face_detection_yunet_2023mar.onnx not found -- far faces may not be blurred.")
        return None
    try:
        return cv2.FaceDetectorYN.create(YUNET_PATH, "", (YUNET_WIDTH, YUNET_WIDTH), 0.6, 0.3, 5000)
    except Exception as exc:
        print(f"YuNet face detector unavailable ({exc}) -- far faces may not be blurred.")
        return None


def detect_faces_yunet(detector, frame):
    """[(bbox_xywh, scale_px)] in full-frame pixels; scale_px is comparable to the iris
    spacing of a face looking at the camera, even when the head is turned."""
    if detector is None:
        return []
    h, w = frame.shape[:2]
    s = YUNET_WIDTH / float(w)
    small = cv2.resize(frame, (YUNET_WIDTH, max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    out = []
    for f in (faces if faces is not None else []):
        x, y, bw, bh = (f[:4] / s).astype(int)
        eye_px = float(np.linalg.norm(f[4:6] - f[6:8])) / s * YUNET_EYE_SCALE
        scale = max(eye_px, YUNET_BOX_RATIO * bh)
        out.append(((int(x), int(y), int(bw), int(bh)), scale))
    return out


def draw_too_close(frame) -> None:
    """Big centered warning banner."""
    text = "Too close! Step back."
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2 * frame.shape[1] / 1280 + 0.4, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (frame.shape[1] - tw) // 2
    y = int(frame.shape[0] * 0.30)                 # below the touch UI's top bar
    x1, y1, x2, y2 = max(0, x - 24), max(0, y - th - 20), min(frame.shape[1], x + tw + 24), y + 20
    frame[y1:y2, x1:x2] = (frame[y1:y2, x1:x2] * 0.35 + np.array([0, 0, 160]) * 0.65).astype(np.uint8)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _inside(px, py, bbox, pad=0.2):
    x, y, w, h = bbox
    return x - pad * w <= px <= x + w * (1 + pad) and y - pad * h <= py <= y + h * (1 + pad)


def apply_filter(frame, face_data, people, irises, selected_color, mode, now, extra_faces=()):
    """Run the distance zone + opt-in logic and draw irises.  New opt-ins get selected_color.

    Who gets blurred (only people who have NOT opted in):
      * closer than MIN_DIST_M (plus "Too close!") or farther than MAX_DIST_M
      * outside the zone sideways, or not looking at the camera (a passerby)
    Who does NOT get blurred:
      * someone in the zone looking at the camera (they see the 2 s opt-in ring)
      * someone who HAS opted in -- wherever they move, turn or look, until they leave the
        webcam's view (see Person.gone).  They never have to opt in again meanwhile.
    extra_faces: YuNet detections (catches faces FaceLandmarker misses, e.g. far or in profile).
    Returns (people currently being filtered, whether someone is too close)."""
    f_w = frame.shape[1]
    too_close = False
    forget_gone_people(people, now, frame.shape)

    # 1. Faces only YuNet sees.  If one belongs to an opted-in person (head turned away,
    #    motion blur, too far for FaceLandmarker) they stay opted in and unblurred.
    claimed = set()
    for bbox, scale in extra_faces:
        cx, cy = _box_center(bbox)
        if any(_inside(cx, cy, d["bbox"]) for d in face_data):
            continue                                   # FaceLandmarker handles this face
        dist = distance_m(scale, f_w)
        owner = find_owner(people, bbox, scale, now, claimed)
        if owner:
            owner.follow_box(bbox, scale, now)
            claimed.add(id(owner))
            too_close |= dist < MIN_DIST_M             # warn, but never blur an opted-in face
            continue
        blur_face(frame, bbox)                         # not opted in -> passerby
        too_close |= dist < MIN_DIST_M

    if not face_data:
        return [], too_close

    faces = [(d["left_eye"], d["right_eye"], d["scale"]) for d in face_data]
    matched = match_people(people, faces, now)
    active = []

    for data, person in zip(face_data, matched):
        person.update_iris("left", data["left_iris"])
        person.update_iris("right", data["right_iris"])
        person.bbox = data["bbox"]
        scale = person.scale
        if scale < 2:
            if not person.opted_in:
                blur_face(frame, data["bbox"])
            continue

        dist = distance_m(scale, f_w)
        offset_m = abs(person.center()[0] - f_w / 2) * REAL_EYE_DIST_M / scale

        if dist < MIN_DIST_M:
            too_close = True
        if not person.opted_in and (dist < MIN_DIST_M or dist > MAX_DIST_M
                                    or offset_m > MAX_HORIZ_OFFSET_M):
            person.in_zone_since = None
            blur_face(frame, data["bbox"])
            continue

        if not person.opted_in:
            # Opting in requires actually looking at the camera for HOLD_SECONDS.
            if data["facing"]:
                person.last_facing = now
            looking = person.last_facing is not None and now - person.last_facing <= FACING_GRACE_S
            if not looking:
                person.in_zone_since = None
                blur_face(frame, data["bbox"])         # passerby glancing / walking past
                continue
            if person.in_zone_since is None:
                person.in_zone_since = now
            held = now - person.in_zone_since
            if held < HOLD_SECONDS:
                draw_hold_ring(frame, person, held / HOLD_SECONDS)
                continue
            person.color = selected_color
        active.append(person)

        fallback_r = scale * IRIS_RADIUS_RATIO
        widest = max(data["left_lid_w"], data["right_lid_w"], 1.0)
        for side in ("left", "right"):
            if data[f"{side}_lid_w"] < HIDDEN_EYE_RATIO * widest:
                continue                               # eye hidden behind the nose (profile)
            eye = np.array(person.iris[side], np.float32)
            if eye[2] < 2.0 or eye[3] < 2.0:
                eye[2] = eye[3] = fallback_r
            lid = data[f"{side}_lid"]
            if mode == 2:
                overlay_iris_png(frame, irises["png"][person.color], eye, lid)
            else:
                overlay_iris_natural(frame, irises["natural"][person.color],
                                     PIGMENTS[person.color], eye, lid)
    return active, too_close


def main():
    load_model()
    with open(TASK_PATH, "rb") as f:
        model_buffer = f.read()

    # --- Iris assets for both modes ---------------------------------------------------------
    print("Extracting iris textures from the PNG photos...")
    png_textures = load_png_irises(model_buffer)
    for name, tex in png_textures.items():
        if name not in PIGMENTS:                    # PNG-only color -> derive a pigment
            PIGMENTS[name] = pigment_from_texture(tex)
    color_names = list(PIGMENTS.keys())
    natural_textures = build_natural_textures(color_names)
    for name in color_names:
        if name not in png_textures:
            print(f"  No PNG for '{name}' -- Mode 2 uses the procedural {name} iris instead.")
            png_textures[name] = natural_textures[name]
    irises = {"natural": natural_textures, "png": png_textures}

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_buffer),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    face_detector = create_face_detector()

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        landmarker.close()
        return
    # 1280x720 = the resolution the distance calibration was measured at.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (cam_w, cam_h) != (1280, 720):
        print(f"Warning: webcam gave {cam_w}x{cam_h}, not 1280x720 -- distances may be off.")

    people = []
    ui = {"mode": 1, "color_idx": 0, "buttons": {}, "flash_btn": None, "flash_time": 0.0}

    def on_touch(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:          # finger tap or mouse click
            handle_tap(ui, people, color_names, x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_touch)
    print("Running.  Tap the on-screen < > arrows, or use the keyboard:")
    print("  1 = Bright mode, 2 = Dark mode, C = next color, Q/Esc = quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        face_data = extract_face_data(result.face_landmarks, w, h) if result.face_landmarks else []
        extra_faces = detect_faces_yunet(face_detector, frame)
        now = time.monotonic()
        face_data += landmark_far_faces(landmarker, frame, extra_faces, face_data, people, now)
        active, too_close = apply_filter(frame, face_data, people, irises,
                                         color_names[ui["color_idx"]], ui["mode"], now,
                                         extra_faces)
        if too_close:
            draw_too_close(frame)
        draw_touch_ui(frame, ui, color_names, irises, active, now)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if key == ord('1'):
            ui_set_mode(ui, 1)
        elif key == ord('2'):
            ui_set_mode(ui, 2)
        elif key in (ord('c'), ord('C')):
            ui_step_color(ui, people, color_names, +1)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
