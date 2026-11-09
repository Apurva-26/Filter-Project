# Gun Gun Shoot — Design Doc

A camera-based "gun gun shoot" game that runs on a Raspberry Pi 5 with a touch screen, set up in a school hallway. A student walks up, starts a game, and plays against the computer using body and hand poses.

_This doc records every decision made so far. Update it whenever a new decision is made._

---

## 1. Hardware

- **Raspberry Pi 5**
- **Touch screen**
- **Regular camera** mounted so it sees most of a person's upper torso (head to waist), including raised arms. At least 1280×720 is preferred, because hands are small at hallway distance.
- **Play zone: 1.0–1.5 m from the camera.** About 1.24–1.26 m works best for everyone. Closer than ~1 m doesn't work well (arms and hands get cut off; 0.33 m was tested and was too close). Mark the zone on the floor with tape.
- **Speaker** for the "gun… gun… SHOOT!" rhythm.
- Runs **fully locally**: no video is saved or sent anywhere.

## 2. Game flow

```
[Attract screen] → tap "Play" → [Position check: 1.0–1.5 m] → [Round loop] → [Game over] → back to attract
```

**Round loop:**
1. The game sets the tempo with sound (and maybe visuals): "gun… gun… SHOOT!"
   - Use separate recorded clips (slap, "gun", "SHOOT") played on a timer, so the code knows exactly when "SHOOT" happens and the tempo can be changed.
   - The player slaps their thighs during "gun gun". This is just for fun; it is **not** detected.
2. The AI **commits to its move before** the player's move is read, so it can't cheat.
3. After "SHOOT", a **capture window** opens and the player's pose is classified (see §5).
4. Both moves are revealed on screen, and bullets and the defensive streak are updated.
5. If someone wins, the game ends. Otherwise the next round starts.

- **Starting a game: tap the screen** (no gesture start). The touchscreen acts like a mouse, so a tap is a click.
- **Kiosk mode:** on power-up the Pi auto-logs in and launches the game fullscreen (no desktop or cursor), and restarts it if it crashes.
- **No per-player calibration.** Normalization plus training data from many people should handle differences between players. (The distance check needs a one-time calibration **per camera**, not per player; see §7.1.)
- **The player leaves:** if no person is detected for about 5 s (exact value TBD), the game aborts and returns to the attract screen.
- **Multiple people in frame:** track the largest or most centered person and ignore everyone else.

## 3. Rules

### Moves
| Move | What it looks like |
|---|---|
| **Reload** | Both hands in the gun shape, pointing up, arms slightly raised (hands above shoulders). |
| **Shoot** | A finger gun pointed at the screen/camera. One hand **or** two hands; **two hands counts the same as one** (1 bullet). |
| **Shield** | Forearms **always crossed in an X**, either on the torso or held slightly in front of it. Hands are often fists. |
| **Mirror** | The two hands form a diamond/triangle: index tips touching and thumb tips touching. Either flat hands or index+thumb with the other fingers curled. |

### Bullets
- Start with 0 bullets (TBD: confirm the starting count).
- **No bullet cap.**
- Reload: +1 bullet.
- Shoot: −1 bullet.
- **Mirror costs 1 bullet every time it's used**, even if the opponent didn't shoot.

### Outcomes (row = you, column = opponent)
|              | Reload          | Shoot                       | Shield          | Mirror              |
|--------------|-----------------|-----------------------------|-----------------|---------------------|
| **Reload**   | both +1         | **you lose**                | you +1          | you +1, they −1     |
| **Shoot**    | **you win**     | draw: both −1, keep playing | blocked, you −1 | **you lose** (reflected), both −1 |
| **Shield**   | they +1         | blocked, they −1            | nothing         | they −1             |
| **Mirror**   | you −1, they +1 | **you win** (reflected)     | you −1          | both −1             |

### Rule breaks (breaking a rule = the other side wins immediately)
- Shooting with 0 bullets.
- Mirroring with 0 bullets.
- **More than 5 defensive moves (shield/mirror) in a row.** A 6th is a rule break. Reload or shoot resets the count.

### Other rules
- **No round cap.** The game continues until someone loses.
- Simultaneous shoot = **draw for that round** (not "who yelled first").
- Changing a pose after it has locked is **ignored** (not a rule break), because detection flicker shouldn't make people lose.
- **Both sides start with 0 bullets** (`rules.STARTING_BULLETS`).
- **No pose detected during the capture window → the round is replayed** ("Didn't catch that!"), with no penalty. **3 misses in a row → the game ends** (the player is assumed to have stopped playing).
- All rules live in `rules.py`; the game and the AI both use it.

## 4. Pose detection pipeline

```
camera frame
  → MediaPipe Pose Landmarker (full frame)
  → crop a box around each wrist → MediaPipe Hand Landmarker on each crop
  → map hand points from crop coordinates back to full-frame coordinates
  → normalize (see §4.2)
  → MLP classifier → per-frame prediction + confidence
  → stability filter during the capture window → locked move
```

### 4.1 Landmarks used
- **Pose (33 points available):** use the nose (0), shoulders (11, 12), elbows (13, 14), wrists (15, 16) and hips (23, 24, for presence/framing). Ignore the face points (1–10) and the legs (25–32). The pose model's rough hand points (17–22) are optional, as a fallback when the hand model loses a hand (e.g. hands touching during mirror).
- **Hands (21 points each):** use all of them. These carry the finger shape.
- Left/right hand identity comes from **which wrist the crop came from**, not from MediaPipe's handedness label.
- If a hand isn't found, fill its values with zeros and set a `hand_present` flag feature.

### 4.2 Normalization
Applied after landmarks are detected and before the MLP, in exactly the same way during data collection and during live play.
- **Body frame:** subtract the shoulder midpoint from all points, then divide by shoulder width. This removes where the player stands, their height, and their distance from the camera.
- **Hand-local frame:** subtract each hand's wrist from that hand's points, then divide by hand size (wrist → middle-finger knuckle). This describes finger shape independent of where the hand is.
- Feed the MLP both: body-frame wrist/arm positions (where the hands are) and hand-local finger points (what shape the hands make).

### 4.3 Classifier
- **Small MLP** (e.g. scikit-learn `MLPClassifier`), not hand-written rules.
- **Classes** (variants are separate classes, then merged into game moves):

| Class | Game move |
|---|---|
| `reload` | Reload |
| `shoot_one` | Shoot |
| `shoot_two` | Shoot |
| `shield` | Shield |
| `mirror_flat` | Mirror |
| `mirror_fingers` | Mirror |
| `none` | (ignored): neutral standing, thigh slapping, in-between motions, random |

- The `none` class is needed because softmax confidence only says which *known* class an input is closest to, not whether it is any of them. A neutral stance can still get 95% "shoot". The confidence threshold is kept as an extra check.
- The shoot pose (finger pointed at the camera) looks short in 2D. The thumb sticking up and the hands being uncrossed at chest height are the main clues. Shield is always crossed, which keeps it distinct from shoot.

### 4.4 Training data
- Collect from **many different people** (target 20–30), each doing every class several times.
- The CSV stores **raw landmarks only**, in **pixels** (not MediaPipe's 0–1 values, which would skew distances on a non-square frame). `train.py` normalizes on load, so normalization can change without re-collecting.
- **Guided session** (`collect.py`): the operator enters a person ID (no names). The screen shows "Next: <POSE>", a 3 s countdown, 3 s of recording (every frame saved), then 1.5 s of "Relax!". Each pose is done **3 times**, in a **fixed order** (the same for everyone). Then 4 `none` prompts run once each: stand normally, slap thighs, move slowly between poses, move hands around. About 2–3 minutes per person. Keys: SPACE start/pause, R redo last take, Q quit. The CSV is saved after every take. A take with too few detected frames is redone automatically.
- Evaluate on **people not in the training set** (split by person, not by frame).
- **Mirror augmentation** (on by default in `train.py`): every training frame gets a left↔right flipped copy. Labels in `FLIP_LABELS` (e.g. `shoot_left` ↔ `shoot_right`) are swapped. Only the training set is augmented; the test set stays real.
- `train.py` also runs a **none check**: it trains a model without `none` and reports how many `none` frames would have passed the confidence threshold. This shows whether the `none` class is pulling its weight.
- If fewer than 3 people have been recorded, the split falls back to by-take (with a warning that results will look too good).

## 5. Locking in the player's move
- Classify every frame during the capture window (as fast as the Pi allows; ~15 FPS expected).
- A frame counts as a move only if the top game move is at least 75% confident and isn't `none`. Otherwise it's an "unsure" frame.
- Confidence is measured **per game move**: the class probabilities of variants are added together (mirror_flat 50% + mirror_fingers 45% = mirror 95%).
- **Hold meter** (replaced the old "continuous hold" rule, which reset on 2 bad frames in a row and felt inaccurate). Every move has a meter measured in seconds of real time, so it works the same at any FPS:
  - a frame of move X: X's meter + frame time; every other meter − 2 × frame time (`SWITCH_DRAIN`)
  - an unsure / `none` frame: every meter − 0.25 × frame time (`UNSURE_DRAIN`)
  - a move **locks** when its meter reaches 0.4 s (`HOLD_S`). One or two bad frames just delay the lock a little; passing through another pose on the way is wiped out quickly.
- The first move to lock wins; later changes are ignored.
- **Beat timeline** (all in `classifier.py`): GUN at 0 s, GUN at 0.4 s (`GUN_GAP_S`), SHOOT! at 1.0 s (`SHOOT_GAP_S` = 0.6 s after the second GUN). The gun-gun part is faster and the gap before SHOOT is longer, so players have time to get into position.
- **Grace period:** camera frames from the first 0.3 s after SHOOT! are ignored (`GRACE_S`), because hands are still moving into place. The capture window is 1.6 s after SHOOT! in total (`CAPTURE_WINDOW_S`), so there's 1.3 s to lock.
- Frames where the player is outside the distance zone (with a 5 cm margin) count as unsure.

## 6. AI opponent
**Why it changed:** the first version only looked one round ahead and valued a bullet at just 0.1. So it shielded all the time (shield is "free" and safe), even when the player had no bullets, and it shot ~98% of the time right after reloading. It felt predictable and robotic.

Now it works in four steps:
1. **Base strategy** (`strategy.py`): the game-theory "perfect player" mix (minimax / Nash equilibrium) for every situation `(player bullets, AI bullets, player defense streak, AI defense streak)`. It is worked out from `rules.py` by value iteration in ~2 s at startup. It can't be beaten in the long run, and it gets common sense from the rules alone:
   - player has 0 bullets → never shields; reloads, sometimes shoots
   - right after the AI reloads vs a 0-bullet player → reload 72% / shoot 28% (not always shoot)
   - player is at 5 defenses in a row → attacks
   - Situations score +1 (sure AI win) to −1; future rounds are discounted by 0.95.
   - Without the 5-defenses-in-a-row rule, perfect play would be "shield forever" (value 0). The streak limit is what forces action.
2. **Pattern exploiter:** count tables of the player's next move at three levels, added together with weights:
   - **exact** `(player bullets, AI bullets, player's last move)`, weight 1.0
   - **last move only**, weight 0.6
   - **overall habits**, weight 0.3. The general levels catch obvious patterns ("always reloads") within 2–3 rounds.
   - Bullets are bucketed as 0, 1, 2, 3+.
   - **Global table:** saved to `data/opponent_memory.json`, never reset. It learns how students in general play.
   - **Session table:** reset every game (one game = one person). Each session round counts 3× a global round.
   - Every legal player move gets a prior count of 1. Illegal player moves get probability 0.
3. **Exploit plan:** each legal AI move is scored against the prediction. Each (AI move, player move) pair is worth +1 / −1 if the game ends, otherwise the value of the next situation from `strategy.py`, so it thinks ahead. Softmax over the scores with `TEMPERATURE` 0.1.
4. **Blend:** chances = (1 − trust) × base + trust × exploit. Trust = `MAX_EXPLOIT` × evidence / (evidence + 8), where evidence = weighted rounds seen in this exact situation or after this last move. With no data it plays the base strategy. `MAX_EXPLOIT` (0.6 = medium) is the difficulty knob. Then it picks randomly using those chances.
- Legal moves only: never shoots or mirrors with 0 bullets, never exceeds 5 defensive moves in a row.
- The AI commits its move at the **start** of the beat, before the capture window opens.
- Simulated results (fresh memory, 200 games each): AI wins ~70% vs a random player, 100% vs "always reload", ~85% vs reload-then-shoot, ~87% vs shield-heavy, ~64% vs a rough "normal human" bot. It shields ~15–25% of the time (it used to be most of the time).

## 7. Screen / UI
- Built with **pygame** (fullscreen, touch, sound). Vision runs in **two background threads**: a camera thread keeps only the newest frame and prepares it for the screen, and a detection thread runs MediaPipe + the classifier on the newest frame. The video stays live at camera speed even when detection is slow (added because the Pi view lagged behind the player); only the skeleton and move detection trail by about one detection frame. The camera is asked for MJPG with a 1-frame buffer so old frames don't queue up.
- **No game name on screen.** The name "Gun Gun Shoot" is internal only (code, docs). The attract screen shows just "Can you beat the computer?" plus the pose guide, and the window title is blank.
- Screens: attract (tagline + pose guide) → position ("step back", must be framed correctly for 1 s) → ready → beat → SHOOT window → reveal → … → game over (tap to play again, or back to the title screen after 15 s).
- Framing check: the player must be **1.0–1.5 m** away (§7.1), and the nose must be below the top 18% of the screen (leaving room for reload). Messages: "Too close! Step BACK", "Too far! Come CLOSER", "Look at the screen!" (eyes not visible), "Crouch a little" (head too high). The position screen also shows the estimated distance.
- During a game, being out of the zone shows "Too close! Step back" / "Too far! Come closer" at the bottom, and those frames don't count toward the lock.
- Live feedback line at the bottom ("I see: RELOAD 91%"), with a skeleton overlay on the mirrored camera image.
- During the capture window, a hold-progress bar.
- Scoreboard: both sides' bullets and defensive streak, plus the round number.
- Reveal: two colored cards (YOU vs CPU) with the moves, plus a one-line description of what happened. AI move is shown as text for now; symbols/animation TBD.
- **Input:** SPACE starts a game. Touch is switched on with `ENABLE_TOUCH = True` at the top of `game.py`, or `--touch`. ESC quits.
- **Sounds:** beeps for now. GUN = 330 Hz (×2), SHOOT = 660 Hz (one octave higher), set in `Sounds.BEEPS` in `game.py`. Optional `.wav` files in `sounds/` (gun, shoot, reveal, hit, win, lose) replace the beeps.

### 7.1 Distance check (`distance.py`)
- Distance is estimated from the **pixel distance between the eyes** (pose landmarks 2 and 5). Eye spacing is nearly the same for everyone (~6.3 cm), unlike shoulder width. `distance = EYE_PX_AT_1M / eye_px`, using the median of the last 7 frames.
- **Calibrate once per camera:** `py distance.py`, stand 1.24 m away (lens to eyes) and press SPACE. This saves `data/distance_calibration.json`. Redo it only after changing the camera or its zoom. Resolution changes are scaled automatically. Without a calibration it guesses from a typical 70° webcam.
- Zone settings are `MIN_DISTANCE_M` = 1.0, `MAX_DISTANCE_M` = 1.5, and `ZONE_MARGIN_M` = 0.05 (extra leeway mid-game).
- Accuracy is about ±10%. Turning the head reads as farther away.
- The eye points are kept in `FrameLandmarks.pose` but are **not** in the CSV or the MLP features, so no retraining was needed.

## 8. Scripts / project structure
- `features.py`: **shared** code: run the landmarkers, crop hands, map points back, normalize, and build the feature vector. Every other script imports this, so collection and live play are guaranteed to process data identically.
- `collect.py`: live camera view. Press a key to pick a class, and record labeled rows (raw + normalized landmarks, person ID) to CSV.
- `collect_none.py`: the same guided session, but records only the `none` class, using 12 everyday non-move prompts × 2 takes (waving, clapping, hands on hips, pretending to look at a phone, etc.). It saves to `<person_id>_none_<date-time>.csv`. Anyone can do it without knowing the game.
- `collect_shoot.py`: the same guided session, but records only extra shoot data: 5 takes of 3.5 s (right hand once, left hand once, two hands three times; about 17 s in total), because the live game struggles to detect shoots. It saves to `<person_id>_shoot_<date-time>.csv`. Use the person ID that person recorded under before, so the train/test split keeps them on one side.
- `train.py`: load the CSVs, split by person, train the MLP, print per-class accuracy and a confusion matrix, and save the model.
- `classifier.py`: `PoseClassifier`, `confident_move()`, `MoveLocker` (hold meter) and **all the beat/lock timing settings**. Shared by `live_test.py` and `game.py`.
- `distance.py`: eye-based distance estimate and zone check. `py distance.py` = calibration screen.
- `strategy.py`: the AI's game-theory base strategy. `py strategy.py` prints example situations.
- `live_test.py`: live camera view showing the predicted move, confidence and lock-in logic, with no game. Used to check the classifier.
- `rules.py`: the game rules (`GameState`, `legal_moves()`, `resolve()`).
- `opponent.py`: the AI. `py opponent.py` plays against it in the terminal with no camera.
- `game.py`: the full game. `py game.py` on the laptop; `py game.py --fullscreen --touch` on the Pi.
- Develop and collect on a laptop, then deploy to the Pi 5. Record training data with the camera at roughly the same height and distance as the hallway setup.
- **GitHub:** this is a group project. Everyone's filter goes in the shared repo `Apurva-26/Filter-Project`, each in its own folder. This one is in `gun_gun_shoot/`. Run every script from inside that folder (`cd gun_gun_shoot`, then `py game.py`); the files import each other by name (`import features`), not as a package. The shared launcher is described in §8.1. Not committed (`.gitignore`): `__pycache__/`, `.history/`, `data/opponent_memory.json` and `data/distance_calibration.json` (both are per machine).

### 8.1 Shared launcher (repo root)
- `launcher.py`: a home screen with one machine per filter (see §8.2): **Clown Nose** (`ai_updated.py`), **Big Eyes** (`glasses_filter_updated.py`), **Iris Colors** (`saumyaa.py`) and **Mystery** (this game; it has no name on screen). Headline "Tap a filter to try it!" and footnote "The camera only runs while you use a filter. Nothing is recorded or saved." Tiles have simple drawn icons, so no image files are needed.
- Tapping a tile closes the menu window and runs the filter as its **own program** (in its own folder). When the filter exits, the menu comes back. If it crashes, the menu shows "Sorry, X isn't working right now." The menu never opens the camera.
- The tile list lives in `filters.json` (name, script, icon, color, args, kiosk_args).
- `py launcher.py --fullscreen` (on the Pi) sets `FILTER_KIOSK=1` for the filters: fullscreen, plus **auto-back after 20 s** with nobody in front of the camera.
- `kiosk.py`: a shared helper for the OpenCV filters. It adds a **Back button (top-right)** that acts like pressing `q`, **on-screen buttons that press the filter's existing keys** (Clown Nose: On/Off = `n`, Switch = `f`; Glasses: `1`–`4`), and the auto-back. Big Eyes (`glasses_filter_updated.py`) now opens straight away with only the Back button: its author removed the "Press Y" question and the extra modes. The filters' behavior is otherwise unchanged.
- This game draws its own Back button (top-right; the CPU scoreboard moved left to make room) and auto-backs after 20 s in kiosk mode. The launcher starts it with `--touch` (and `--fullscreen` in kiosk mode).
- Model files are committed next to the scripts (`face_detection_yunet_2023mar.onnx`, `face_landmarker.task`), so the Pi needs no internet. Still missing from the repo: `clown nose.png` and `glasses.png`.

### 8.2 Arcade menu
- A row of arcade machines **replaces** the tile menu (and the carousel idea), so the menu needs no camera. Any tap anywhere on a machine starts it. Drawn entirely in code in `arcade_art.py`; `launcher.py` handles taps and starting filters.
- Black-and-white line art, with color only on the animated parts. The layout is planned on a design canvas and scaled to fit any screen, keeping proportions; tall screens get rows of two.
- Title "Arcade" in **Limelight**. Below it, one centered row of 4 symbols from the team, in machine order (`SUBTITLE_SYMBOLS` in `launcher.py`, added later). Plain characters like `:)` use the title font; emojis use the black-and-white **Noto Emoji** font. Both fonts are free (OFL) and live in `assets/fonts/`, so the Pi has them. The privacy line stays at the bottom.
- Machines, in order: **Clown Nose** (clown face with glasses; red nose pulses and glows), **Big Eyes** (two eyes slowly swell and pop back), **Iris Colors** (an eye whose iris changes color at each blink, using the filter's real three colors) and **Mystery** (same machine, blank screen for now).
- Still open: the delay while a filter loads, when the menu has closed and the filter hasn't appeared yet.

## 9. Open questions
- Tuning the timings (gun gap, shoot gap, grace, capture window, hold, leave-timeout) through real play.
- Symbols/animations for the moves (currently text on colored cards).
- Difficulty levels (currently one fixed `MAX_EXPLOIT`).
- Recording the real sounds (`sounds/*.wav`).
- Kiosk-mode setup on the Pi.
- Privacy/permission from the school administration.
