"""
train.py - train the pose classifier (MLP) from the CSVs that collect.py recorded.

What it does, in order:
  1. Loads every CSV in data/raw/.
  2. Splits the data into TRAIN and TEST **by person**: everyone in the test set
     is someone the model has never seen. (Splitting by frame would cheat:
     frames from the same take are nearly identical, so accuracy would look fake-high.)
  3. Adds mirrored (left<->right flipped) copies of the training frames, so the
     model learns both-handed versions of every pose.
  4. Normalizes everything with features.normalize() (the same code live play uses).
  5. Trains an MLP and prints a report on the TEST people:
        - accuracy per class, and a confusion matrix (which poses get mixed up)
        - accuracy per GAME MOVE (e.g. mirror_flat vs mirror_fingers mix-ups don't matter)
        - what happens at different confidence thresholds
        - the "none check": how often non-poses would sneak past the confidence
          threshold if there were NO none class
  6. Retrains on ALL the data and saves the model to models/pose_classifier.joblib.

Usage:
  py train.py
  py train.py --no-mirror        (turn off the left/right flip augmentation)

Setup:
  py -m pip install scikit-learn
"""

import argparse
import csv
import glob
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features

# =============================================================================
# Settings
# =============================================================================

DATA_DIR = os.path.join(features.HERE, "data", "raw")
MODEL_PATH = os.path.join(features.HERE, "models", "pose_classifier.joblib")

TEST_FRACTION = 0.25      # roughly this share of PEOPLE go into the test set
HIDDEN_LAYERS = (64, 32)  # MLP size: two hidden layers with 64 and 32 neurons
CONF_THRESHOLD = 0.75     # the threshold live play uses (reported on below)

# When a frame is mirrored, a left-handed pose becomes a right-handed one.
# Any label listed here gets swapped; every other label stays the same.
FLIP_LABELS = {"shoot_left": "shoot_right", "shoot_right": "shoot_left"}


# =============================================================================
# 1. Loading
# =============================================================================


def load_data():
    """Read every CSV. Returns (raw rows, labels, person IDs, take IDs) as numpy arrays."""
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSV files in {DATA_DIR}. Record some data with collect.py first.")

    raws, labels, people, takes = [], [], [], []
    unknown_labels = set()
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = set(features.RAW_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                # Happens if features.py's landmark list changed after this file was recorded.
                print(f"SKIPPING {os.path.basename(path)}: it's missing {len(missing)} columns")
                continue
            for row in reader:
                if row["label"] not in features.CLASSES:
                    unknown_labels.add(row["label"])
                    continue
                raws.append([float(row[c]) for c in features.RAW_COLUMNS])
                labels.append(row["label"])
                people.append(row["person_id"])
                # A "take" = one recording of one pose; used for splitting when there are few people.
                takes.append(f"{os.path.basename(path)}|{row['label']}|{row['take']}")

    if unknown_labels:
        print(f"Ignored rows with labels not in features.CLASSES: {sorted(unknown_labels)}")
    if not raws:
        raise SystemExit("No usable rows found.")
    return np.array(raws, dtype=np.float32), np.array(labels), np.array(people), np.array(takes)


# =============================================================================
# 2. Mirror augmentation (done on RAW rows, before normalizing)
# =============================================================================


def _build_mirror_tables():
    """Work out, once, which column swaps with which when the image is flipped."""
    index = {c: i for i, c in enumerate(features.RAW_COLUMNS)}
    swap = []
    for c in features.RAW_COLUMNS:
        if c.startswith("left_"):
            other = "right_" + c[len("left_"):]
        elif c.startswith("right_"):
            other = "left_" + c[len("right_"):]
        else:
            other = c
        swap.append(index[other])
    x_cols = [i for i, c in enumerate(features.RAW_COLUMNS) if c.endswith("_x")]
    return np.array(swap), np.array(x_cols), index["frame_w"]


_SWAP, _X_COLS, _FRAME_W = _build_mirror_tables()


def mirror_raw(raws: np.ndarray) -> np.ndarray:
    """Flip frames left<->right, as if the person had done the pose mirrored.
       - every left_* column trades places with its right_* column
         (the person's right hand becomes their left hand)
       - every x becomes (frame width - x)
    """
    flipped = raws[:, _SWAP].copy()
    flipped[:, _X_COLS] = flipped[:, [_FRAME_W]] - flipped[:, _X_COLS]
    return flipped


def mirror_labels(labels: np.ndarray) -> np.ndarray:
    return np.array([FLIP_LABELS.get(label, label) for label in labels])


# =============================================================================
# 3. Model
# =============================================================================


def make_model():
    # StandardScaler: rescales each feature to mean 0 / std 1, which MLPs train much better with.
    # alpha: regularization; it keeps the weights small so the model doesn't just memorize the training frames.
    # (early_stopping=True would be nicer, but scikit-learn 1.8 crashes with it when labels are text.)
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, alpha=1e-3, max_iter=500, random_state=0),
    )


def to_features(raws: np.ndarray) -> np.ndarray:
    return np.stack([features.normalize(r) for r in raws])


def to_moves(labels) -> np.ndarray:
    """Class names -> game moves (none stays 'none')."""
    return np.array([features.CLASS_TO_MOVE[label] or "none" for label in labels])


# =============================================================================
# 4. Reporting
# =============================================================================


def print_confusion(y_true, y_pred, labels, title):
    """Rows = what the person actually did, columns = what the model said."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    short = [label[:8] for label in labels]
    width = max(len(label) for label in labels) + 2
    print(f"\n{title}  (rows = actual, columns = predicted)")
    print(" " * width + "".join(f"{s:>9}" for s in short))
    for label, row in zip(labels, matrix):
        print(f"{label:<{width}}" + "".join(f"{n:>9}" for n in row))


def report_thresholds(model, X_test, y_test):
    """For several thresholds: how many frames are confident enough to count, and how accurate those are."""
    proba = model.predict_proba(X_test)
    confidence = proba.max(axis=1)
    predicted_moves = to_moves(model.classes_[proba.argmax(axis=1)])
    actual_moves = to_moves(y_test)

    print("\nConfidence thresholds (move-level, test people):")
    print("  threshold   frames kept   accuracy of kept frames")
    for t in (0.5, 0.6, 0.75, 0.9):
        kept = confidence >= t
        acc = (predicted_moves[kept] == actual_moves[kept]).mean() if kept.any() else float("nan")
        marker = "   <- used by live play" if t == CONF_THRESHOLD else ""
        print(f"  {t:>9.2f}   {kept.mean():>10.0%}   {acc:>10.1%}{marker}")


def none_check(raw_train, y_train, raw_test, y_test, use_mirror):
    """Train a model WITHOUT the none class, then feed it the test people's none frames.
    Any none frame scoring >= CONF_THRESHOLD on some pose would have been a false move."""
    test_none = y_test == "none"
    if not test_none.any() or not (y_train == "none").any():
        print("\nNone check skipped: need 'none' data in both the train and test sets.")
        return

    keep = y_train != "none"
    raw, y = raw_train[keep], y_train[keep]
    if use_mirror:
        raw, y = np.concatenate([raw, mirror_raw(raw)]), np.concatenate([y, mirror_labels(y)])
    model = make_model().fit(to_features(raw), y)

    proba = model.predict_proba(to_features(raw_test[test_none]))
    confident = proba.max(axis=1) >= CONF_THRESHOLD
    print(f"\nNone check: a model trained WITHOUT 'none' was shown {test_none.sum()} 'none' frames.")
    print(f"  {confident.mean():.0%} of them scored >= {CONF_THRESHOLD:.0%} on some pose "
          "(each one would have been a fake move without a none class).")
    if confident.any():
        fooled = model.classes_[proba[confident].argmax(axis=1)]
        names, counts = np.unique(fooled, return_counts=True)
        print("  They were mistaken for: " + ", ".join(f"{n} ({c})" for n, c in zip(names, counts)))


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train the Gun Gun Shoot pose classifier.")
    parser.add_argument("--no-mirror", action="store_true", help="don't add left/right flipped training copies")
    args = parser.parse_args()
    use_mirror = not args.no_mirror

    raws, labels, people, takes = load_data()
    n_people = len(set(people))
    print(f"Loaded {len(raws)} frames from {n_people} people.")
    for label in features.CLASSES:
        print(f"  {label:<16}{(labels == label).sum():>6} frames")

    # --- Split by person (or by take, if there aren't enough people yet) ---
    if n_people >= 3:
        groups = people
    else:
        groups = takes
        print(f"\nWARNING: only {n_people} person(s). Splitting by TAKE instead of by person, so the test")
        print("results will look better than they really are. Record more people for honest numbers.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=0)
    train_idx, test_idx = next(splitter.split(raws, labels, groups))
    raw_train, y_train = raws[train_idx], labels[train_idx]
    raw_test, y_test = raws[test_idx], labels[test_idx]
    if n_people >= 3:
        print(f"\nTest people (never seen in training): {sorted(str(p) for p in set(people[test_idx]))}")

    # --- Augment the TRAINING set only (the test set stays real data) ---
    if use_mirror:
        raw_train_aug = np.concatenate([raw_train, mirror_raw(raw_train)])
        y_train_aug = np.concatenate([y_train, mirror_labels(y_train)])
    else:
        raw_train_aug, y_train_aug = raw_train, y_train

    # --- Train + evaluate ---
    print(f"\nTraining on {len(raw_train_aug)} frames"
          f"{' (half of them mirrored copies)' if use_mirror else ''}, testing on {len(raw_test)}...")
    model = make_model().fit(to_features(raw_train_aug), y_train_aug)
    X_test = to_features(raw_test)
    y_pred = model.predict(X_test)

    present = [c for c in features.CLASSES if c in set(y_test) | set(y_pred)]
    print("\nPer-class results on the test set:")
    print(classification_report(y_test, y_pred, labels=present, zero_division=0))
    print_confusion(y_test, y_pred, present, "Confusion matrix (classes)")

    moves_true, moves_pred = to_moves(y_test), to_moves(y_pred)
    move_names = [m for m in ("reload", "shoot", "shield", "mirror", "none") if m in set(moves_true) | set(moves_pred)]
    print(f"\nMove-level accuracy: {(moves_true == moves_pred).mean():.1%}")
    print_confusion(moves_true, moves_pred, move_names, "Confusion matrix (game moves)")

    report_thresholds(model, X_test, y_test)
    none_check(raw_train, y_train, raw_test, y_test, use_mirror)

    # --- Final model: retrain on everything and save ---
    if use_mirror:
        raw_all, y_all = np.concatenate([raws, mirror_raw(raws)]), np.concatenate([labels, mirror_labels(labels)])
    else:
        raw_all, y_all = raws, labels
    print(f"\nRetraining on all {len(raw_all)} frames for the final model...")
    final = make_model().fit(to_features(raw_all), y_all)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": final,
            # Saved so live_test.py / game.py can refuse to run if features.py changed since training.
            "feature_names": features.FEATURE_NAMES,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
