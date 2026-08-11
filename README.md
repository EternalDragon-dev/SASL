# Hand Gesture Recognizer — Manual Build

A real-time hand gesture and SASL fingerspelling recognizer built manually
as a learning exercise. Uses MediaPipe for landmark detection and OpenCV for
display. Written in Python.

---

## What it does

- Tracks up to 2 hands using MediaPipe's 21-point hand landmark model
- Draws the hand skeleton, fingertip labels, and palm center on screen
- Tracks both forearms using MediaPipe's 33-point pose landmark model
- Recognises 6 gestures in real time (Gesture mode)
- Recognises SASL fingerspelling letters using joint-angle classification (SASL mode)
- Skeleton and forearm color changes to match the detected gesture/letter

---

## Project structure

```
hand_gesture_recognizer_manual/
├── main.py               ← main application (webcam loop, drawing, classification)
├── joint_angles.py       ← computes 15 joint flexion angles from hand landmarks
├── sign_classifier.py    ← 22-dim feature vector + rule-based SASL letter classifier
├── calibrate.py          ← data collection tool for SASL training samples
├── models/
│   ├── hand_landmarker.task       ← MediaPipe hand landmark model (21 points)
│   └── pose_landmarker_lite.task  ← MediaPipe pose landmark model (33 points)
└── data/                 ← SASL training data goes here (one CSV per letter)
```

---

## Setup

Install dependencies (once):

```bash
pip install mediapipe opencv-python numpy
```

---

## Running

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Personal_Projects/hand_gesture_recognizer_manual
MPLBACKEND=Agg python main.py
```

> `MPLBACKEND=Agg` is required on macOS to prevent matplotlib (a MediaPipe
> dependency) from trying to open a display during its font scan, which hangs
> the app before the webcam window opens.

---

## Controls

| Key | Action |
|-----|--------|
| `M` | Toggle between GESTURE mode and SASL mode |
| `Q` | Quit |

---

## How the classification pipeline works

### Gesture mode

Each frame, `classify_gesture(lms)` evaluates a set of boolean rules based
on landmark y-coordinates:

- **Extended finger**: `tip.y < pip.y` (tip is higher in the frame = smaller y)
- **Thumb up**: tip clears both the thumb MCP and the index knuckle
- **Thumb lateral**: tip is meaningfully left of the IP joint (> 4% of frame width)

Rules are checked in order — most specific first — and the first match wins.
This is why Thumbs Up is checked before Fist (both have all four fingers curled).

Detected gestures: Thumbs Up, Peace, Rock On, Pointing, Fist, Open Hand.

### SASL mode

SASL (South African Sign Language) letter classification uses a richer
feature set than simple y-comparisons, making it more robust to hand rotation:

1. **`compute_joint_angles(lms)`** — computes 15 flexion angles (3 per finger)
   using the dot product of the two bone vectors meeting at each joint.
   A straight finger = ~180°; a fully curled finger = ~70–90°.

2. **`build_feature_vector(lms, angles)`** — assembles a 22-dimensional vector:
   - [0–14] 15 joint angles
   - [15] thumb abduction angle (how far thumb is spread from index)
   - [16–20] 5 tip-to-wrist distances, normalized by hand scale
   - [21] ring–pinky lateral spread angle

3. **`classify_static(fv, angles)`** — checks the vector against a set of
   rules (one per letter). Each rule defines a range for key joint angles.
   Returns `(letter, confidence)` if a rule scores ≥ 70%, otherwise `None`.

4. **Voting buffer** — the raw result for each frame is stored in a 14-frame
   ring buffer (`deque`). A letter is only displayed if it wins a majority
   vote (≥ 7 of 14 frames), which filters out single-frame noise.

Currently rule-based letters: **A, B, C, D, E, I, L, O, W, Y**

The remaining 14 static letters require an ML classifier trained on collected
data (see below).

---

## Collecting SASL training data

The goal of collection is to build a sufficiently large and varied set of
22-dimensional feature vectors (one row per capture) for each SASL letter.

If your copy of the repo includes a `collect.py` helper, run that. If not,
Use the included `calibrate.py` to capture training samples. It computes
`build_feature_vector()` for any detected hand and appending a CSV row to
`data/<LETTER>.csv`. `calibrate.py` additionally prints the observed
min/max ranges for each run, which is useful for bootstrapping rule-based ranges in `sign_classifier.py`.

Example (calibrate or collect):
`train_and_export.py` which exports an ONNX model that `sign_classifier.py`
```bash
MPLBACKEND=Agg python calibrate.py --letter F
MPLBACKEND=Agg python calibrate.py --letter G --auto   # capture automatically
```

Controls while running `calibrate.py`/`collect.py`:

- `SPACE` — Capture one sample (writes one row to `data/<LETTER>.csv`)
- `A` — Toggle auto-capture (one sample every `0.5` seconds) while a hand is visible
- `Q` — Quit and print a calibration summary

File format and feature order
-----------------------------
Each CSV row is the 22-dimensional vector produced by
`sign_classifier.build_feature_vector()` in this order:

- features 0–14: 15 joint flexion angles (degrees)
- feature 15: thumb abduction (degrees)
- features 16–20: fingertip-to-wrist distances normalized by hand scale
- feature 21: ring–pinky spread (degrees)

Recommended minimum: **~500 samples per letter**. Capture a broad range of
hand rotations, slight translations, and natural variation (pressure,
minor movement) to make the ML classifier robust.

Once enough data is collected, train the ML classifier using
`train_classifier.py` (in the Archives folder) which exports an ONNX model
that `sign_classifier.py` can load automatically.

How to test how well calibration worked
---------------------------------------
1. Visual sanity check (quick):
    - Run `calibrate.py --letter X` and capture a few dozen examples.
    - Quit and inspect the printed min/max ranges — they should be within
       expected geometric ranges (angles 0–180°, thumb abduction not NaN).

2. Quick programmatic rule test (one-shot):
    - Load one or more rows from `data/X.csv`, reconstruct the angles dict
       from the first 15 entries, and call `sign_classifier.classify_static()`
       to see whether the rule-based classifier matches your target letter.

Example Python snippet (run from the project root):

```bash
python - <<'PY'
import numpy as np
from joint_angles import JOINT_ANGLE_DEFS
from sign_classifier import classify_static

angle_keys = [name for name, *_ in JOINT_ANGLE_DEFS]
fv = np.loadtxt('data/A.csv', delimiter=',', ndmin=2)
for row in fv[:10]:
      angles = {k: float(row[i]) for i, k in enumerate(angle_keys)}
      res = classify_static(row, angles)
      print('result=', res)
PY
```

3. Live end-to-end test (recommended):
    - Start the real-time app:

```bash
MPLBACKEND=Agg python main.py
```

    - Switch to SASL / fingerspelling mode (press `M` if available), hold the
       target letter shape and observe whether the classifier displays the
       correct letter and confidence above the palm. Move the hand slightly to
       verify stability across frames (voting buffer reduces flicker).

4. Quantitative validation (best):
    - Collect a labelled validation set: for each letter, record a set of
       frames where you intentionally hold the correct shape and write their
       feature vectors and ground-truth label.
    - Run a small evaluation script that applies `classify_static()` to each
       row (reconstructing `angles` as above) and reports accuracy, false
       positives, and which letters are confused.

Example evaluation outline (pseudo):

```python
# load CSV rows and their ground-truth label for letter 'A'
# for each row: angles = {angle_keys[i]: row[i]}
#   pred = classify_static(row, angles)
#   tally whether pred matches 'A'
```

If the rule-based classifier underperforms on many samples, you either
need to widen the rule ranges (use the printed calibration block) or
collect more data and train the ML classifier.

## More detailed code breakdown

For a file-by-file description of the repository source code, see
[`FILE_BREAKDOWN.md`](FILE_BREAKDOWN.md).

## Learn the math and ML behind this project

If you want a student-friendly explanation of the geometry, machine
learning, and Python libraries used in this project, see
[`EDUCATION.md`](EDUCATION.md).

---

## MediaPipe landmark reference

### Hand model — 21 landmarks

```
        4          8    12    16   20   ← fingertips
        3          7    11    15   19   ← DIP joints
        2          6    10    14   18   ← PIP joints
        1          5     9    13   17   ← MCP (knuckles)
         \         |     |     |    |
          `--------0 (wrist)----------'
```

### Pose model — relevant landmarks for forearm

| Index | Landmark |
|-------|----------|
| 13 | Left elbow |
| 14 | Right elbow |
| 15 | Left wrist |
| 16 | Right wrist |

---

## Why Python?

Python is not the fastest language but is the right choice for this project:

- MediaPipe's best-supported API is Python
- The performance-critical parts (neural network inference) run in optimized
  C++ underneath regardless of which language calls them
- The ML training pipeline (scikit-learn, ONNX) needed for full SASL support
  lives in Python
- Development speed matters when iterating on classification logic

For a production mobile app (iOS/Android), the inference layer would be ported
to Swift or Kotlin while Python continues to handle model training.

---

## Next steps

- [ ] Collect training data for the remaining 14 SASL letters
- [ ] Train the ML classifier (`train_classifier.py`)
- [ ] Implement accuracy improvements: rotation-invariant finger extension,
      hysteresis on gesture transitions
- [ ] Add sequence classification for motion letters J and Z
- [ ] Add arm location features for full SASL sign recognition
