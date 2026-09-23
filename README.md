# SASL Handshape Recognizer

A real-time South African Sign Language (SASL) handshape recognizer built as a
learning project. It uses MediaPipe for landmark detection, OpenCV for the
camera and display, and a small ONNX model for static handshape inference.

The next development stage is documented in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). It specifies how to add
temporal segmentation, motion letters, completed letter events, words, and
optional token-based suggestions without requiring a local LLM.

For a fresh planning perspective that assumes the static foundation is
complete but treats all sequence work as unimplemented, see
[IMPLEMENTATION_PLAN_FROM_SCRATCH.md](IMPLEMENTATION_PLAN_FROM_SCRATCH.md).

---

## What it does

- Tracks up to 2 hands using MediaPipe's 21-point hand landmark model
- Draws the hand skeleton, fingertip labels, and palm center on screen
- Recognises SASL fingerspelling letters using joint-angle classification
- Uses an exported ONNX classifier when available, with rule-based fallback
- Smooths predictions across frames to reduce flicker
- Keeps the Python environment outside iCloud Drive for native-library stability

Current scope:

- Static letter recognition is implemented.
- A motion-aware subset of letters—including H, J, P, Q, and Z—requires
  trajectory recognition and must be calibrated explicitly.
- Word and sentence recognition are not yet implemented.
- Language-model suggestions are optional future functionality, not required by
   the recognition pipeline.

---

## Project structure

```
hand_gesture_recognizer_manual/
├── main.py               ← main application (webcam loop, drawing, classification)
├── joint_angles.py       ← computes 15 joint flexion angles from hand landmarks
├── sign_classifier.py    ← 22-dim feature vector + rule-based SASL letter classifier
├── calibrate.py          ← data collection tool for SASL training samples
├── motion_calibrate.py   ← ordered trajectory capture for motion letters
├── motion_classifier.py  ← trajectory normalization and template matching
├── train_motion_templates.py ← exports captured motion templates
├── motion_demo.py        ← live webcam demo for motion templates
├── train_and_export.py   ← trains and exports the static classifier
├── IMPLEMENTATION_PLAN.md← temporal, motion, word, and token roadmap
├── IMPLEMENTATION_PLAN_FROM_SCRATCH.md ← fresh-start sequence plan
├── models/
│   ├── hand_landmarker.task       ← MediaPipe hand landmark model (21 points)
└── data/                 ← SASL training data goes here (one CSV per letter)
```

---

## Setup

Use the local Python 3.11 environment so compiled libraries are stored outside
iCloud Drive. The project code and data can remain in iCloud Drive.

Create the environment once:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv ~/venvs/sasl311
~/venvs/sasl311/bin/python -m pip install --upgrade pip setuptools wheel
~/venvs/sasl311/bin/python -m pip install opencv-python mediapipe numpy scikit-learn pandas joblib skl2onnx onnx onnxruntime
```

Activate it for each session:

```bash
source ~/venvs/sasl311/bin/activate
```

---

## Running

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Personal_Projects/hand_gesture_recognizer_manual
source ~/venvs/sasl311/bin/activate
MPLBACKEND=Agg python main.py
```

> `MPLBACKEND=Agg` avoids matplotlib display initialization during startup.
> On this Mac, the app tries camera index 1 first because index 0 may be the
> iPhone Continuity Camera and can stop providing frames after a short time.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |

---

## How the classification pipeline works

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

The rule-based path is a conservative fallback. The trained ONNX model covers
the calibrated static letters listed in `models/sign_classifier_labels.json`.
The motion subset is not limited to J and Z in the final design: letters such as
H, J, P, Q, and Z must be checked against live motion data and routed to a
trajectory-aware classifier when appropriate rather than forced into one static
22-dimensional vector.

### Planned sequence pipeline

The current voting buffer smooths frames; it does not know when a letter is
complete. The planned Recommendation 6 architecture adds this separation:

```text
MediaPipe landmarks
   -> temporal features (palm position, velocity, speed)
   -> boundary state machine
   -> static classifier or motion-letter trajectory classifier
   -> completed letter event
   -> word buffer
   -> optional token suggestions
```

The first sequence milestone is a reliable event stream such as:

```text
LETTER(A), LETTER(B), LETTER(J), WORD_END
```

The complete implementation sequence, thresholds, data formats, tests,
metrics, and escalation options are in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

---

## Collecting SASL training data

The goal of collection is to build a sufficiently large and varied set of
22-dimensional feature vectors (one row per capture) for each SASL letter.

Use the included `calibrate.py` to capture training samples. It computes
`build_feature_vector()` for any detected hand and appends a CSV row to
`data/<LETTER>.csv`. `calibrate.py` additionally prints the observed
min/max ranges for each run, which is useful for bootstrapping rule-based ranges in `sign_classifier.py`.

Example calibration commands:
```bash
MPLBACKEND=Agg python calibrate.py --letter F
MPLBACKEND=Agg python calibrate.py --letter G --auto   # capture automatically
```

Motion-letter calibration uses a separate ordered-sequence format. Do not put
motion trajectories in the static `data/<LETTER>.csv` files. Start the motion
recorder, press `SPACE` to begin a gesture, perform the motion, then press
`SPACE` again to save the sequence:

```bash
MPLBACKEND=Agg python motion_calibrate.py --letter J
MPLBACKEND=Agg python motion_calibrate.py --letter Z
```

The same recorder can be used for H, P, and Q while their motion/static status
is being validated. Samples are written to `motion_data/<LETTER>/sample_*.csv`.
The current `motion_classifier.py` provides offline trajectory normalization and
template matching; live integration into the sequence coordinator is still
pending.

After collecting motion samples, export them with:

```bash
python train_motion_templates.py
```

This creates `models/motion_templates.npz`. Unlike the static classifier, this
is a normalized template artifact rather than an ONNX model. It can be tested
offline with `motion_classifier.classify_trajectory()` and will be connected to
the live sequence coordinator in the next integration step.

To try the exported motion templates yourself through the webcam:

```bash
python motion_demo.py
```

Press `SPACE`, perform one recorded motion, then press `SPACE` again. The demo
will display the closest motion label and confidence, or reject the trajectory
when no template is close enough. Press `Q` to quit.

Controls while running `calibrate.py`:

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
`train_and_export.py`, which writes the joblib fallback, validation metrics,
label mapping, and an ONNX model that `sign_classifier.py` can load.

```bash
source ~/venvs/sasl311/bin/activate
python train_and_export.py
```

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

      - Hold the target letter shape and observe whether the classifier displays
         the correct letter and confidence above the palm. Move the hand slightly
         to verify stability across frames (the voting buffer reduces flicker).

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

## Learning model

The project separates four ideas:

1. **Detection**: MediaPipe turns camera pixels into 21 hand landmarks.
2. **Features**: geometry turns landmarks into angles and normalized distances.
3. **Recognition**: rules or a trained classifier map features to static letters.
4. **Sequence prediction**: temporal logic finds boundaries; optional token
   models suggest likely words or sentences.

A local LLM is not needed for the first three steps or for the initial
prefix-dictionary suggestion layer. This distinction is important: the
handshape classifier is machine learning for numeric features, while a
language model predicts sequences of letters or words.

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

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full checklist.
The immediate order is:

- [x] Add initial temporal observations and boundary logic
- [x] Emit completed static letter events through the existing classifier
- [x] Add ordered motion capture, template export, and a live motion demo
- [ ] Collect and validate motion samples for H, J, P, Q, and Z
- [ ] Connect motion templates to the main sequence coordinator
- [ ] Tune live letter transitions and automatic word boundaries
- [ ] Add prefix suggestions without an LLM
- [ ] Evaluate whether an n-gram or neural language model is justified
