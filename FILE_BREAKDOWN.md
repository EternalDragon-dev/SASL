# File-by-File Code Breakdown

This repository contains the following key Python source files for building a real-time SASL handshape recognizer and training an ML classifier from collected hand samples.

## `joint_angles.py`

Purpose:
- Compute 15 finger joint flexion angles from MediaPipe hand landmarks.
- Provide a stable geometric representation of handshape.

Capabilities:
- Defines `JOINT_ANGLE_DEFS`, a list of joint names and landmark index triplets.
- Implements `compute_joint_angles(landmarks)`:
  - Takes MediaPipe hand landmarks (21 points).
  - Computes a joint angle for each defined finger joint using vector math.
  - Returns a dictionary mapping angle names to degrees (0–180).

Usage:
- Used by `sign_classifier.py` to build the 22-dimensional feature vector.
- Used by `calibrate.py` and `main.py` to generate angle-based features for SASL classification.

## `sign_classifier.py`

Purpose:
- Build a 22-dimensional feature vector from hand landmarks.
- Classify static SASL handshapes using rule-based logic.
- Load and run a trained ML classifier when available.

Capabilities:
- `build_feature_vector(landmarks, angles)`:
  - Produces a feature vector of length 22.
  - Features 0–14: 15 joint flexion angles.
  - Feature 15: thumb abduction angle.
  - Features 16–20: fingertip-to-wrist distances normalized by hand scale.
  - Feature 21: ring–pinky spread angle.
- Rule-based classification (`classify_static`):
  - Uses `_RULES` to match handshape constraints for SASL letters.
  - Computes a confidence score from how many rule constraints are satisfied.
  - Returns `(label, confidence)` only when the score is high enough.
- ML classifier helpers:
  - `load_ml_classifier(model_path)` loads an ONNX model via `onnxruntime`.
  - `classify_static_ml(session, feature_vector, threshold)` infers a label/probability from the ONNX model.

Notes:
- The rule-based rules in `_RULES` are currently the baseline for static letters.
- ML classification is only enabled when `models/sign_classifier.onnx` exists and `onnxruntime` is installed.

## `calibrate.py`

Purpose:
- Capture handshape feature vectors for a single SASL letter.
- Save labelled samples to `data/<LETTER>.csv`.
- Print observed angle ranges and a block suitable for bootstrapping rule definitions.

Capabilities:
- Accepts `--letter` and optional `--auto` command-line arguments.
- Opens the webcam and uses MediaPipe HandLandmarker to detect one hand.
- Displays a live overlay with hand skeleton and per-joint values.
- Supports manual capture with `SPACE`.
- Supports auto capture with `A` to toggle one sample every 0.5 seconds.
- Writes each captured 22-dimensional feature vector to a CSV file.
- Tracks min/max observed values for all calibrated features.
- Prints a summary and a ready-to-paste rule block after quitting.

Usage:
- `MPLBACKEND=Agg python calibrate.py --letter A`
- `MPLBACKEND=Agg python calibrate.py --letter B --auto`

On macOS, the camera helper tries index 1 before index 0. This prioritizes the
Mac camera when an iPhone Continuity Camera is also connected.

## `main.py`

Purpose:
- Run the real-time SASL handshape recognizer from webcam input.
- Visualize hand landmarks and detected letters in a live window.

Capabilities:
- Opens a webcam, preferring camera index 1 before index 0, and a MediaPipe
  HandLandmarker model.
- Mirrors the webcam image and draws hand skeletons, fingertips, and palm center.
- Computes joint angles and builds the 22D SASL feature vector every frame.
- Attempts to load `models/sign_classifier.onnx` and use the ML classifier if available.
- Falls back to `sign_classifier.classify_static()` if no ONNX model is loaded.
- Uses a voting buffer (`deque` of length 14) to smooth letter output across frames.
- Displays the detected hand label, the current SASL letter, and confidence percentage.

Notes:
- This is the live application entrypoint.
- It relies on the `sign_classifier` and `joint_angles` modules for feature extraction and classification.
- Use Python 3.11 from `~/venvs/sasl311`; keep the environment outside iCloud
  Drive so OpenCV's native libraries remain locally available.

## `train_and_export.py`

Purpose:
- Train a machine learning model from collected `data/*.csv` samples.
- Save the trained model as a joblib fallback and export an ONNX file for runtime inference.

Capabilities:
- Loads labelled feature vectors from CSV files in `data/`.
- Supports training either an MLP or Random Forest classifier.
- Standardizes features with `StandardScaler`.
- Splits data into training and validation sets with stratified sampling.
- Evaluates model performance and writes metrics to `models/metrics.json`.
- Saves a scikit-learn pipeline and label encoder to `models/sign_classifier.joblib`.
- Attempts to export the trained pipeline to `models/sign_classifier.onnx` using `skl2onnx`.

Notes:
- If ONNX export is unavailable, the joblib pipeline remains as a fallback.
- The resulting `models/sign_classifier.onnx` is the file `main.py` tries to load for live ML inference.

## How the files fit together

- `joint_angles.py` computes the low-level geometric angles.
- `sign_classifier.py` turns those angles into a feature vector and decides a letter.
- `calibrate.py` creates new `data/<LETTER>.csv` training samples.
- `train_and_export.py` turns those CSV samples into a trained ML model.
- `main.py` uses the trained model (or the rule-based fallback) to recognize letters live.

## Student learning resources

For an explanation tailored to a robotics student, including the math used to compute angles, the purpose of ONNX, and the libraries involved, see [`EDUCATION.md`](EDUCATION.md).
