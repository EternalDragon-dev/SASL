# Education Guide for Hand Gesture Recognizer

This file explains the project from a robotics student perspective. It covers the math, the machine learning concepts, the Python syntax, libraries used, and how the files work together.

## 1. What this project actually does

This project builds a real-time handshape recognizer using a webcam. It detects a hand, computes geometric features from the hand pose, and then decides which SASL letter the shape represents.

There are two main recognition methods:
- Rule-based classification: deterministic rules over joint angles
- ML classification: a trained model that learns from example hand samples

It is not a language model (LM) project. It is a computer vision + pattern recognition project that uses machine learning (ML), not natural language generation.

## 2. The data flow: from camera to detected letter

1. `main.py` reads a frame from the webcam.
2. MediaPipe finds the hand and gives 21 landmarks (x, y, z points).
3. `joint_angles.py` computes 15 joint angles from those landmarks.
4. `sign_classifier.py` builds a 22-dimensional feature vector from the angles and distances.
5. The classifier decides a letter either by:
   - rules in `classify_static()`, or
   - ML inference in `classify_static_ml()` with an ONNX model.
6. `main.py` smooths the output with a small voting buffer so letters do not flicker.

## 3. The mathematics used

### 3.1 Geometry and vectors

A hand landmark is a point in 2D or 3D space. The project uses these points to build vectors.

A vector is simply a direction and magnitude between two points. For example, if landmark A is at `(x1, y1)` and landmark B is at `(x2, y2)`, the vector from A to B is:

- `dx = x2 - x1`
- `dy = y2 - y1`

This project uses these vectors to describe bones in the hand.

### 3.2 Dot product and angles

The angle between two vectors is computed with the dot product. In mathematical form:

- `v · w = |v| * |w| * cos(theta)`

Where:
- `v · w` is the dot product of vectors `v` and `w`
- `|v|` and `|w|` are the lengths (magnitudes) of the vectors
- `theta` is the angle between them

Rearranged to get the angle:

- `theta = acos((v · w) / (|v| * |w|))`

In the code, that is used to compute joint flexion angles around finger joints. Straight fingers produce angles near 180°, while curled joints produce smaller angles.

### 3.3 Distance normalization

The feature vector also includes distances from each fingertip to the wrist. Those distances are divided by a stable hand scale so the numbers are not affected by the size of the person’s hand or how close the hand is to the camera.

The hand scale is the distance between the wrist (landmark 0) and the middle knuckle (landmark 9). That is a stable reference length.

### 3.4 Confidence and thresholds

The rule-based classifier uses confidence as a measure of how well the current handshape matches a set of rules. If the confidence is below a threshold (for example `0.70`), the result is rejected and treated as invalid.

The ML classifier also returns a probability value. If the model is not confident enough, the code can return `None` so the application does not show a wrong letter.

## 4. What ONNX is and why it is used

### 4.1 What is ONNX?

ONNX stands for Open Neural Network Exchange. It is a model format for machine learning models.

The main idea is:
- train a model in one environment (for example Python + scikit-learn)
- export it to ONNX
- load it in another environment with an ONNX runtime

In this project, the runtime is `onnxruntime`, which can run the exported model efficiently without needing the original training libraries.

### 4.2 Why export to ONNX?

Benefits:
- portability: the model becomes a single file (`models/sign_classifier.onnx`)
- speed: `onnxruntime` is optimized for inference
- separation: training and runtime are decoupled

This means `main.py` can use the trained classifier without importing scikit-learn directly.

## 5. The libraries and tools used

### 5.1 Python

The repository is written in Python. Important Python features you should know here:
- `import`: bring modules into your file
- `def`: define a function
- `if __name__ == "__main__"`: run code only when the file is executed directly
- `dict`, `list`, `tuple`: basic data containers
- type hints like `-> dict[str, float]` help document expected return types
- `with` blocks manage resources cleanly (like opening the MediaPipe model)

### 5.2 NumPy (`numpy`)

NumPy provides efficient numerical arrays and math operations.
- `np.array(...)` builds arrays
- `np.vstack(...)` stacks rows into a matrix
- `np.loadtxt(...)` reads CSV data
- `np.max(...)` and `np.argmax(...)` pick the largest value

### 5.3 OpenCV (`cv2`)

OpenCV handles the webcam, drawing, and display.
- `cv2.VideoCapture(0)` opens the webcam
- `cv2.imshow(...)` shows a window
- `cv2.putText(...)` draws text on the image
- `cv2.line(...)` and `cv2.circle(...)` draw skeletons and landmarks

### 5.4 MediaPipe

MediaPipe provides the hand and pose detectors. It converts the webcam image into landmarks.
- `HandLandmarker` detects 21 hand landmarks
- The model output is a list of points like `(x, y, z)`
- These landmarks are the raw input to the geometry math

### 5.5 scikit-learn (`sklearn`)

This is the ML training library.
- `StandardScaler`: normalizes features to zero mean and unit variance
- `MLPClassifier`: trains a small neural network classifier
- `Pipeline`: chains preprocessing and classifier together
- `train_test_split`: splits a dataset into training and validation sets
- `classification_report`: reports accuracy and confusion

### 5.6 ONNX and `onnxruntime`

- `skl2onnx` converts a scikit-learn pipeline to the ONNX format.
- `onnxruntime` loads that model in `main.py` for fast inference.

### 5.7 `joblib`

Saves the scikit-learn pipeline as a Python artifact. It is used as a fallback when ONNX export is not available.

## 6. Classical ML vs language models vs rule-based methods

### 6.1 Rule-based classification

`sign_classifier.py` has a rule-based path (`classify_static`). It checks human-defined ranges for joint angles.

This is:
- easy to understand
- explainable
- limited by the rules you write

Example rule: letter `B` requires all four fingers extended and together.

### 6.2 Machine learning classification

The ML path uses examples of actual hand shapes to learn the decision boundaries.

This is:
- more flexible than hard-coded rules
- able to handle subtle variations
- dependent on good training data

### 6.3 Language models (LMs)

This project does not use a language model.

A language model is a system trained to predict or generate text or language sequences, like GPT.
This project uses a different type of model: a handshape classifier that works with numeric feature vectors.
If you see `ML` in this project, it means classical machine learning, not natural language processing.

## 7. File-by-file mental model

- `joint_angles.py`: low-level geometry. Converts landmark coordinates into angles.
- `sign_classifier.py`: feature engineering + classifier logic.
- `calibrate.py`: data capture tool. Records feature vectors for a chosen letter.
- `train_and_export.py`: trains the ML model and exports it to ONNX.
- `main.py`: live application. Uses the webcam, detects hands, and displays results.

## 8. Practical study advice

1. Start by running `calibrate.py` for one letter and watch the printed angle values.
2. Open `joint_angles.py` and trace how the two vectors are built for one joint.
3. In `sign_classifier.py`, examine one rule in `_RULES` and compare it to the shape you made.
4. If you want to understand ML, inspect `train_and_export.py` and see how `X` and `y` are built from `data/*.csv`.
5. Finally, run `main.py` and hold one shape for several seconds to see how the vote buffer stabilizes the output.

## 9. Why `F` can appear too often

This is a useful robotics lesson: when your model only knows two classes, it will always choose the closest known class.

If your dataset has only `A` and `F`, then any unfamiliar shape is forced into either `A` or `F`. This is not a sign of wrong math, it is the nature of classification without an explicit "unknown" category.

To fix that, you need:
- more letters in the training data,
- better balance between classes,
- a stricter confidence threshold,
- or an explicit reject/unknown class.

---

## 10. Helpful terms glossary

- Landmark: a detected key point on the hand
- Feature vector: a numeric list of values used by a classifier
- Joint angle: the angle at a finger joint, measured in degrees
- Normalization: scaling values so they become comparable
- ONNX: a portable model format for inference
- Inference: using a trained model to make a prediction
- Rule-based: decisions made by deterministic logical conditions
- ML: decisions learned from examples
- LM: language model, not used here
