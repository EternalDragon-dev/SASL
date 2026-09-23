# SASL Hand Gesture Recognizer
# Retrospective Implementation Plan From Project Inception to Present

## Purpose

This document reconstructs the project as an implementation plan written at the moment the project was first conceived. It describes the decisions, phases, requirements, problems, and validation steps that carried the project from an empty starting point to its current state.

The plan is written in the past tense because those phases have already happened. The final section marks the exact boundary between completed work and work that has not yet started.

This document therefore answers two questions:

1. **How was the project intended to be built from the beginning?**
2. **What had been completed by the current moment, and what remained?**

The project was a real-time South African Sign Language (SASL) handshape recognizer. Its original scope was fingerspelled letter recognition, with a future path toward motion letters, words, and token-based sentence suggestions.

---

## 1. Original Project Definition

### 1.1 Problem statement

The project was intended to observe a signer through a webcam and recognize SASL handshapes in real time.

The initial version was deliberately limited to static handshapes. A static handshape is a letter whose identity can be inferred mostly from the arrangement of the fingers at one moment.

The intended first pipeline was:

```text
webcam image
    -> hand landmark detection
    -> geometric feature extraction
    -> static handshape classification
    -> temporal smoothing
    -> displayed SASL letter
```

### 1.2 Original non-goals

The first version did not attempt to solve all of sign-language understanding. It did not initially include:

- Full lexical SASL sign recognition
- Facial expression recognition
- Body-pose grammar
- Sentence translation
- Natural-language generation
- Motion-letter recognition
- Continuous word segmentation
- A local LLM

Those capabilities were treated as later stages because they require temporal and linguistic information that one static handshape does not contain.

### 1.3 Initial learning goals

The project was also a learning exercise. It was intended to teach:

- Webcam and image processing
- MediaPipe landmark detection
- 2D and 3D geometry
- Feature engineering
- Rule-based classification
- Supervised machine learning
- Model evaluation
- ONNX model export and inference
- Real-time application design

The implementation was therefore planned to progress from transparent geometry to learned classification rather than beginning with an opaque end-to-end neural model.

---

## 2. Project Requirements at Inception

The original implementation requirements were divided into functional, technical, educational, and operational requirements.

### 2.1 Functional requirements

The application needed to:

1. Open a camera.
2. Detect one or more hands.
3. Obtain the 21 MediaPipe landmarks for a hand.
4. Draw the hand skeleton for visual feedback.
5. Compute a stable numerical representation of the handshape.
6. Recognize supported SASL static letters.
7. Display a letter only when the prediction was sufficiently stable.
8. Collect examples for training.
9. Train a classifier from collected examples.
10. Run the exported model during live inference.

### 2.2 Technical requirements

The implementation needed to:

- Use Python.
- Use MediaPipe for hand landmarks.
- Use OpenCV for camera input and display.
- Use NumPy for numerical calculations.
- Keep feature dimensions consistent between calibration, training, and inference.
- Provide a rule-based fallback when the trained model was unavailable.
- Avoid blocking the UI with unnecessary work.
- Handle camera read failures without crashing immediately.

### 2.3 Educational requirements

The code needed to remain understandable to a student. Important calculations were to be explicit enough that a reader could trace:

```text
landmark -> vector -> angle -> feature -> classifier decision
```

### 2.4 Operational requirements

The application needed to run reliably on the available Mac. This became especially important when native Python libraries and camera devices behaved differently from the ideal development setup.

---

## 3. Phase One: Repository and Toolchain Setup

### 3.1 Initial repository structure

The project was organized around a small number of focused Python files:

```text
main.py
joint_angles.py
sign_classifier.py
calibrate.py
train_and_export.py
models/
data/
```

The responsibilities were separated as follows:

| File or directory | Planned responsibility |
|---|---|
| `main.py` | Live application and camera loop |
| `joint_angles.py` | Joint-angle geometry |
| `sign_classifier.py` | Feature-vector construction and static classification |
| `calibrate.py` | Training-data collection |
| `train_and_export.py` | Model training and export |
| `models/` | MediaPipe and classifier artifacts |
| `data/` | Collected feature vectors |

### 3.2 Python version

Python 3.11 was selected because it provided a stable compatibility target for the project’s scientific and computer-vision libraries.

The environment was ultimately created outside iCloud Drive:

```bash
/opt/homebrew/bin/python3.11 -m venv ~/venvs/sasl311
source ~/venvs/sasl311/bin/activate
```

### 3.3 Dependency plan

The required libraries were installed into the local environment:

- `opencv-python`
- `mediapipe`
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `joblib`
- `skl2onnx`
- `onnx`
- `onnxruntime`
- `matplotlib` as a MediaPipe-related dependency

The environment was kept outside iCloud Drive because the virtual environment’s native libraries did not load reliably when their files were synchronized or temporarily unavailable through iCloud.

### 3.4 Operational lesson

The project learned that Python source files and data could remain in iCloud Drive, but compiled libraries were safer in a local filesystem location. This became a permanent setup requirement.

---

## 4. Phase Two: Camera and Startup Reliability

### 4.1 Initial camera plan

The first camera implementation attempted to open a camera and read frames in the usual way:

```python
cap = cv2.VideoCapture(index)
ret, frame = cap.read()
```

The application was expected to release the camera and destroy OpenCV windows during cleanup.

### 4.2 Failure discovered

The application showed application-not-responding behavior. Investigation found two important contributors:

1. The iPhone Continuity Camera could begin streaming and then stop providing frames after approximately 90 frames.
2. OpenCV’s native libraries were vulnerable to loading problems when the virtual environment lived inside iCloud-backed storage.

Memory pressure was also investigated, but it was not identified as the primary cause of the recurring camera failure.

### 4.3 Camera-selection solution

The camera-opening helper was changed to try the stable Mac camera first and the iPhone camera second:

```text
camera indices: (1, 0)
```

The helper warmed each candidate camera by reading several frames before accepting it. A camera that opened but failed to provide valid warm-up frames was released and the next candidate was tried.

### 4.4 Frame-read recovery

The frame loop was changed to tolerate transient failures. Instead of exiting on the first failed read, it:

- Counted consecutive failed reads.
- Waited briefly between attempts.
- Continued when a valid frame returned.
- Exited only after approximately 30 consecutive failures.

This represented roughly 1.5 seconds of tolerance under the chosen delay.

### 4.5 Camera validation

The camera behavior was tested by reading frames for an extended period. The Mac camera remained stable for more than 120 reads, while the Continuity Camera failed after approximately 92 frames in the observed test.

### 4.6 Result of this phase

The camera and startup work produced a stable operational foundation:

- The local Python environment avoided iCloud native-library loading problems.
- The stable camera was preferred automatically.
- Temporary frame failures did not immediately terminate the application.
- The user did not need to choose a camera index manually during normal use.

---

## 5. Phase Three: MediaPipe Hand Landmark Detection

### 5.1 Model selection

The project used MediaPipe’s hand landmark model:

```text
models/hand_landmarker.task
```

The model provided 21 landmarks per detected hand.

The landmarks were indexed consistently:

```text
0   wrist
1-4   thumb
5-8   index finger
9-12  middle finger
13-16 ring finger
17-20 little finger
```

The fingertip indices were:

```text
4, 8, 12, 16, 20
```

### 5.2 Runtime mode

MediaPipe’s video mode was used so that timestamps could help the model maintain temporal tracking.

The application supplied monotonically increasing timestamps based on `time.monotonic()`.

### 5.3 Hand drawing

The application drew:

- Bone connections
- Landmark points
- Fingertip labels
- Palm center
- Current classification state

This visual overlay was important for debugging because it allowed the student to compare the detected geometry with the physical hand position.

### 5.4 World landmarks

Where available, world landmarks were preferred for feature construction because they were more suitable for scale-invariant 3D hand geometry. Image landmarks remained available for display and screen-space movement.

### 5.5 Hand-label handling

MediaPipe handedness labels were used to keep separate history for left and right hands. The first static version could process up to two hands, although the future sequence system was later identified as needing an explicit active-hand policy.

---

## 6. Phase Four: Geometric Feature Engineering

### 6.1 Why geometry was used

The classifier needed numbers rather than raw image pixels. Hand landmarks were therefore converted into geometric measurements.

This made the first classifier easier to understand and reduced the amount of data needed compared with training directly on images.

### 6.2 Joint vectors

For each joint, two bone vectors were constructed from landmark coordinates.

If points $A$ and $B$ define a bone, the vector from $A$ to $B$ was:

$$
\vec{v} = B - A
$$

### 6.3 Joint angles

The angle between two vectors was computed using the dot product:

$$
\vec{v} \cdot \vec{w} = |\vec{v}| |\vec{w}| \cos(\theta)
$$

Therefore:

$$
\theta = \cos^{-1}\left(\frac{\vec{v} \cdot \vec{w}}{|\vec{v}| |\vec{w}|}\right)
$$

The implementation clamped the cosine value before calling `acos` so small floating-point errors could not produce an invalid input.

### 6.4 Fifteen joint angles

The project computed 15 joint flexion angles:

- Three for the thumb
- Three for the index finger
- Three for the middle finger
- Three for the ring finger
- Three for the little finger

A relatively straight finger produced an angle closer to 180 degrees, while a curled joint produced a smaller angle.

### 6.5 Additional features

The joint angles alone were not enough to describe all handshapes. Additional features were added:

1. Thumb abduction angle
2. Five fingertip-to-wrist distances
3. Ring-pinky spread angle

The fingertip distances were normalized by wrist-to-middle-MCP hand scale:

$$
\text{normalized distance} =
\frac{\text{tip-to-wrist distance}}
{\text{wrist-to-middle-MCP distance}}
$$

### 6.6 Final feature vector

The final vector had 22 dimensions:

```text
[0:15]   joint angles
[15]     thumb abduction
[16:20]  normalized fingertip distances
[21]     ring-pinky spread
```

The exact order was kept consistent across:

- Calibration
- CSV storage
- Training
- ONNX export
- Runtime inference

This consistency was one of the most important data-contract requirements in the project.

---

## 7. Phase Five: Rule-Based Static Classification

### 7.1 Reason for starting with rules

A rule-based classifier provided an immediate baseline before enough training data existed. It also gave the student a direct connection between hand geometry and letter decisions.

### 7.2 Rule structure

Each letter rule specified ranges for selected features. A candidate handshape matched a rule when its required measurements fell within the configured ranges.

The classifier returned:

```python
(label, confidence)
```

or `None` when no rule was sufficiently convincing.

### 7.3 Confidence scoring

The rule score represented the fraction of constraints satisfied. A tolerance band allowed partial credit near a boundary, while measurements clearly outside a required range caused rejection.

A confidence threshold was used to avoid displaying weak guesses.

### 7.4 Rule-based limitations

The rules worked best for geometrically distinctive shapes such as:

- A
- B
- C
- D
- E
- I
- L
- O
- W
- Y

Many other letters had subtle distinctions that were difficult to encode reliably with hand-written ranges. The rule system was therefore retained as a fallback and educational baseline rather than treated as the final classifier for every static letter.

---

## 8. Phase Six: Static Data Collection

### 8.1 Calibration tool

`calibrate.py` was created to collect 22-dimensional feature vectors for a selected letter.

The tool:

1. Opened the camera.
2. Ran MediaPipe hand detection.
3. Selected a detected hand.
4. Computed joint angles and features.
5. Wrote rows to `data/<LETTER>.csv`.
6. Printed observed minimum and maximum feature ranges.

### 8.2 Capture controls

The calibration workflow supported:

- `SPACE` for one manual sample
- `A` to toggle automatic capture
- `Q` to quit and print a summary

### 8.3 Data requirements

The target was approximately 500 samples per static letter. Variation was intentionally included through:

- Slight hand rotation
- Small translation
- Natural changes in pressure
- Minor changes in finger placement
- Different distances from the camera

The purpose was not to collect 500 identical frames. The purpose was to sample the range of valid handshapes.

### 8.4 Data validation

Each row was checked to ensure it contained exactly 22 features. This protected the training pipeline from malformed or inconsistent samples.

### 8.5 Static data status reached

The project ultimately collected and validated data for 20 static letters:

```text
A, B, C, D, E, F, G, I, K, L, M, N, O, R, S, T, U, V, W, Y
```

J and Z were intentionally excluded because their identity depends on movement.

---

## 9. Phase Seven: Supervised Static Model Training

### 9.1 Why machine learning was added

The rule-based system could not reliably encode every subtle static handshape. A supervised model was therefore trained from the collected CSV examples.

Each CSV row became one training example:

```text
X = 22-dimensional feature vector
y = letter label from the source CSV filename
```

### 9.2 Training pipeline

The training process:

1. Loaded all available letter CSV files.
2. Combined rows into a feature matrix.
3. Assigned labels from the corresponding filenames.
4. Split data into training and validation portions.
5. Standardized the numeric features.
6. Trained a classifier.
7. Evaluated predictions.
8. Saved metrics and label ordering.
9. Exported the runtime model.

### 9.3 Model artifacts

The training phase produced:

```text
models/sign_classifier.onnx
models/sign_classifier.joblib
models/metrics.json
models/sign_classifier_labels.json
```

The ONNX model was used for runtime inference. The joblib artifact preserved a Python-side fallback and training representation.

### 9.4 Runtime loading

`sign_classifier.py` loaded the ONNX session through ONNX Runtime. It also loaded the saved label order so numeric model outputs could be mapped back to letter names.

The runtime behavior was:

```text
ONNX model available -> use ML classifier
otherwise            -> use rule-based classifier
```

### 9.5 Model validation

The model artifacts were checked for:

- Existence
- Correct label coverage
- Compatibility with the 22-feature input shape
- Successful ONNX Runtime loading
- Successful live inference

---

## 10. Phase Eight: Live Static Recognition

### 10.1 Live loop

The final static runtime loop followed this structure:

```text
read frame
    -> recover from transient camera failure
    -> flip frame for mirror behavior
    -> create MediaPipe image
    -> detect hand landmarks
    -> select world or image landmarks
    -> compute angles
    -> build 22-feature vector
    -> classify with ONNX or rules
    -> update voting history
    -> draw hand and result
```

### 10.2 Frame voting

A 14-frame deque was maintained for each hand. The displayed letter was updated only when a label obtained a majority of the available votes.

This reduced flicker caused by occasional uncertain frames.

### 10.3 Important limitation discovered

The voting buffer stabilized a displayed letter but did not segment a continuous sequence. It answered:

> What letter has been most common recently?

It did not answer:

> When should this letter be emitted once and replaced by the next letter?

This limitation defined the next architectural stage.

---

## 11. Phase Nine: Documentation and Reproducibility

The project setup, camera behavior, model workflow, and feature format were documented so the environment could be reproduced.

Documentation covered:

- Python 3.11 setup
- Local virtual environment location
- Camera preference and fallback behavior
- Frame-read retry logic
- Calibration commands
- Feature ordering
- Model training and export
- Static recognition limitations
- Planned sequence-recognition work

The documentation was later consolidated into a concise README and a detailed implementation plan to reduce duplicate Markdown files.

The documentation cleanup removed redundant guides while preserving:

- Setup instructions
- Student-oriented theory
- File responsibilities
- Sequence-recognition architecture
- Token prediction planning

---

## 12. Current Completed State

By the current moment, the following work had been completed:

- [x] Python 3.11 environment established.
- [x] Virtual environment moved outside iCloud Drive.
- [x] Required computer-vision and ML libraries installed.
- [x] Camera startup failure investigated.
- [x] Stable camera preference implemented.
- [x] Transient frame-read retry behavior implemented.
- [x] MediaPipe hand landmark detection integrated.
- [x] Hand skeleton and landmark visualization implemented.
- [x] Fifteen joint angles implemented.
- [x] Thumb, distance, and spread features implemented.
- [x] Consistent 22-dimensional feature contract established.
- [x] Rule-based static classifier implemented.
- [x] Calibration tool implemented.
- [x] Twenty static letter datasets collected.
- [x] Static model trained and exported.
- [x] ONNX runtime inference integrated.
- [x] Rule-based fallback preserved.
- [x] Frame-level voting implemented.
- [x] Setup and architecture documented.

The current application therefore represents a completed static-letter baseline, not a completed word recognizer.

---

## 13. The Exact Boundary at the Current Moment

The project had not yet implemented the following:

- A temporal observation object
- Palm velocity tracking
- Speed smoothing for segmentation
- A letter-boundary state machine
- A word-boundary state machine
- A completed-letter event model
- A word buffer
- J trajectory data
- Z trajectory data
- J/Z template matching
- Offline landmark replay
- Prefix dictionary suggestions
- N-gram prediction
- Sentence prediction
- Local LLM integration

These items were not treated as completed merely because they had been planned or documented.

The next implementation phase therefore started at:

```text
existing static frame predictions
    -> temporal history
    -> boundary detection
    -> completed letter events
```

---

## 14. Next Phase: Recommendation 6

The next phase was Recommendation 6:

```text
rule-based segmentation + existing ML classification
```

The planned sequence was:

```text
MediaPipe observations
    -> temporal features
    -> boundary state machine
    -> static or motion classifier
    -> completed letter event
    -> word buffer
    -> optional token predictor
```

### 14.1 Why no local LLM was required

The existing ONNX model already classified static handshapes. The missing problem was temporal segmentation, not text generation.

A local LLM would have been premature because it could suggest plausible words while the underlying recognizer was still duplicating or missing letters.

### 14.2 First next-step objective

The first next-step objective was to produce one event for each completed segment:

```text
LETTER(A)
LETTER(B)
LETTER(J)
WORD_END
```

Only after that event stream became reliable would word and sentence prediction be added.

---

## 15. Planned Temporal Data Model

A temporal observation was to wrap the existing static result with movement information:

```python
FrameObservation(
    timestamp,
    hand_label,
    palm_x,
    palm_y,
    palm_z,
    velocity_x,
    velocity_y,
    velocity_z,
    speed,
    normalized_speed,
    feature_vector,
    static_prediction,
    static_confidence,
)
```

The 22-dimensional feature vector was not to be replaced. Temporal features were to be added around it.

A completed segment was to produce:

```python
RecognizedLetter(
    label="A",
    confidence=0.89,
    start_time=...,
    end_time=...,
    segment_type="static",
)
```

This event model was necessary to prevent a stable frame prediction from becoming repeated letters.

---

## 16. Planned Boundary Detection

### 16.1 Movement calculation

If palm position at time $t$ was $p_t$, velocity was to be calculated as:

$$
 v_t = \frac{p_t - p_{t-1}}{t_t - t_{t-1}}
$$

Speed was the magnitude:

$$
 s_t = \|v_t\|
$$

Speed was to be normalized by hand scale so thresholds were less sensitive to camera distance:

$$
 s_t^* = \frac{s_t}{\text{hand scale}}
$$

### 16.2 Planned state machine

The boundary detector was to use:

```text
NO_HAND
WAITING_FOR_LETTER
COLLECTING_SEGMENT
STABLE_LETTER
POSSIBLE_BOUNDARY
```

A low-speed period was to create a candidate boundary, but the boundary was to be confirmed only after:

- The segment had reached a minimum duration.
- The low-speed period lasted long enough.
- The hand did not resume the same segment.

### 16.3 Initial timing targets

The first experimental targets were:

| Event | Approximate duration |
|---|---:|
| Minimum candidate segment | 10 frames or equivalent time |
| Letter pause | 0.25 to 0.60 seconds |
| Word pause | 0.80 to 1.50 seconds |
| Speed smoothing window | 5 frames |
| Hand-loss timeout | Configured and measured experimentally |

These were to be treated as tuning values, not universal constants.

---

## 17. Planned Static Segment Classification

When a static segment was finalized, it was to be classified using the existing model:

1. Select valid observations from the segment.
2. Remove or ignore transition frames where appropriate.
3. Run the ONNX classifier on the 22-dimensional vectors.
4. Reject predictions below the confidence threshold.
5. Aggregate the remaining predictions.
6. Emit one letter event.

The sequence module was not to duplicate feature construction or ONNX parsing. It was to call the existing functions:

- `build_feature_vector()`
- `classify_static_ml()`
- `classify_static()` as fallback

---

## 18. Planned J and Z Recognition

### 18.1 Why J and Z needed a separate method

J and Z were motion letters. Their identity was contained in the ordered path of the hand or fingertip, not merely in one handshape frame.

They therefore needed ordered trajectory samples rather than static CSV rows.

### 18.2 First planned solution

The first motion classifier was to use normalized trajectory templates:

1. Collect multiple J and Z demonstrations.
2. Translate each path so its first point was at the origin.
3. Normalize by hand scale or trajectory extent.
4. Smooth the path.
5. Resample paths to a common length.
6. Compare live paths against stored templates.

For template $T$ and observed path $X$:

$$
D(T, X) = \frac{1}{n}\sum_{i=1}^{n} \|T_i - X_i\|
$$

The lowest-distance template was to be selected only when it passed a rejection threshold.

### 18.3 Planned motion-data structure

```text
motion_data/
    J/
        sample_001.csv
    Z/
        sample_001.csv
```

Each row was planned to preserve order and include:

```text
timestamp,palm_x,palm_y,palm_z,index_x,index_y,index_z,hand_scale
```

A learned trajectory model was reserved for a later stage if template matching proved too signer-specific.

---

## 19. Planned Word Buffer

Completed letter events were to be accumulated as a current word:

```text
A -> A
P -> AP
P -> APP
L -> APPL
E -> APPLE
```

The word was to be committed by:

- A confirmed long pause
- Hand disappearance after a timeout
- The `Space` key during testing
- A future explicit gesture

Repeated letters were to be supported correctly. Deduplication was only to prevent repeated finalization of one segment; it was not to remove legitimate repeated letters such as `LL`.

---

## 20. Planned Token Prediction

### 20.1 Token definition

A token was to be treated as a discrete unit that could be stored and predicted:

- Letter token: `A`
- Word token: `APPLE`
- Boundary token: `<WORD_END>`
- Future punctuation token

The recognizer was to emit structured tokens rather than passing only display strings.

### 20.2 Prefix suggestions

The first prediction layer was to use a local word list. For the current raw prefix:

```text
HEL
```

it could return:

```text
HELLO
HELP
HELM
```

This required no LLM and no additional machine-learning framework.

### 20.3 N-gram suggestions

After committed words became reliable, a simple n-gram model could estimate:

$$
P(w_t \mid w_{t-1})
$$

for a bigram, or:

$$
P(w_t \mid w_{t-2}, w_{t-1})
$$

for a trigram.

A permitted SASL-relevant text corpus would have been preferred. Generic English text would not automatically represent SASL vocabulary or grammar.

### 20.4 Local LLM as optional escalation

A local LLM would only have been considered after:

- Letter-event accuracy was measured.
- Word boundaries were measured.
- Raw word accuracy was known.
- A vocabulary and corpus had been selected.
- The UI separated recognition from generated suggestions.
- Latency and memory were acceptable.

The model would not have been allowed to silently rewrite raw recognition.

---

## 21. Planned Testing Strategy for the Next Phase

The sequence work was to be tested in layers.

### 21.1 Unit tests

Synthetic tests were to cover:

- Stationary palm
- Stationary palm with jitter
- Constant movement
- Curved movement
- Invalid timestamps
- Speed smoothing
- Short pauses
- Long pauses
- Hand disappearance
- Minimum segment rejection
- Duplicate finalization
- Repeated letters
- Empty word commits
- Motion rejection

### 21.2 Offline replay

Landmark observations were to be recorded and replayed without opening the camera. This would allow threshold changes to be compared on identical input.

### 21.3 Live test order

The planned live order was:

1. Hold one static letter.
2. Move between two static letters.
3. Test repeated letters.
4. Add deliberate letter pauses.
5. Test J.
6. Test Z.
7. Test a short fingerspelled word.
8. Test word boundaries.
9. Test multiple words.
10. Reduce pauses and measure performance.
11. Add prefix suggestions.
12. Add next-word suggestions only after raw recognition was stable.

### 21.4 Metrics

The following metrics were to be kept separate:

- Frame-level static accuracy
- Completed-letter accuracy
- Motion-letter accuracy
- Boundary precision
- Boundary recall
- Duplicate-letter rate
- Missed-letter rate
- False-letter rate
- Word accuracy
- End-to-end latency
- Suggestion top-1 accuracy
- Suggestion top-3 accuracy

High frame accuracy alone would not prove that word recognition worked.

---

## 22. Final Project State at the Planning Boundary

This section is a historical snapshot, not the live repository status. At the
time this document was written, motion capture, motion-template export, and
the standalone webcam motion demo had not yet been created.

The project had successfully reached the static recognition milestone:

```text
camera
    -> MediaPipe landmarks
    -> 22-dimensional handshape features
    -> rule-based or ONNX static classifier
    -> frame-smoothed displayed letter
```

The project had not yet crossed into sequence recognition:

```text
static frame predictions
    -> temporal segmentation
    -> completed letters
    -> words
    -> tokens
    -> sentences
```

That was the correct point at which to begin the next implementation phase.

The historical implementation plan therefore ended with this transition:

```text
Completed past work:
    reliable static SASL handshape recognition

Next work not yet started:
    temporal boundaries, J/Z motion, words, and token prediction
```

The system was not to be described as a word or sentence recognizer until those temporal stages had been implemented and evaluated.

### Current status after this historical boundary

Since this plan was written, the repository has added:

- temporal boundary and sequence modules
- ordered motion capture in `motion_calibrate.py`
- normalized motion-template export in `train_motion_templates.py`
- a live standalone motion demo in `motion_demo.py`

The main recognizer still needs live integration of motion templates, broader
H/J/P/Q/Z sample validation, automatic word boundaries, and token suggestions.
