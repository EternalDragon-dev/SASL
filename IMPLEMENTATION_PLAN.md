# SASL Sequence Recognition Implementation Plan

This document is the implementation specification for extending the current SASL handshape recognizer from isolated letters to motion letters, fingerspelled words, and eventually token-based word and sentence suggestions.

It is written as both an engineering plan and a study guide. The central lesson is to separate four problems that are related but not identical:

1. **Detection**: where is the hand and what landmarks does it have?
2. **Recognition**: which static handshape or motion pattern is being shown?
3. **Segmentation**: when did one letter or word end?
4. **Prediction**: given recognized letters or words, what is likely to come next?

The first implementation should solve recognition and segmentation before adding language prediction.

---

## 1. Current Baseline

The repository currently has a working static-handshape pipeline:

```text
webcam frame
    -> MediaPipe HandLandmarker
    -> 21 hand landmarks
    -> 15 joint angles + 7 additional features
    -> static classifier
    -> frame voting buffer
    -> displayed letter
```

Important existing components:

| Component | Responsibility |
|---|---|
| `main.py` | Camera, MediaPipe, drawing, live application loop |
| `joint_angles.py` | Computes 15 joint flexion angles from landmarks |
| `sign_classifier.py` | Builds the 22-feature vector and classifies static handshapes |
| `calibrate.py` | Captures static handshape samples into `data/<LETTER>.csv` |
| `train_and_export.py` | Trains and exports the static classifier |
| `models/sign_classifier.onnx` | Runtime static classifier, when available |
| `models/hand_landmarker.task` | MediaPipe hand landmark model |
| `data/*.csv` | Static letter training examples |

The static feature vector is:

```text
[0:15]  15 joint flexion angles
[15]    thumb abduction angle
[16:20] five fingertip-to-wrist distances, normalized by hand scale
[21]    ring-pinky spread angle
```

The current static classifier covers the calibrated static letters. The motion subset is broader than a J/Z-only assumption: letters such as H, J, P, Q, and Z must be validated against trajectory data and must not be forced into the static classifier unless they are proven to be pose-based in the chosen SASL variant.

### Important distinction

The current 14-frame voting buffer prevents display flicker, but it does not yet create a completed letter event. Holding `A` for two seconds can produce many repeated frame predictions of `A`; a sequence system must emit exactly one completed `A` event for that held segment.

The new architecture therefore needs this distinction:

```text
raw frame prediction -> completed letter event -> word buffer -> token predictor
```

---

## 2. Goal and Non-Goals

### First milestone

Recognize a short fingerspelled word as a sequence of completed letter events, including a motion letter, without repeatedly emitting a letter while it is held.

A successful early demonstration might be:

```text
user signs A, B, pauses or transitions, J
system emits A -> B -> J
word buffer displays ABJ once
```

### In scope for the first implementation

- Temporal hand observation history
- Palm position and velocity calculation
- Smoothed movement signals
- Rule-based letter and word boundary detection
- Static segment classification using the existing ONNX model
- A motion-letter classifier for the dynamic subset (at minimum J/Z, and likely H/P/Q depending on calibration)
- Completed letter events
- Word buffering and explicit word commit
- Offline replay and unit testing
- Clear diagnostics for thresholds and transitions

### Not required initially

- A local LLM
- A Transformer
- LSTM/GRU or CTC training
- A new static handshape model
- Automatic sentence generation
- Full continuous sign-language translation
- A large external language corpus

The implementation should remain useful without a language model installed.

---

## 3. Do We Need a Local LLM?

No. A local LLM is not needed for Recommendation 6.

Recommendation 6 means:

```text
rule-based temporal segmentation + existing ML static classifier
```

The existing ONNX model answers:

> Which static handshape best matches this frame?

The new boundary detector answers:

> When should a group of frames be treated as one letter?

A future language model answers a different question:

> Given the letters or words already recognized, what is likely to come next?

Using an LLM before segmentation is reliable would make debugging harder. If the recognizer emits duplicate or missing letters, an LLM could hide the actual problem by autocorrecting the result.

### Recommended prediction progression

1. **No model**: emit recognized letters and words.
2. **Prefix dictionary**: use a local word list to suggest completions such as `HEL`, `HELLO`, and `HELP`.
3. **N-gram model**: predict likely next words from word history.
4. **Neural language model or local LLM**: only when sentence-level suggestions require broader context.

A prefix dictionary requires no additional package. A small n-gram model can be implemented with the Python standard library. A local LLM is optional future infrastructure, not a prerequisite for sequence recognition.

---

## 4. Dependencies

### Existing dependencies are sufficient for the MVP

- Python 3.11
- OpenCV
- MediaPipe
- NumPy
- ONNX Runtime
- Existing project modules

Use the local environment outside iCloud Drive:

```bash
source ~/venvs/sasl311/bin/activate
```

The project files may remain in iCloud Drive, but compiled Python packages should remain in the local virtual environment because native libraries stored in iCloud have previously caused loading failures.

### Optional dependencies

Do not install these until a concrete stage needs them:

| Package | Use | Needed for MVP? |
|---|---|---:|
| `scipy` | Savitzky-Golay filtering and signal processing | No |
| `rapidfuzz` | Fast fuzzy dictionary matching | No |
| `torch` | LSTM, GRU, CNN, or Transformer models | No |
| `sentencepiece` / `tokenizers` | Subword tokenization | No |
| Ollama or `llama.cpp` | Running a local LLM | No |

Start with moving-average smoothing and NumPy calculations. This keeps the first version inspectable and avoids dependency churn.

---

## 5. Target Architecture

```text
main.py
  camera + MediaPipe
        |
        v
sequence_classifier.py
  per-hand temporal coordinator
        |
        +--> temporal_features.py
        |      palm position, velocity, speed, path statistics
        |
        +--> boundary_detector.py
        |      state machine, letter boundary, word boundary
        |
        +--> sign_classifier.py
        |      existing static feature/classifier path
        |
        +--> motion_classifier.py
               motion-letter trajectory normalization and matching
         +--> motion_calibrate.py / train_motion_templates.py
             ordered capture and template artifact export
        |
        v
RecognizedLetter events
        |
        v
word_buffer.py or sequence buffer
        |
        +--> current letters and committed words
        |
        +--> optional token predictor
               prefix dictionary -> n-gram -> neural model/LLM later
```

The modules should be usable without a camera. A recorded sequence of landmark observations should be sufficient to test temporal logic.

### Proposed files

```text
temporal_features.py       # temporal observation and motion calculations
boundary_detector.py       # state machine and boundary decisions
motion_classifier.py       # J/Z trajectory templates
sequence_classifier.py     # coordinates segmentation and classification
word_buffer.py              # completed letters, words, and commit behavior
TOKEN_PREDICTIONS.md        # optional future data format, if needed later

tests/
    test_temporal_features.py
    test_boundary_detector.py
    test_motion_classifier.py
    test_sequence_classifier.py
    test_word_buffer.py
```

The first implementation may combine `word_buffer.py` into `sequence_classifier.py` if the file remains small. The boundaries between responsibilities should still remain explicit.

---

## 6. Data Model

### 6.1 Frame observation

Each tracked hand produces one observation per usable frame:

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
    feature_vector,
    static_prediction,
    static_confidence,
)
```

The existing 22-dimensional vector is retained. Temporal fields are added around it.

### 6.2 Recognized letter

A completed segment produces one event:

```python
RecognizedLetter(
    label="A",
    confidence=0.89,
    start_time=...,
    end_time=...,
    segment_type="static",
)
```

For motion letters:

```python
RecognizedLetter(
    label="J",
    confidence=0.82,
    start_time=...,
    end_time=...,
    segment_type="motion",
)
```

A completed event must be emitted once. It must not be emitted once per stable frame.

### 6.3 Word event

A word is committed when a word boundary is detected or when the user presses `Space` during testing:

```python
RecognizedWord(
    text="ABJ",
    confidence=0.81,
    start_time=...,
    end_time=...,
)
```

The confidence should be documented as an aggregate, such as the mean or geometric mean of letter confidences. Do not imply that it is a calibrated probability unless it has been evaluated as one.

---

## 7. Stage 1: Temporal Feature Extraction

### Theory

A single hand landmark frame describes a pose. Movement is a relationship between multiple frames. If $p_t$ is the palm position at time $t$, velocity is approximately:

$$
v_t = \frac{p_t - p_{t-1}}{t_t - t_{t-1}}
$$

Speed is the magnitude of velocity:

$$
s_t = \|v_t\|
$$

The system should use a normalized movement signal because raw pixel movement depends on camera distance and hand size. The existing hand scale, wrist-to-middle-MCP distance, is a suitable first normalization factor:

$$
s^{*}_t = \frac{s_t}{\text{hand scale}}
$$

### Required work

1. Add a helper to compute the palm center from the existing palm indices.
2. Retain the previous observation for each hand label.
3. Compute time delta using monotonic timestamps.
4. Reject zero or implausibly large time deltas.
5. Calculate normalized velocity and speed.
6. Store the last 5 speed values for a moving average.
7. Store palm positions and static predictions in a bounded deque.

### Initial constants

| Parameter | Starting value | Reason |
|---|---:|---|
| Speed smoothing window | 5 frames | Reduces landmark jitter |
| Minimum valid `dt` | Small positive value | Avoids division by zero |
| Segment minimum | 10 frames | Rejects accidental fragments |
| Segment maximum | 90 frames initially | Prevents an unbounded buffer |
| Static confidence | 0.60 to 0.70 | Aligns with current model thresholds |

These are experimental starting points. They must be exposed as configuration values and measured against real recordings.

### Cheap validation

Feed synthetic stationary points into the helper. The normalized speed should remain near zero even when small random noise is added.

---

## 8. Stage 2: Boundary State Machine

### Why a state machine?

A threshold alone cannot remember context. For example, a short pause may occur while forming one letter. A state machine remembers whether the system is waiting, collecting, or deciding.

Use these states:

```text
NO_HAND
WAITING_FOR_LETTER
COLLECTING_SEGMENT
STABLE_LETTER
POSSIBLE_BOUNDARY
```

### State definitions

#### `NO_HAND`

No hand is currently tracked.

- Clear temporary trajectory history.
- Preserve committed words.
- If a word is active, begin a hand-loss timer.
- Commit the word after the configured hand-loss duration.

#### `WAITING_FOR_LETTER`

A hand appeared, but there is insufficient history.

- Collect observations.
- Do not emit a letter yet.
- Transition to `COLLECTING_SEGMENT` after the first valid observation.

#### `COLLECTING_SEGMENT`

Frames belong to the current candidate letter.

- Append observations.
- Track normalized speed, position, path length, confidence, and predictions.
- Keep collecting through small pauses.

#### `STABLE_LETTER`

The handshape has remained sufficiently stable.

- Continue collecting enough context for classification.
- Avoid emitting repeatedly.
- A confirmed boundary can finalize the segment.

#### `POSSIBLE_BOUNDARY`

A boundary signal has appeared.

Confirm it only if the pause or transition persists and the segment is long enough. Otherwise return to collection.

### Boundary signals

Use multiple signals rather than one condition:

1. Smoothed normalized palm speed falls below a threshold.
2. The candidate segment is at least the minimum duration.
3. Static predictions have been stable, or a motion pattern has completed.
4. The hand changes position substantially before the next stable shape.
5. Confidence drops during a transition.
6. The hand disappears for long enough to commit a word.

A first implementation may use:

```text
candidate boundary = low smoothed speed for N frames
confirmed boundary = candidate boundary AND minimum segment length
```

Then add position change, confidence, and motion completion as the tests reveal failure cases.

### Letter and word pauses

These are different events:

| Event | Initial interpretation |
|---|---|
| 0.25 to 0.60 seconds | Possible letter boundary |
| 0.80 to 1.50 seconds | Word boundary candidate |
| Hand disappears | Commit current word after timeout |
| `Space` key | Manual word commit during testing |

At 30 FPS, 0.3 seconds is approximately 9 frames. The implementation should calculate frame counts from timestamps rather than assuming the camera always delivers exactly 30 FPS.

### Important limitation

Pause-based segmentation is excellent for the first controlled prototype but is not equivalent to natural continuous signing. A later system may need trajectory segmentation, sliding windows, HMMs, temporal CNNs, or LSTM/CTC. Those are alternatives, not part of the first Recommendation 6 implementation.

---

## 9. Stage 3: Segment Type Detection

When a segment is ready, calculate movement statistics:

```text
total_path_length
direct_displacement
maximum_speed
mean_speed
speed_variance
direction_changes
static_feature_variance
prediction_stability
```

Path length is important because a gesture can move in a curve and end near its starting position:

$$
\text{path length} = \sum_{t=1}^{n} \|p_t - p_{t-1}\|
$$

A simple normalized motion score is:

$$
M = \frac{\text{path length}}{\text{hand scale}}
$$

### Static segment indicators

- Low normalized path length
- Low speed variance
- Low feature-vector variance
- One stable static prediction
- Adequate static confidence

### Motion segment indicators

- Larger normalized path length
- Clear direction changes or curvature
- Higher fingertip or palm movement
- Static predictions are uncertain or inconsistent
- Segment duration is long enough for a trajectory

Do not classify a segment as motion merely because MediaPipe has jitter. Use a minimum motion score and smoothing.

---

## 10. Stage 4: Static Segment Classification

For static segments, reuse the existing classifier rather than training a new model.

1. Select observations from the stable portion of the segment.
2. Run the ONNX classifier on each selected 22-feature vector.
3. Discard predictions below the configured confidence threshold.
4. Aggregate by weighted vote.
5. Emit one `RecognizedLetter` event.

Example:

```text
A 0.91
A 0.88
A 0.90
A 0.86
A 0.89

final: A, aggregate confidence 0.89
```

The rule-based path remains the fallback when ONNX is unavailable.

The existing functions remain the owning abstraction:

- `build_feature_vector()` creates the input.
- `classify_static_ml()` runs ONNX inference.
- `classify_static()` provides the rule-based fallback.

The sequence layer should call these functions, not duplicate their feature logic.

---

## 11. Stage 5: Motion Classification for Dynamic Letters

A letter subset including J and Z, and possibly H, P, and Q depending on calibration, cannot be reliably recognized from a single static vector. Their identity is in the trajectory.

The first implementation must treat this as a motion-letter calibration problem, not a J/Z-only assumption. The exact dynamic subset should be determined from live data, but the project should be structured to support a motion classifier for the dynamic letters rather than attempting to force them into the static pipeline.

### First method: template matching

Collect multiple demonstrations for each motion letter. Each trajectory should contain palm and, preferably, index-fingertip positions.

Normalize each template:

1. Translate the first point to the origin.
2. Scale by hand size or total trajectory extent.
3. Resample to the same number of points.
4. Optionally smooth the points.
5. Store the normalized trajectory and label.

For a live trajectory $X$ and template $T$, compare corresponding points:

$$
D(T, X) = \frac{1}{n}\sum_{i=1}^{n} \|T_i - X_i\|
$$

The lowest distance wins if it is below a rejection threshold.

### Motion data collection

Add a dedicated collection mode rather than putting motion sequences into the existing static CSV format. The project now provides `motion_calibrate.py`; a motion sample uses ordered frames, not an unordered collection of independent rows.

Suggested format:

```text
motion_data/
    J/
        sample_001.csv
        sample_002.csv
    Z/
        sample_001.csv
```

Each row should include:

```text
timestamp,palm_x,palm_y,palm_z,index_x,index_y,index_z,hand_scale
```

The current recorder writes timestamped palm and index trajectories with hand
scale and frame order. The current template utilities normalize and compare
those trajectories offline. Live sequence integration and real sample
collection remain later milestones.

Collect at least 30 to 50 examples per motion letter initially, across:

- Different speeds
- Slightly different starting positions
- Different hand distances from the camera
- Natural variation in trajectory size
- Both successful and rejected examples for threshold tuning

### Future method: engineered motion classifier

If templates are too sensitive to personal style, calculate trajectory features and train a small scikit-learn model. Candidate features include:

- Start/end displacement
- Horizontal and vertical displacement
- Path length
- Maximum speed
- Number of direction changes
- Curvature
- Relative fingertip-to-palm movement

A learned motion model is a later replacement, not a requirement for the first prototype.

---

## 12. Stage 6: Completed Letter and Word Buffers

### Letter buffer rules

- Append only completed `RecognizedLetter` events.
- Do not append raw frame labels.
- Prevent duplicate emission for the same segment with a segment ID or state transition.
- Preserve uncertain segments for diagnostics, but do not add them to the visible word unless they pass the threshold.

### Word buffer rules

- Add each completed letter to the current word.
- Display the current word in real time.
- Commit on a confirmed word boundary, hand-loss timeout, or test `Space` key.
- Clear the current word after committing.
- Preserve committed words as sentence tokens.

Example:

```text
letter event A -> current word: A
letter event P -> current word: AP
letter event P -> current word: APP
letter event L -> current word: APPL
letter event E -> current word: APPLE
word boundary -> committed words: [APPLE]
```

The system must not use dictionary correction to silently replace the raw result during this stage. Keep raw recognition and suggestions visibly separate.

---

## 13. Stage 7: Token Methodology

### What is a token?

A token is a discrete unit the language layer can count and predict. Depending on the stage, a token can be:

- A letter: `A`, `B`, `J`
- A word: `HELLO`, `WORLD`
- A boundary marker: `<WORD_END>`
- A punctuation token
- A subword unit in an advanced model

The recognizer should emit structured tokens rather than raw display strings:

```text
LETTER(A)
LETTER(P)
LETTER(P)
LETTER(L)
LETTER(E)
WORD_END
```

This makes later prediction independent of the camera and segmentation implementation.

### Stage 7A: Prefix dictionary

The first prediction layer needs only a local vocabulary:

```text
HELLO
HELP
HELM
WORLD
```

Given the current letter prefix `HEL`, return matching words sorted by frequency or a simple configured order.

Requirements:

- A plain UTF-8 word list
- Case normalization
- Minimum prefix length before showing suggestions
- A maximum suggestion count, such as three
- An out-of-vocabulary path that preserves the raw letters

This is not a language model. It is a deterministic lookup and is the best first test of token integration.

### Stage 7B: N-gram prediction

An n-gram model predicts the next token using recent history:

$$
P(w_t \mid w_{t-1}, w_{t-2}, \ldots)
$$

A bigram estimates:

$$
P(w_t \mid w_{t-1})
$$

A trigram estimates:

$$
P(w_t \mid w_{t-2}, w_{t-1})
$$

The model counts adjacent words in a text corpus and applies smoothing for unseen combinations. It is fast and explainable, but it cannot understand long-distance context well.

Initial data target:

- 1,000 sentences: proof of concept
- 10,000 or more sentences: more useful baseline
- SASL-specific transcriptions are preferable to generic English text

### Stage 7C: Neural language model or local LLM

Only consider this after the recognizer can reliably produce words. A neural model can predict across longer contexts; a local LLM can produce richer sentence suggestions. Both require additional model management, memory, latency, and evaluation.

The interface should remain:

```python
predictions = language_model.suggest(
    committed_words,
    current_prefix=current_word,
    limit=3,
)
```

The recognizer must continue operating when `language_model` is absent.

---

## 14. Detailed Implementation Sequence

### Milestone 0: Freeze and verify the baseline

1. Confirm the current static app runs from `~/venvs/sasl311`.
2. Confirm the ONNX model loads.
3. Confirm a known static letter produces a stable display.
4. Record the current static behavior before changing `main.py`.
5. Do not modify the existing feature ordering.

Acceptance: static recognition still works after every temporal change.

### Milestone 1: Add temporal utilities

1. Create `temporal_features.py`.
2. Define an observation data structure.
3. Add palm-center extraction in normalized coordinates.
4. Add velocity, speed, scale normalization, and path-length helpers.
5. Add moving-average smoothing.
6. Write synthetic unit tests.

Acceptance: stationary, linear, noisy, and curved synthetic trajectories produce expected measurements.

### Milestone 2: Add the boundary detector

1. Create `boundary_detector.py`.
2. Define the state enum.
3. Define configurable thresholds.
4. Implement no-hand handling.
5. Implement segment accumulation.
6. Implement low-speed candidate boundaries.
7. Implement minimum and maximum segment durations.
8. Implement letter versus word pause durations.
9. Return boundary events, not labels.
10. Write state transition tests.

Acceptance: a synthetic held letter produces one segment finalization, not repeated finalizations.

### Milestone 3: Connect static classification

1. Create `sequence_classifier.py`.
2. Pass existing static predictions into the segmenter.
3. Finalize static segments with weighted voting.
4. Emit `RecognizedLetter` objects.
5. Add event deduplication.
6. Replace or supplement the current raw display voting in `main.py`.
7. Keep the existing visual overlay during transition.

Acceptance: signing or replaying `A` followed by `B` produces exactly `A`, `B` events.

### Milestone 4: Add word buffering

1. Create `word_buffer.py` or an equivalent focused class.
2. Consume only completed letter events.
3. Display current letters.
4. Commit on a word boundary.
5. Add `Space` as a temporary manual commit control.
6. Display committed words separately from the current word.
7. Add tests for empty words, repeated letters, hand loss, and manual commit.

Acceptance: `A`, `P`, `P`, `L`, `E`, followed by a boundary, produces one committed `APPLE`.

### Milestone 5: Collect and classify motion letters

1. Extend or add a motion collection tool.
2. Store ordered trajectory files for the validated dynamic subset, beginning with H, J, P, Q, and Z.
3. Normalize and resample trajectories.
4. Implement template distance.
5. Add a rejection threshold.
6. Test each collected dynamic letter offline first.
7. Connect motion classification to segment finalization.
8. Emit `segment_type="motion"` events.

Acceptance: held static letters still route to the static classifier, while recorded dynamic trajectories route to the motion classifier.

### Milestone 6: Add prefix suggestions

1. Add a small local vocabulary file.
2. Implement prefix lookup without external packages.
3. Show suggestions separately from recognized text.
4. Preserve the raw word when there are no matches.
5. Add tests for case, empty prefixes, unknown prefixes, and suggestion limits.

Acceptance: `HEL` returns matching suggestions without changing the recognized letters.

### Milestone 7: Add word-sequence prediction

1. Decide on a text corpus and licensing/permission for its use.
2. Normalize sentences into word tokens.
3. Add start and end boundary markers.
4. Count bigram transitions.
5. Add smoothing.
6. Expose top-k predictions with scores.
7. Evaluate on held-out sentences.
8. Integrate behind an optional language-model interface.

Acceptance: committed word history returns deterministic, reproducible suggestions and the app still runs with the model disabled.

### Milestone 8: Evaluate natural continuous signing

1. Measure errors with deliberate pauses.
2. Reduce pauses gradually.
3. Test transitions that contain temporary stops.
4. Compare duplicate, missed, and false boundaries.
5. Tune thresholds from recorded data, not memory.
6. Decide whether Recommendation 6 is sufficient.
7. Only then consider sliding windows, HMMs, temporal CNNs, or LSTM/CTC.

---

## 15. Testing Strategy

### Unit tests

Use synthetic data for deterministic tests:

- Stationary palm
- Stationary palm with jitter
- Constant horizontal movement
- Curved movement returning near its starting point
- Low-speed pause
- Pause shorter than the boundary threshold
- Hand disappearance
- Segment shorter than the minimum
- Segment longer than the maximum
- Duplicate finalization attempt

### Offline replay tests

Record real landmark observations and replay them without opening a camera. This makes threshold changes comparable because every version sees the same input.

An offline record should include at least:

```text
timestamp,hand_label,landmarks or derived palm position,feature vector,ground truth event
```

Do not depend on a live webcam for regression tests.

### Live test order

1. Hold one static letter.
2. Move from one static letter to another.
3. Fingerspell two letters.
4. Fingerspell repeated letters such as `LL` or `PP`.
5. Add deliberate pauses.
6. Test H, J, P, Q, and Z as motion candidates.
7. Test a short word.
8. Test a longer word.
9. Test word boundaries.
10. Test without pauses between every letter.

### Metrics

Track these separately:

- Static letter accuracy
- Motion letter accuracy
- Boundary precision
- Boundary recall
- Duplicate-letter rate
- Missed-letter rate
- False-letter rate
- Word accuracy
- End-to-end latency
- Suggestions accepted versus ignored

A classifier can have high frame accuracy and poor word accuracy if it emits duplicates. These metrics expose that difference.

---

## 16. Debugging and Observability

Display or log these values during development:

```text
hand label
current state
normalized speed
smoothed speed
segment frame count
static prediction and confidence
motion score
last emitted letter
current word
word-boundary timer
```

Add a debug mode rather than permanently crowding the normal UI. The debug output should make it possible to answer:

- Did MediaPipe lose the hand?
- Did the boundary detector see a pause?
- Was the segment too short?
- Did static classification reject the segment?
- Did motion classification reject the trajectory?
- Was an event emitted twice?
- Did the word buffer commit too early?

---

## 17. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Landmark jitter appears as motion | Smooth speed and use a minimum motion score |
| One held letter emits repeatedly | Emit only on segment finalization and track segment IDs |
| Pause inside a sign creates a false boundary | Require a minimum pause duration and segment context |
| User speed varies | Normalize by timestamps and expose thresholds |
| Camera FPS is inconsistent | Calculate durations from timestamps, not frame counts alone |
| J/Z vary by signer | Collect multiple templates and normalize trajectories |
| Static classifier rejects useful frames | Aggregate several frames and preserve confidence diagnostics |
| Dictionary silently changes output | Keep raw recognition separate from suggestions |
| Language model masks recognition defects | Add it only after raw word accuracy is measured |
| iCloud native-library failures return | Keep the Python environment outside iCloud |
| Two hands cause interleaved events | Start with one selected hand; add multi-hand coordination later |

### Two-hand policy for the first version

The current app can detect up to two hands. Recommendation 6 should initially select one active hand, preferably the hand with the highest detection confidence or the hand already being tracked. Do not interleave left and right events until there is an explicit two-hand grammar design.

---

## 18. Alternative Architectures and When to Escalate

Recommendation 6 is the first implementation because it reuses the current system and is easy to inspect. Other approaches remain valid when specific failure patterns appear.

| Approach | Use when |
|---|---|
| Continuous state machine | A simple real-time boundary controller is sufficient |
| Trajectory segmentation first | Turning points are more reliable than pauses |
| Sliding window | Fixed windows are more stable than event boundaries |
| LSTM/GRU with CTC | Rapid sequences need learned alignment and enough data exists |
| HMM | Probabilistic, interpretable sequence decoding is needed |
| Hybrid rule + ML | Current recommendation; rules segment and models classify |
| Overlapping-window voting | One window size is unreliable |
| Dictionary constraint | Vocabulary is known and error correction is useful |
| Two-pass processing | Segmentation and classification need independent evaluation |
| Temporal 1D CNN | Short-range motion patterns need learned recognition with low latency |

Do not escalate because an initial threshold needs tuning. Escalate when measured data shows that the architecture cannot meet the target behavior.

---

## 19. Definition of Done

The first Recommendation 6 implementation is complete when:

- [ ] Existing static recognition still works.
- [ ] Temporal observations are timestamped and normalized.
- [ ] Motion signals are smoothed.
- [ ] Boundary detection is implemented as a tested state machine.
- [ ] A held static letter emits one completed event.
- [ ] Static segments reuse the existing ONNX/rule classifier.
- [ ] The validated dynamic subset has ordered trajectory data and a first classifier.
- [ ] Motion segments route to the motion classifier.
- [ ] Letter events feed a word buffer.
- [ ] Word boundaries can be tested with `Space` and detected by timeout.
- [ ] Raw recognized text is separate from suggestions.
- [ ] Prefix suggestions work without an LLM.
- [ ] Offline replay tests exist.
- [ ] Duplicate, missed, false-boundary, and latency metrics are recorded.
- [ ] The app works when optional language-model components are absent.

---

## 20. Recommended First Coding Session

The first coding session should implement only the smallest independently testable slice:

1. Add `temporal_features.py`.
2. Add synthetic tests for stationary and moving palms.
3. Run those tests.
4. Add the boundary state machine.
5. Run state transition tests.
6. Only then connect it to `main.py`.

This order protects the working static recognizer and gives each new concept a falsifiable test before live-camera integration.

The first practical target is not an LLM or a sentence generator. It is a trustworthy event stream:

```text
LETTER(A), LETTER(B), LETTER(J), WORD_END
```

Once that stream is reliable, token prediction becomes a replaceable layer rather than a source of confusion.
