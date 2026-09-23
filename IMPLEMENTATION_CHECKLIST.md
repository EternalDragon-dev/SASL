# SASL Recognizer Implementation Checklist

This document is the working checklist for the project. It tracks what is already completed, what is currently working, and what still needs to be built before the recognizer can be considered a reliable full-alphabet system.

Status legend:
- ✅ Done
- 🔄 In progress
- ⏳ Planned / not started
- ⚠️ Needs revision

---

## 1. Foundation and environment

- [x] Diagnose startup crash and camera issue
- [x] Fix camera selection fallback strategy (prefer index 1 before 0)
- [x] Add warm-start webcam retry logic
- [x] Move Python environment outside iCloud Drive to avoid native-library loading issues
- [x] Establish working local Python 3.11 environment
- [x] Confirm project runs from local venv
- [x] Update project documentation and implementation notes
- [x] Reduce duplicate project docs and retain the main plans

---

## 2. Static recognition pipeline

- [x] MediaPipe hand model integration
- [x] OpenCV webcam loop and live display
- [x] Hand skeleton drawing and palm center calculation
- [x] Joint-angle extraction from landmarks
- [x] 22-feature feature-vector construction
- [x] Rule-based static classifier baseline
- [x] ONNX classifier loading and fallback logic
- [x] Real-time static letter display in the webcam app
- [x] Majority-vote smoothing to reduce flicker
- [x] Validation of static recognition for a working subset of letters

### Static letters currently treated as standard pose-based letters

This should be treated as an active calibration list, not a final fixed list:

- [ ] A
- [ ] B
- [ ] C
- [ ] D
- [ ] E
- [ ] F
- [ ] G
- [ ] H
- [ ] I
- [ ] K
- [ ] L
- [ ] M
- [ ] N
- [ ] O
- [ ] P
- [ ] Q
- [ ] R
- [ ] S
- [ ] T
- [ ] U
- [ ] V
- [ ] W
- [ ] X
- [ ] Y

Note: H, J, P, Q, and Z are not safe to assume as purely static letters without actual calibration and motion analysis.

---

## 3. Motion-letter handling

- [x] Recognize the need for motion-based letter handling
- [x] Add the concept of trajectory-aware motion classification into the plan
- [x] Add a dedicated ordered motion-trajectory capture tool
- [x] Add trajectory normalization and template-distance matching utilities
- [x] Add motion-template export and runtime loading
- [x] Add standalone live motion-template demo
- [ ] Collect motion samples for J
- [ ] Collect motion samples for Z
- [ ] Collect motion samples for H
- [ ] Collect motion samples for P
- [ ] Collect motion samples for Q
- [x] Build an offline motion-letter classifier that compares normalized trajectory paths
- [ ] Add a rejection threshold for uncertain motion matches
- [ ] Decide on a segment-type classifier: static vs motion before final label selection
- [ ] Confirm whether H/P/Q are static in the chosen SASL dialect or motion-dependent in this project

### Motion-letter status

- [ ] J: requires trajectory matching
- [ ] Z: requires trajectory matching
- [ ] H: needs calibration and decision whether it is static or motion-driven
- [ ] P: needs calibration and decision whether it is static or motion-driven
- [ ] Q: needs calibration and decision whether it is static or motion-driven

This is a major unresolved design point and must be tested against real signed examples rather than assumptions.

---

## 4. Temporal segmentation and sequence layer

- [x] Add Recommendation 6 temporal concepts to the project
- [x] Add initial sequence-classification module
- [x] Add boundary detector for low-speed pause and segment closure
- [x] Prevent duplicate repeated emissions of the same held letter
- [x] Add basic word accumulator buffer
- [x] Add manual space-triggered commit for testing
- [ ] Tune low-speed threshold against real hand motion
- [ ] Tune segment minimum and maximum durations against live examples
- [ ] Tune transition behavior between neighboring letters
- [ ] Confirm that a stable held letter emits exactly one event
- [ ] Confirm that a letter change emits the new label only once
- [ ] Add per-hand state tracking for left/right hand selection and switching

---

## 5. Word and sentence recognition

- [x] Add a basic current-word buffer
- [x] Add manual word commit command for testing
- [ ] Commit words automatically on pause or word boundary
- [ ] Add committed-word history display
- [ ] Add explicit sentence buffer
- [ ] Add prefix suggestions from a local vocabulary
- [ ] Add a small n-gram or token predictor
- [ ] Decide whether token-based language suggestions are required for the next milestone
- [ ] Keep raw recognition and suggestions visibly separated

---

## 6. Calibration and data collection

- [x] Create calibration tooling
- [x] Capture sample rows for training data
- [x] Create separate ordered motion-data format
- [ ] Collect a solid sample set per static letter
- [ ] Validate ranges and feature spread for each letter
- [ ] Collect motion samples for dynamic letters
- [ ] Store motion data in dedicated ordered sequence format
- [ ] Build a reproducible offline replay / evaluation dataset

### Recommended data targets

- [ ] At least 100–300 high-quality samples per static letter for initial tuning
- [ ] At least 30–50 motion sequences per dynamic letter
- [ ] Include variation in:
  - hand distance from camera
  - slight rotation
  - different speeds
  - different start/end positions
  - minor noise and partial occlusion

---

## 7. Testing and validation

- [x] Add sequence-layer regression tests for boundary and word logic
- [x] Add offline trajectory normalization and template tests
- [x] Confirm the temporal sequence tests pass
- [x] Try the standalone motion-template demo through the webcam
- [ ] Validate the app on a real-time webcam stream against actual hand motion
- [ ] Check repeated same-letter emission issue is eliminated in live conditions
- [ ] Validate static letter recognition across multiple people / hand sizes
- [ ] Validate J and Z motion detection on live samples
- [ ] Validate H/P/Q motion or static classification after calibration
- [ ] Record a before/after benchmark for each letter
- [ ] Test transition correctness across consecutive letters

---

## 8. Next milestone priorities

### Priority 1: full static calibration pass

- [ ] Collect data and validate A, B, C, D, E, F, G, I, K, L, M, N, O, R, S, T, U, V, W, X, Y
- [ ] Figure out which of H/P/Q are truly static or dynamic in the target SASL variant

### Priority 2: dynamic-letter calibration

- [ ] J
- [ ] Z
- [ ] H
- [ ] P
- [ ] Q

### Priority 3: segmentation and transition tuning

- [ ] Letter boundary timing
- [ ] Distinguish static motion from jitter
- [ ] Avoid false-positive repeated events

### Priority 4: word recognition and suggestions

- [ ] Word buffer and commit logic
- [ ] Prefix or dictionary suggestions
- [ ] Token-level representation for future language model work

---

## 9. Final goal

The final goal is not just “the app shows some letters.” The final goal is:

- [ ] reliably recognize the full intended alphabet
- [ ] handle dynamic motion letters correctly
- [ ] emit one completed event per letter
- [ ] build words from completed letters
- [ ] optionally suggest likely next words or tokens

This is the true completion bar for Recommendation 6 and the next translation stages.
