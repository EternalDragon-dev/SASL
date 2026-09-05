# Sequence Recognition for SASL: From Letters to Words and Sentences

## The Core Challenge

Recognizing individual SASL letters is the foundation, but real sign language communication requires:
- **Letter boundaries**: knowing when one letter ends and another begins
- **No enforced pauses**: users shouldn't have to freeze between letters
- **Word and sentence recognition**: understanding context to predict and correct
- **Scalability**: from letters to words to complete thoughts

This document outlines 10 architectural approaches and a token-based language model layer that can be combined.

---

## Part 1: Letter/Word Boundary Detection & Sequence Recognition

### 1. Continuous State Machine with Letter Boundaries

**Theory**: Think of this like a traffic light system that monitors hand behavior and decides when a transition occurs.

**How it works**:
- Maintain a state machine with three states: `IDLE`, `LETTER_ACTIVE`, `BOUNDARY_DETECTED`.
- Track three signals:
  1. **Palm velocity**: compute distance of palm center between frames; smooth over 5-frame window
  2. **Hand position change**: detect major repositioning (> 15% of frame width)
  3. **Classifier confidence**: track running average; drop below 50% = uncertainty zone

**Boundary detection triggers**:
```
IF velocity_smoothed < 0.02 px/frame AND frames_since_start > 10
   AND classifier_confidence > 0.7
THEN emit letter and reset state
```

**Requirements**:
- Velocity threshold tuning (0.01–0.05 px/frame depending on camera FPS and hand size)
- Minimum frame count before accepting a letter (10–20 frames = 0.3–0.6 seconds at 30 FPS)
- Confidence smoothing window (5–10 frames)

**Pros**:
- Simple to implement and debug
- Real-time
- Works with existing classifiers

**Cons**:
- Requires manual threshold tuning per user/camera
- Fails on natural hand pauses within a single letter
- No inherent way to handle rapid fingerspelling

**Data structures needed**:
```python
class LetterBuffer:
    frames_buffer: deque(maxlen=60)  # hold recent frames
    confidence_history: deque(maxlen=20)
    palm_positions: deque(maxlen=30)
    accumulated_letters: list
    state: Enum(IDLE, ACTIVE, BOUNDARY)
```

---

### 2. Trajectory Segmentation First, Then Classify

**Theory**: Instead of asking "what letter is this frame?", ask "what letter should this trajectory segment be?"

The idea: hand motion has natural structure. When you sign a letter, your hand follows a path. Between letters, that path changes direction or speed. Segment the path, then classify each segment.

**Algorithm**:
1. Record hand trajectory (palm position) over 30–60 frames.
2. Compute the speed profile: speed at each frame.
3. Identify **turning points**: where speed drops sharply or direction changes by >45°.
4. Split trajectory at turning points.
5. Classify each segment as a letter.

**Example**: Signing "AB"
```
A: hand moves down-right (specific angles)
[pause]
B: hand moves in a different pattern
```

The speed dip marks the boundary; you have two trajectories to classify independently.

**Requirements**:
- Minimum segment length: 5–10 frames
- Speed threshold for turning point: depends on user speed (requires calibration)
- Direction change threshold: 30–60°
- Smoothing: apply Savitzky–Golay or moving-average filter to speed profile to reduce noise

**Pros**:
- Theoretically sound for natural sign language
- Handles variable letter duration
- No need for confidence thresholds

**Cons**:
- Sensitive to noise in trajectory
- Requires tuning turning-point detection
- Fails if user moves continuously without pausing

**Mathematical foundation**:
```
speed[t] = distance(palm[t], palm[t-1]) / dt
direction[t] = atan2(palm[t].y - palm[t-1].y, palm[t].x - palm[t-1].x)
angle_change[t] = abs(direction[t] - direction[t-1])
turning_point = (speed[t] < 0.01) AND (angle_change[t] > 45°)
```

---

### 3. Sliding Window with Overlap

**Theory**: Imagine sliding a magnifying glass along the hand trajectory. At each position, you ask: "what letter is in this window?"

This is inspired by image recognition (sliding-window CNNs) but applied to time.

**Algorithm**:
1. Use a fixed-size sliding window: 30, 45, or 60 frames.
2. Slide it frame-by-frame (stride = 1).
3. For each window, classify the hand shape/trajectory.
4. Track which letter is consistently detected.
5. Letter boundary = when the highest-confidence letter changes.

**Example with 45-frame window**:
```
Frames 0–44:   confidence(A)=0.9, confidence(B)=0.1  → output "A"
Frames 1–45:   confidence(A)=0.85, confidence(B)=0.15 → output "A"
...
Frames 10–54:  confidence(A)=0.2, confidence(B)=0.9  → boundary! output "B"
```

**Requirements**:
- Window size: 30–60 frames (1–2 seconds at 30 FPS)
- Stride: 1 frame (dense overlap)
- Confidence threshold: 0.6–0.8 for locking a letter
- Stability check: letter must be consistent for 3–5 consecutive windows before emission

**Pros**:
- No manual threshold tuning
- Handles variable-speed signing
- Robust to single-frame noise

**Cons**:
- Computationally expensive (sliding window every frame)
- Latency: must wait for window to fill before output
- Redundant: overlapping windows compute similar features

**Implementation consideration**:
Use a ring buffer and only recompute the classifier on the new frame entering the window, rather than recomputing all 45 frames.

---

### 4. Recurrent Neural Network (LSTM/GRU with CTC Loss)

**Theory**: Teach a neural network to consume a *sequence* of hand observations and output a *sequence* of letters, without explicit boundary annotations.

**Background**: RNNs have hidden state that persists across frames. The network learns patterns like:
- "After seeing angles like A for 20 frames, expect a boundary"
- "Rapidly changing angles = motion letter"

CTC (Connectionist Temporal Classification) is a loss function that learns to align variable-length input sequences to variable-length output sequences *automatically*. You don't need to manually mark when letter boundaries are.

**Architecture**:
```
Input (each frame): [15 joint angles, palm_x, palm_y, palm_z, velocity_x, velocity_y] = 20D vector
↓
LSTM layer 1 (64 units)
↓
LSTM layer 2 (64 units)
↓
Dense layer: project to (# letters + blank) classes
↓
CTC loss: aligns output to ground-truth letter sequence
```

**Training**:
- Collect videos of users signing words/sentences.
- Manually annotate which frames correspond to which letters (no precise boundary needed).
- Train with CTC loss; the model learns boundaries.

**Inference**:
```
feed sequence of 100–300 frames →
LSTM outputs probability distribution over letters for each frame →
CTC decoder (beam search or greedy) → letter sequence
```

**Requirements**:
- LSTM hidden size: 64–128
- Dropout: 0.2–0.5 (prevent overfitting)
- Training data: 10–20 videos per word, varying speeds
- GPU recommended (LSTM is slow on CPU)
- Sequence length: typically 100–400 frames (3–13 seconds at 30 FPS)

**Pros**:
- State-of-the-art for sequence problems
- Learns boundaries automatically
- Handles variable speed and style

**Cons**:
- Requires significant training data and computational resources
- Black-box: hard to interpret what the model learned
- High latency (need full sequence before output)
- Overkill for static letters

---

### 5. Hidden Markov Model (HMM) per Letter

**Theory**: Each letter is a stochastic process. You observe hand angles, which are noisy observations of an underlying "true" letter state.

HMM is a probabilistic model with:
- **States**: e.g., for letter A: `state_0_start → state_1_middle → state_2_end`
- **Observations**: hand joint angles (what you measure)
- **Transitions**: probability of moving from state_i to state_j
- **Emissions**: probability of observing angles given a state

**How it works**:
1. Train one HMM per letter using collected data.
2. To recognize a sequence, run all HMMs in parallel.
3. Use **Viterbi algorithm** to find the most likely path through all HMMs (which letters, in what order, with what boundaries).

**Example**:
```
Observe angles [frame0, frame1, ..., frame30]
↓
Run all 20 HMM models in parallel
↓
Viterbi finds: "most likely = A (frames 0-10) → B (frames 11-30)"
```

**Requirements**:
- Number of hidden states per letter: 3–5 (typically)
- Training data per letter: 50+ sequences
- Bayesian parameter estimation (e.g., Baum-Welch algorithm)
- Smoothing: add pseudocounts to avoid zero probabilities

**Pros**:
- Probabilistic; interpretable
- Handles variable-length sequences
- Fast inference (Viterbi is efficient)
- Works with small datasets compared to neural networks

**Cons**:
- Assumes hand angles follow Markov property (current state depends only on previous state, not history) — often violated
- Requires careful state number tuning
- Sensitive to initialization

**Mathematical intuition**:
```
P(letters | observations) ∝ P(observations | letters) × P(letters)
Viterbi finds argmax_letters P(letters | observations)
```

---

### 6. Hybrid: Rule-Based Segmentation + ML Classification

**Theory**: Combine hand-crafted heuristics (which are robust) with learned models (which are powerful).

Use motion rules to find boundaries; use your trained static/motion classifiers to label each segment.

**Algorithm**:
1. Monitor palm velocity and position in real-time.
2. When velocity drops below threshold for 0.3 seconds → **candidate boundary**.
3. Extract the segment between boundaries (e.g., frames 50–100).
4. Classify this segment:
   - If segment is static (low motion variance) → use static classifier
   - If segment has motion → use trajectory matching or motion classifier
5. Output recognized letter.

**Requirements**:
- Velocity threshold: 0.01–0.05 px/frame (calibrate per user)
- Minimum segment duration: 0.2–0.5 seconds
- Static vs. motion decision: if `motion_variance < 0.1` → static, else → motion
- Classifier confidence threshold: 0.6+

**Pros**:
- Immediately testable (use existing classifiers)
- Interpretable (rules are explicit)
- Robust to unusual hand sizes/camera angles
- Low computational cost

**Cons**:
- Requires manual tuning
- Fails if user signs without pausing
- Velocity threshold varies across users

**Implementation snippet**:
```python
def segment_and_classify(frames_buffer, static_classifier, motion_classifier):
    if palm_velocity_smoothed < VELOCITY_THRESHOLD:
        segment = extract_segment_since_last_boundary()
        motion_variance = compute_variance(segment.palm_positions)
        
        if motion_variance < STATIC_THRESHOLD:
            letter = static_classifier.classify(segment)
        else:
            letter = motion_classifier.classify_trajectory(segment)
        
        return letter
```

---

### 7. Voting Across Overlapping Windows

**Theory**: Ask multiple "judges" (windows of different sizes) and use their consensus to find boundaries.

Instead of one window size, use 3–4 overlapping windows (30, 45, 60 frames). Each votes. Majority wins.

**Algorithm**:
1. Maintain three sliding windows: W30, W45, W60.
2. Classify each window independently.
3. At each frame, count votes:
   - If 2+ windows agree on letter L → strong signal
   - If windows disagree → uncertainty zone (likely boundary)
4. Emit letter only when consensus is strong (2+ agree for 3+ frames).

**Requirements**:
- Window sizes: stagger them (30, 45, 60) or (20, 40, 60)
- Voting threshold: 2 out of 3 agree
- Stability: require consensus for 3+ consecutive frames
- Tie-breaking: use confidence scores from classifiers

**Pros**:
- Robust to noise and single-window failures
- Handles variable letter speeds (long windows catch slow signers)
- No manual threshold tuning needed

**Cons**:
- 3x computation cost compared to single window
- Adds latency (must wait for largest window to fill)
- Complexity in tie-breaking

---

### 8. Dictionary Constraint

**Theory**: Constrain recognized sequences to a dictionary of known words. When in doubt, pick the word that's in the dictionary.

**How it works**:
1. Maintain a vocabulary: `["apple", "book", "cat", ..., "hello"]`
2. As letters are recognized, build a candidate word.
3. If candidate doesn't match any word prefix, backtrack or reject.
4. Use a language model (n-gram) to score partial sequences.

**Example**:
- Raw recognition: "A-P-L" (some frames were ambiguous)
- Dictionary options:
  - "A-P-L-E" (apple) ✓ valid word
  - "A-P-L" + boundary (unlikely; not a word)
- Decision: autocorrect to "apple"

**Requirements**:
- Vocabulary: 100–1000+ words (or domain-specific)
- Language model: bigram or trigram probabilities
  - P(B | A) = probability of letter B after letter A in your language
  - P(word_2 | word_1) = probability of word_2 following word_1
- Edit distance: to handle partial misrecognition

**Pros**:
- Powerful error correction
- Improved accuracy for constrained domains
- Users feel like system "understands" context

**Cons**:
- Requires pre-built vocabulary and language model
- Fails for out-of-vocabulary words (typos, new signs)
- Can silently correct errors (user doesn't know)
- Language models require good training data

**Example probability**:
```
P(word_sequence | observations) = 
  P(observations | word_sequence) × P(word_sequence)
  
where P(word_sequence) = P(word1) × P(word2|word1) × P(word3|word2|word1)
```

---

### 9. Two-Pass Approach

**Theory**: Separate the "where" problem (finding boundaries) from the "what" problem (classifying content).

**Pass 1: Segmentation**
- Use trajectory analysis, velocity thresholds, or motion heuristics.
- Output: list of candidate segments `[(start_frame, end_frame), ...]`

**Pass 2: Classification**
- For each segment, decide: static or motion?
- Use the appropriate classifier.
- Output: sequence of recognized letters.

**Algorithm**:
```
# Pass 1: Find boundaries
segments = find_motion_boundaries(palm_trajectory, velocity_threshold=0.02)
→ segments = [(0, 25), (26, 50), (51, 100)]

# Pass 2: Classify each
for (start, end) in segments:
    segment_frames = frames[start:end+1]
    motion_variance = compute_motion_variance(segment_frames)
    if motion_variance < THRESHOLD:
        letter = static_classifier(segment_frames)
    else:
        letter = motion_classifier(segment_frames)
    print(letter)
```

**Requirements**:
- Motion boundary detection: threshold on velocity or direction change
- Static vs. motion threshold: 0.05–0.2 (normalized motion variance)
- Per-segment padding: include 2–3 frames before/after for context

**Pros**:
- Clean separation of concerns
- Easily debug segmentation or classification independently
- Flexible: can swap segmentation or classification methods

**Cons**:
- Errors in Pass 1 propagate to Pass 2 (can't recover)
- Requires good boundary detector (non-trivial to build)

---

### 10. Temporal Convolution (1D CNN)

**Theory**: Use convolutional layers to detect short-range temporal patterns in hand motion.

A 1D CNN is like a sliding-window detector for time series. Filter learns patterns like:
- "Rapid angle change in 3 frames" → motion letter
- "Stable angles for 10 frames" → static letter

**Architecture**:
```
Input: [sequence of 60 frames, each 20D] → (60, 20) tensor
↓
Conv1D(32 filters, kernel=5)  → learns 5-frame patterns
↓
MaxPool1D(2)  → reduce temporal dim
↓
Conv1D(64 filters, kernel=3)  → learn 3-frame patterns
↓
Flatten → Dense(128) → Dense(# letters)
↓
Output: letter probabilities for each frame (or every 3rd frame)
```

**Requirements**:
- Input window: 30–90 frames (1–3 seconds at 30 FPS)
- Kernel sizes: 3–7 (small kernels = short-term patterns)
- Stride: 1 (dense output) or 3–5 (sparse, faster)
- Training data: 100+ sequences per letter

**Pros**:
- Much faster than RNN (no recurrence)
- Good for learning local temporal patterns
- Real-time capable on CPU
- Simpler to train than LSTM

**Cons**:
- Limited temporal receptive field (can't capture 30+ frame dependencies as easily as LSTM)
- Still requires significant data
- Not as state-of-the-art as LSTM/GRU

**Computational advantage**:
```
LSTM: O(seq_len × hidden_dim²) per forward pass
1D CNN: O(seq_len × kernel_size × filters) per forward pass
→ CNN is 2–5x faster
```

---

## Part 2: Token-Based Language Modeling for Prediction

### Token Methodology: Predicting Next Letters, Words, and Sentences

**Theory**: Once you have a recognized sequence of letters/words, use a **language model** to predict what comes next and offer suggestions or auto-corrections.

This is similar to phone autocomplete: after you type "hel", the keyboard suggests "hello" because it's a common word.

### Architecture

**Tokens and Vocabulary**:
- **Token**: a discrete unit (letter, word, or subword)
- **Vocabulary**: set of all possible tokens
  - Letter-level: 26 letters + word boundary marker + punctuation = ~30 tokens
  - Word-level: 1000–100,000 words
  - Subword-level (BPE): 5,000–50,000 subwords (middle ground)

**Language Model Types**:

#### 10a. N-gram Language Model (Simplest)

**Theory**: Predict next token based on the previous N-1 tokens.

Bigram: P(word_t | word_{t-1})
Trigram: P(word_t | word_{t-2}, word_{t-1})

**Implementation**:
1. Count co-occurrences in training data (e.g., SASL video transcriptions).
2. Store as probability table:
   ```
   P(B | A) = count(A→B) / count(A)
   ```
3. To predict: look up table.

**Example**:
```
Training: "hello world", "hello friend", "goodbye world"
P(world | hello) = 1/2 = 0.5
P(friend | hello) = 1/2 = 0.5
P(world | goodbye) = 1.0

User signs: "hello" → model predicts: world (50%), friend (50%)
```

**Requirements**:
- Training corpus: 10,000+ sentences of sign language
- N (n-gram order): 2–3 (higher is data-hungry)
- Smoothing: add-one smoothing to handle unseen n-grams

**Pros**:
- Super simple and fast
- Requires little data
- Fully interpretable

**Cons**:
- Can't capture long-range dependencies (e.g., subject-verb agreement 5+ words apart)
- Limited by vocabulary sparsity
- No learning; purely statistical

#### 10b. Neural Language Model (Transformer)

**Theory**: Use a neural network to learn patterns in token sequences.

A Transformer language model (like GPT) predicts the next token by:
1. Embedding each previous token into a vector space.
2. Using attention layers to find which previous tokens are most relevant.
3. Predicting a probability distribution over next tokens.

**Architecture** (mini version):
```
Input tokens: [t_1, t_2, ..., t_k]  (e.g., letter indices)
↓
Embedding layer: convert to (k, embedding_dim) tensor
↓
Positional encoding: add position info (transformer needs it)
↓
Transformer block (self-attention + feed-forward)
↓
Output layer: predict P(t_{k+1} | t_1, ..., t_k)
```

**How attention works** (intuition):
```
When predicting after "hello world is ...", the model attends:
- HIGH to "world" (probably refers back)
- MEDIUM to "hello" (context)
- ZERO to early punctuation
→ learns to predict "beautiful" or "big"
```

**Requirements**:
- Embedding dimension: 64–256
- Number of attention heads: 4–8
- Training data: 50,000+ sentences
- GPU strongly recommended

**Pros**:
- Captures long-range dependencies
- State-of-the-art predictions
- Generalizes to unseen n-grams

**Cons**:
- Requires significant training data and compute
- Black-box (hard to interpret)
- Slow at inference compared to n-gram

#### 10c. Hybrid: Fallback from Neural to N-gram

**Strategy**: Try neural model first; if prediction is uncertain, fall back to n-gram.

```python
def predict_next(history_tokens):
    neural_pred = transformer_model.predict(history_tokens)
    
    if neural_pred.confidence < 0.6:  # uncertain
        ngram_pred = ngram_model.predict(history_tokens[-2:])
        return ngram_pred
    else:
        return neural_pred
```

**Pros**:
- Robust (fallback available)
- Good confidence scores
- Balances performance and simplicity

---

### Integration with Sequence Recognition

**Workflow**:
1. **Segmentation & Classification**: recognize letters using approach #1–#10.
2. **Accumulate tokens**: store recognized letters/words in a buffer.
3. **Language model**: every few frames, predict next token using n-gram or neural LM.
4. **Display**: show recognized sequence + top-3 predictions for next letter/word.

**Example UI**:
```
Recognized: "HEL"
Predictions: [HELLO (80%), HELP (15%), HELM (5%)]

User continues signing...
Recognized: "HELLO WO"
Predictions: [WORLD (70%), WOULD (20%), WOMAN (10%)]
```

**Requirements for LM**:
- Training corpus: 5,000–50,000 SASL video transcriptions with manual annotations
- Token vocabulary: 100–5,000 (depending on coverage goal)
- Sequence length: use last 2–4 tokens (bigram/trigram + current)

---

## Recommended Implementation Roadmap

### Phase 1: Static + Motion Recognition (Weeks 1–2)
- Use approach **#6 (Hybrid)** for boundaries
- Classify static letters with existing model
- Implement motion letter detection using trajectory matching

### Phase 2: Word Boundaries (Weeks 2–3)
- Add velocity-based boundary detection (**approach #1**)
- Accumulate letters into words
- Display recognized word in real-time

### Phase 3: Language Model (Weeks 3–4)
- Collect 1,000+ video transcriptions
- Train **10a (bigram model)** as a baseline
- Integrate: show top-3 predictions for next letter/word

### Phase 4: Robustness & Optimization (Weeks 4+)
- If performance is good: stop (keep it simple)
- If boundaries are unreliable: upgrade to **#7 (voting)** or **#3 (sliding window)**
- If vocabulary is large: switch to **10b (transformer)** for neural LM

### Phase 5: Advanced (Optional)
- Try **#4 (LSTM/CTC)** if you need rapid fingerspelling
- Try **#5 (HMM)** for interpretability
- Combine multiple approaches for robustness

---

## Comparison Table

| Approach | Complexity | Accuracy | Speed | Tuning | Data Needed |
|----------|-----------|----------|-------|--------|------------|
| 1. State Machine | Low | Medium | Real-time | High | Little |
| 2. Trajectory Seg | Medium | Medium-High | Real-time | Medium | Medium |
| 3. Sliding Window | Medium | High | Moderate | Low | Medium |
| 4. LSTM/CTC | High | Very High | Slow | Low | Lots |
| 5. HMM | Medium | Medium-High | Fast | High | Medium |
| 6. Hybrid (Rule+ML) | Low-Med | High | Real-time | Medium | Little-Med |
| 7. Voting | Medium | High | Moderate | Low | Medium |
| 8. Dictionary | Low | Improves existing | Fast | Low | Vocab file |
| 9. Two-Pass | Medium | High | Real-time | Medium | Medium |
| 10. 1D CNN | High | High | Real-time | Low | Lots |
| 10a. N-gram LM | Very Low | Medium | Real-time | Very Low | Lots of text |
| 10b. Transformer LM | Very High | Very High | Moderate | Low | Lots |

---

## Summary

**Start here**: Combine **#6 (hybrid rule+ML)** + **#8 (dictionary)** + **10a (bigram LM)**.
- Simple, testable, low-cost
- Sufficient for word-level recognition
- Predictions improve user experience

**Graduate to**: **#3 (sliding window)** + **#9 (two-pass)** if boundaries are unreliable, then add **10b (transformer)** when you have 10,000+ training sequences.

The modular design lets you swap components as you learn more about your users' signing patterns.
