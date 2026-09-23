"""
Sign Classifier — static South African Sign Language (SASL) handshape recognition.

Phase 1 of the Sign-to-Text pipeline.

Architecture
------------
Phase 1A  Feature vector extraction from MediaPipe landmarks + joint angles.
Phase 1B  Rule-based classifier — deterministic baseline, calibrated per SASL
          handshape. Useful immediately; no training data required.
Phase 1C  ML classifier — small MLP trained on data collected with collect.py.
          Use load_ml_classifier() / classify_static_ml() once a model exists.

Feature vector (22 dimensions)
-------------------------------
  [0:15]   Joint flexion angles in degrees — from joint_angles.py
  [15]     Thumb abduction angle (degrees) — how far thumb is spread from index
  [16:20]  Wrist-normalised fingertip distances — tip-to-wrist / hand_scale
  [21]     Ring–pinky spread angle (degrees)

Usage (rule-based, Phase 1B)
-----------------------------
    from sign_classifier import build_feature_vector, classify_static
    from joint_angles import compute_joint_angles

    angles = compute_joint_angles(world_landmarks)
    fv     = build_feature_vector(world_landmarks, angles)
    result = classify_static(fv, angles)
    if result:
        label, confidence = result

Usage (ML, Phase 1C — requires a trained model)
------------------------------------------------
    from sign_classifier import load_ml_classifier, classify_static_ml

    session = load_ml_classifier("models/sign_classifier.onnx")
    result  = classify_static_ml(session, fv)

Notes on SASL
-------------
South African Sign Language (SASL) is the sign language used in South Africa.
It has its own manual alphabet (fingerspelling) and vocabulary distinct from
other sign languages (ASL, BSL, etc.).

⚠  CALIBRATION REQUIRED: The angle ranges in _RULES below are structural
   starting points for the most geometrically distinct SASL handshapes.
   They MUST be validated and refined against real SASL reference material
   and measured samples collected with collect.py (Phase 2).

   Authoritative SASL references:
   - Deaf Federation of South Africa (DeafSA): https://www.deafsa.co.za
   - SASL grammar materials produced by the Centre for Deaf Studies,
     University of the Witwatersrand
"""

import json
import math
import os
from typing import Optional

import numpy as np

from joint_angles import compute_joint_angles

# ── SASL label set ─────────────────────────────────────────────────────────
# Motion-based letters are not limited to J and Z in the final architecture.
# Letters such as H, J, P, Q, and Z may require trajectory-aware calibration
# and should be routed through the sequence classifier's motion path when their
# identity depends on movement rather than a single static pose.
# All other letters are treated as static handshapes here.
STATIC_LABELS: list[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    # "J" — motion sign → sequence_classifier.py
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
    # "Z" — motion sign → sequence_classifier.py
]

NUM_FEATURES: int = 22

# Path to the label order produced by train_and_export.py.
LABELS_PATH = "models/sign_classifier_labels.json"

# Confidence threshold below which classify_static returns None.
CONFIDENCE_THRESHOLD: float = 0.70


# ── Feature vector ──────────────────────────────────────────────────────────

def build_feature_vector(landmarks, angles: dict) -> np.ndarray:
    """
    Extract a 22-dimensional, pose-invariant feature vector.

    Prefer world landmarks (hand_world_landmarks) — they are metric-scale and
    palm-centred, which gives better scale invariance than image landmarks.
    Image landmarks work but produce noisier distance features.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList (list-like, 21 items).
        angles:    Dict from compute_joint_angles() — 15 named angles in degrees.

    Returns:
        np.ndarray shape (22,), dtype float32.
    """
    # [0:15] Joint flexion angles
    _ANGLE_KEYS = [
        "thumb_cmc",  "thumb_mp",    "thumb_ip",
        "index_mcp",  "index_pip",   "index_dip",
        "middle_mcp", "middle_pip",  "middle_dip",
        "ring_mcp",   "ring_pip",    "ring_dip",
        "little_mcp", "little_pip",  "little_dip",
    ]
    angle_vec = np.array(
        [angles.get(k, 0.0) for k in _ANGLE_KEYS], dtype=np.float32
    )

    # [15] Thumb abduction angle
    thumb_abd = _thumb_abduction_angle(landmarks)

    # [16:21] Wrist-normalised fingertip distances
    # hand_scale = wrist (0) → middle MCP (9), a stable metric for hand size
    hand_scale = _dist3d(landmarks[0], landmarks[9])
    if hand_scale < 1e-6:
        hand_scale = 1.0
    tip_indices = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
    tip_dists = np.array(
        [_dist3d(landmarks[i], landmarks[0]) / hand_scale for i in tip_indices],
        dtype=np.float32,
    )

    # [21] Ring–pinky lateral spread
    ring_pinky = _spread_angle(landmarks, 13, 17)

    return np.concatenate(
        [angle_vec, [thumb_abd], tip_dists, [ring_pinky]]
    ).astype(np.float32)


# ── Rule-based classifier (Phase 1B) ────────────────────────────────────────
#
# Each rule maps feature names to (min_deg, max_deg) inclusive ranges.
# A letter matches when ALL listed features fall within range.
# Features absent from a rule are unconstrained (wildcard).
#
# Feature names for the angle dict: thumb_cmc, thumb_mp, thumb_ip,
#   index_mcp, index_pip, index_dip, middle_mcp, middle_pip, middle_dip,
#   ring_mcp, ring_pip, ring_dip, little_mcp, little_pip, little_dip.
# Special names: thumb_abduction (index 15), ring_pinky_spread (index 21).
#
# ⚠  These rules are structural placeholders.  They model the most
#    geometrically extreme handshapes (fully open, fully closed, clear
#    single-finger extension) where angles are unambiguous.  Most SASL
#    letters share subtle shape differences that a rule can't reliably
#    resolve — those require the ML classifier (Phase 1C).
#    DO NOT ship these rules as final without validating them against
#    measured SASL handshape data.

_RULES: dict[str, dict[str, tuple[float, float]]] = {

    # ── A: Fist with thumb resting beside index, not across ──────────────
    # All fingers tightly curled; thumb extends slightly laterally.
    "A": {
        "index_pip":   (0,  65),
        "middle_pip":  (0,  65),
        "ring_pip":    (0,  65),
        "little_pip":  (0,  65),
        "thumb_ip":    (100, 180),   # thumb relatively straight, not curled in
    },

    # ── B: Flat hand — all fingers fully extended and together ────────────
    "B": {
        "index_pip":   (155, 180),
        "middle_pip":  (155, 180),
        "ring_pip":    (155, 180),
        "little_pip":  (155, 180),
        "thumb_ip":    (140, 180),
    },

    # ── C: Curved/cupped hand — moderate curl on all fingers ─────────────
    "C": {
        "index_pip":   (85, 135),
        "middle_pip":  (85, 135),
        "ring_pip":    (85, 135),
        "little_pip":  (85, 135),
    },

    # ── D: Index extended and curved toward thumb; others curled ─────────
    "D": {
        "index_pip":   (120, 175),
        "middle_pip":  (0,   65),
        "ring_pip":    (0,   65),
        "little_pip":  (0,   65),
    },

    # ── E: All fingers curled tightly toward palm, thumb tucked ──────────
    "E": {
        "index_pip":   (0,  60),
        "middle_pip":  (0,  60),
        "ring_pip":    (0,  60),
        "little_pip":  (0,  60),
        "thumb_ip":    (0,  70),
    },

    # ── I: Pinky only extended; all others tightly curled ────────────────
    "I": {
        "little_pip":  (145, 180),
        "ring_pip":    (0,   65),
        "middle_pip":  (0,   65),
        "index_pip":   (0,   65),
    },

    # ── L: Thumb and index extended at ~90°; other fingers curled ─────────
    "L": {
        "index_pip":       (150, 180),
        "middle_pip":      (0,   65),
        "ring_pip":        (0,   65),
        "little_pip":      (0,   65),
        "thumb_ip":        (140, 180),
        "thumb_abduction": (60,  120),   # thumb spread wide from index
    },

    # ── O: All fingertips meet thumb tip — O/circle shape ────────────────
    "O": {
        "index_pip":   (65, 115),
        "middle_pip":  (65, 115),
        "ring_pip":    (65, 115),
        "little_pip":  (65, 115),
        "thumb_ip":    (50, 110),
    },

    # ── W: Index, middle, ring extended and spread; pinky and thumb curled
    "W": {
        "index_pip":       (145, 180),
        "middle_pip":      (145, 180),
        "ring_pip":        (145, 180),
        "little_pip":      (0,   75),
        "ring_pinky_spread": (10, 45),   # fingers visibly spread
    },

    # ── Y: Thumb and pinky extended; index/middle/ring curled ────────────
    "Y": {
        "little_pip":  (140, 180),
        "ring_pip":    (0,   70),
        "middle_pip":  (0,   70),
        "index_pip":   (0,   70),
        "thumb_ip":    (130, 180),
    },
}

# Map special feature names to their index in the feature vector
_SPECIAL_FEATURE_INDEX: dict[str, int] = {
    "thumb_abduction":  15,
    "ring_pinky_spread": 21,
}


def classify_static(
    feature_vector: np.ndarray,
    angles: dict,
) -> Optional[tuple[str, float]]:
    """
    Classify a static handshape using the rule-based classifier (Phase 1B).

    Args:
        feature_vector: Output of build_feature_vector() — shape (22,).
        angles:         Raw angles dict from compute_joint_angles().

    Returns:
        (label, confidence) if a rule matches with score >= CONFIDENCE_THRESHOLD.
        None if no rule matches.

    Note:
        Once training data exists, replace this with classify_static_ml().
        The rule-based classifier is intentionally conservative — it returns
        None rather than a low-confidence guess to avoid polluting the
        letter buffer with incorrect characters.
    """
    # Build a unified feature lookup dictionary from raw angles and special features.
    lookup = dict(angles)
    lookup["thumb_abduction"]   = float(feature_vector[15])
    lookup["ring_pinky_spread"] = float(feature_vector[21])

    matches: list[tuple[str, float]] = []
    for label, rule in _RULES.items():
        score = _score_rule(lookup, rule)
        if score >= CONFIDENCE_THRESHOLD:
            matches.append((label, score))

    if not matches:
        return None

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0]


# ── ML classifier (Phase 1C) ────────────────────────────────────────────────

def _load_label_names(labels_path: str = LABELS_PATH) -> Optional[list[str]]:
    """Load the label names saved during ML model training."""
    if not os.path.exists(labels_path):
        return None
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            label_names = json.load(f)
        if isinstance(label_names, list):
            return [str(x) for x in label_names]
    except Exception:
        pass
    return None


def load_ml_classifier(model_path: str = "models/sign_classifier.onnx"):
    """
    Load the trained ONNX classifier produced by train_classifier.py.

    Args:
        model_path: Path to the .onnx model file.

    Returns:
        Tuple of (InferenceSession, label_names) where label_names may be None.

    Raises:
        ImportError       if onnxruntime is not installed.
        FileNotFoundError if the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run train_classifier.py after collecting data with collect.py."
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for the ML classifier.\n"
            "Install it with:  pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(model_path)
    label_names = _load_label_names()
    return session, label_names


def classify_static_ml(
    session_info,
    feature_vector: np.ndarray,
    threshold: float = 0.60,
) -> Optional[tuple[str, float]]:
    """
    Classify a static handshape using the trained ML classifier (Phase 1C).

    Args:
        session_info:   Either ONNX InferenceSession or tuple(session, label_names).
        feature_vector: Output of build_feature_vector(), shape (22,).
        threshold:      Minimum confidence to return a result.

    Returns:
        (label, confidence) or None if best class probability < threshold.
    """
    if isinstance(session_info, tuple):
        session, label_names = session_info
    else:
        session = session_info
        label_names = None

    inp = feature_vector.reshape(1, -1).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})

    # Prefer label output if the ONNX model provides it.
    if len(outputs) >= 1 and isinstance(outputs[0], np.ndarray) and outputs[0].ndim == 1:
        label_idx = int(outputs[0][0])
        if label_names is not None and 0 <= label_idx < len(label_names):
            label = label_names[label_idx]
        elif 0 <= label_idx < len(STATIC_LABELS):
            label = STATIC_LABELS[label_idx]
        else:
            label = str(label_idx)
    else:
        label = None
        label_idx = None

    confidence = 0.0

    if len(outputs) >= 2:
        probs_output = outputs[1]
        if isinstance(probs_output, list) and probs_output:
            first_item = probs_output[0]
            if isinstance(first_item, dict):
                if label_idx is not None and label_idx in first_item:
                    confidence = float(first_item[label_idx])
                else:
                    confidence = float(max(first_item.values()))
            else:
                probs = np.array(first_item, dtype=np.float32)
                if probs.ndim == 1 and label_idx is not None and label_idx < probs.shape[0]:
                    confidence = float(probs[label_idx])
                elif probs.ndim == 1:
                    confidence = float(np.max(probs))
        elif isinstance(probs_output, np.ndarray):
            probs = probs_output
            if probs.ndim == 2 and label_idx is not None and label_idx < probs.shape[1]:
                confidence = float(probs[0, label_idx])
            elif probs.ndim == 1:
                confidence = float(np.max(probs))

    if label is None:
        # Fallback: if the model returned a probability array only, choose the best index.
        fallback = None
        if len(outputs) >= 1 and isinstance(outputs[0], np.ndarray):
            probs = np.array(outputs[0][0], dtype=np.float32) if outputs[0].ndim == 2 else np.array(outputs[0], dtype=np.float32)
            if probs.size:
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                if label_names is not None and best_idx < len(label_names):
                    fallback = label_names[best_idx]
                elif best_idx < len(STATIC_LABELS):
                    fallback = STATIC_LABELS[best_idx]
                else:
                    fallback = str(best_idx)
        label = fallback if fallback is not None else "?"

    if confidence < threshold:
        return None
    return label, confidence


# ── Private helpers ─────────────────────────────────────────────────────────

def _dist3d(a, b) -> float:
    """Compute the Euclidean distance between two 3D landmarks."""
    return math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    )


def _thumb_abduction_angle(landmarks) -> float:
    """
    Angle (degrees) between thumb CMC→MCP bone and index MCP→PIP bone.
    Measures how far the thumb is spread from the index finger.
    """
    tx = landmarks[2].x - landmarks[1].x
    ty = landmarks[2].y - landmarks[1].y
    tz = landmarks[2].z - landmarks[1].z

    ix = landmarks[6].x - landmarks[5].x
    iy = landmarks[6].y - landmarks[5].y
    iz = landmarks[6].z - landmarks[5].z

    dot   = tx * ix + ty * iy + tz * iz
    mag_t = math.sqrt(tx**2 + ty**2 + tz**2)
    mag_i = math.sqrt(ix**2 + iy**2 + iz**2)

    if mag_t < 1e-9 or mag_i < 1e-9:
        return 0.0

    # Return the angle between the thumb and index finger base bones.
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_t * mag_i)))))


def _spread_angle(landmarks, mcp_a: int, mcp_b: int) -> float:
    """
    Angle (degrees) between two MCP→PIP bone vectors.
    Used to detect lateral spread between adjacent fingers.
    """
    ax = landmarks[mcp_a + 1].x - landmarks[mcp_a].x
    ay = landmarks[mcp_a + 1].y - landmarks[mcp_a].y
    az = landmarks[mcp_a + 1].z - landmarks[mcp_a].z

    bx = landmarks[mcp_b + 1].x - landmarks[mcp_b].x
    by = landmarks[mcp_b + 1].y - landmarks[mcp_b].y
    bz = landmarks[mcp_b + 1].z - landmarks[mcp_b].z

    dot   = ax * bx + ay * by + az * bz
    mag_a = math.sqrt(ax**2 + ay**2 + az**2)
    mag_b = math.sqrt(bx**2 + by**2 + bz**2)

    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0

    # Return the angle between two adjacent finger bones.
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_a * mag_b)))))


def _score_rule(lookup: dict, rule: dict[str, tuple[float, float]]) -> float:
    """
    Score how well the current feature values satisfy a rule.

    Returns a value in [0, 1]: fraction of constraints fully satisfied.
    A hard tolerance band of ±5° is applied — constraints missed by more than
    5° cause an immediate 0.0 (no partial credit beyond that margin).
    Returns 0.0 if any required feature is missing.
    """
    if not rule:
        return 0.0

    TOLERANCE = 5.0
    satisfied = 0.0

    for feature, (lo, hi) in rule.items():
        val = lookup.get(feature)
        if val is None:
            return 0.0          # required feature missing → hard failure

        if lo <= val <= hi:
            satisfied += 1.0
        elif lo - TOLERANCE <= val <= hi + TOLERANCE:
            satisfied += 0.5    # within tolerance band → partial credit
        else:
            return 0.0          # outside tolerance → hard failure

    return satisfied / len(rule)
