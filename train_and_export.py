"""
Train an ML classifier on collected CSV feature vectors and export to ONNX.

Usage:
    python train_and_export.py

Output:
 - models/sign_classifier.onnx   (if skl2onnx & onnx available)
 - models/sign_classifier.joblib (scikit-learn pipeline fallback)
 - models/metrics.json           (training/validation metrics)

Notes:
 - Requires: scikit-learn, numpy, pandas (optional), skl2onnx, onnxruntime
 - Install with:
     pip install scikit-learn pandas skl2onnx onnx onnxruntime joblib

This script is intentionally conservative: if ONNX conversion isn't available
it will still save a joblib sklearn pipeline that can be used for local testing
or converted later.
"""

import json
import os
from pathlib import Path
import glob
import sys

import numpy as np

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
# Ensure the models folder exists so we can save model artifacts there.
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load all CSVs from data/*.csv. Returns X (N,22) and y (N,).

    Expects each CSV named LETTER.csv and containing rows of 22 floats.
    """
    X_list = []
    y_list = []

    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        print("No data files found in data/*.csv — collect samples with calibrate.py first.")
        sys.exit(1)

    for f in files:
        label = Path(f).stem
        try:
            rows = np.loadtxt(f, delimiter=',', ndmin=2)
        except Exception as exc:
            print(f"Warning: failed to load {f}: {exc}")
            continue
        if rows.size == 0:
            continue
        # Ensure proper shape: (N, 22)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != 22:
            print(f"Skipping {f}: expected 22 features per row, found {rows.shape[1]}")
            continue
        X_list.append(rows)
        y_list.extend([label] * rows.shape[0])

    if not X_list:
        print("No valid data rows found.")
        sys.exit(1)

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def build_and_train(X_train, y_train, use_rf=False):
    """Create a standardised sklearn pipeline and fit it."""
    if use_rf:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)

    # Scale the features to zero mean / unit variance before training.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline, X, y, label_encoder):
    """Evaluate the fitted pipeline on validation data and return metrics."""
    preds = pipeline.predict(X)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds, labels=label_encoder.transform(label_encoder.classes_))
    return report, cm


def export_onnx(pipeline, X_sample, target_path: Path):
    """Attempt to export the trained sklearn pipeline to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as exc:
        print(f"ONNX conversion unavailable: {exc}")
        return False

    initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
    try:
        onx = convert_sklearn(pipeline, initial_types=initial_type)
        with open(target_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"Exported ONNX model to {target_path}")
        return True
    except Exception as exc:
        # If export fails, keep the joblib pipeline and continue.
        print(f"Failed to convert to ONNX: {exc}")
        return False


def main():
    X, y = load_data()
    print(f"Loaded {X.shape[0]} samples across {len(np.unique(y))} labels")

    # Encode labels to ensure consistent ordering
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Persist the label ordering so ONNX inference can map indices back to strings.
    labels_path = MODELS_DIR / "sign_classifier_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(le.classes_.tolist(), f, indent=2)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Train classifier (start with MLP)
    pipeline = build_and_train(X_train, y_train, use_rf=False)

    # Evaluate
    report, cm = evaluate(pipeline, X_val, y_val, le)
    metrics = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": le.classes_.tolist(),
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")

    # Save sklearn pipeline as a fallback artifact for local inspection or later conversion.
    joblib_path = MODELS_DIR / "sign_classifier.joblib"
    joblib.dump({"pipeline": pipeline, "label_encoder": le}, joblib_path)
    print(f"Saved sklearn pipeline to {joblib_path}")

    # Attempt ONNX export
    onnx_path = MODELS_DIR / "sign_classifier.onnx"
    exported = export_onnx(pipeline, X[:1], onnx_path)
    if not exported:
        print("ONNX export failed — joblib saved for local use. Install skl2onnx to enable ONNX conversion.")

if __name__ == "__main__":
    main()
