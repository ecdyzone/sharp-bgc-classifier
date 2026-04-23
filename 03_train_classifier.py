#!/usr/bin/env python3
"""
03_train_classifier.py

Trains and evaluates a protein role classifier on ESM-2 embeddings.
Compares Random Forest vs Logistic Regression.
Outputs classification report, confusion matrix, and saves the best model.

Usage:
    python 03_train_classifier.py --input data/processed/ --output results/

Outputs:
    results/classification_report.txt
    results/confusion_matrix.csv
    results/best_model.pkl
    results/metrics_summary.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(input_dir: Path):
    embeddings = np.load(input_dir / "embeddings.npy")
    metadata   = pd.read_csv(input_dir / "metadata.csv")
    assert len(embeddings) == len(metadata), "Embeddings and metadata row count mismatch."
    return embeddings, metadata


def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold):
    """Cross-validated predictions → metrics."""
    print(f"\n  [{name}] Running {cv.get_n_splits()}-fold cross-validation...")
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    f1     = f1_score(y, y_pred, average="macro")
    report = classification_report(y, y_pred)
    cm     = confusion_matrix(y, y_pred)
    print(f"  [{name}] Macro F1: {f1:.4f}")
    print(f"\n{report}")
    return f1, y_pred, report, cm


def main():
    parser = argparse.ArgumentParser(description="Train protein role classifier on ESM-2 embeddings.")
    parser.add_argument("--input",  default="data/processed/")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--cv",     type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: Load data ===")
    embeddings, metadata = load_data(Path(args.input))
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Role distribution:\n{metadata['role'].value_counts().to_string()}")

    # Encode labels
    le     = LabelEncoder()
    labels = le.fit_transform(metadata["role"])
    classes = le.classes_
    print(f"  Classes: {list(classes)}")

    # Scale embeddings (important for Logistic Regression)
    scaler = StandardScaler()
    X      = scaler.fit_transform(embeddings)
    y      = labels

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    print("\n=== Step 2: Train and compare models ===")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
    }

    results = {}
    best_f1    = -1
    best_name  = None
    best_model = None
    best_preds = None
    best_cm    = None

    for name, model in models.items():
        f1, preds, report, cm = evaluate_model(name, model, X, y, cv)
        results[name] = {"macro_f1": round(float(f1), 4), "report": report}
        if f1 > best_f1:
            best_f1    = f1
            best_name  = name
            best_model = model
            best_preds = preds
            best_cm    = cm

    print(f"\n  Best model: {best_name} (Macro F1 = {best_f1:.4f})")

    print("\n=== Step 3: Save results ===")

    # Classification report
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best model: {best_name}\nMacro F1: {best_f1:.4f}\n\n")
        f.write(results[best_name]["report"])
    print(f"  Saved: {report_path}")

    # Confusion matrix
    cm_df = pd.DataFrame(best_cm, index=classes, columns=classes)
    cm_path = out_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"  Saved: {cm_path}")

    # Metrics summary
    summary = {
        "best_model":  best_name,
        "macro_f1":    round(best_f1, 4),
        "classes":     list(classes),
        "n_samples":   int(len(y)),
        "cv_folds":    args.cv,
        "all_models": {k: v["macro_f1"] for k, v in results.items()},
    }
    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Fit best model on full data and save
    print(f"\n  Fitting {best_name} on full dataset...")
    best_model.fit(X, y)
    model_bundle = {"model": best_model, "scaler": scaler, "label_encoder": le}
    model_path = out_dir / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"  Saved: {model_path}")

    # Save predictions for visualization script
    metadata["predicted_role"] = le.inverse_transform(best_preds)
    metadata["correct"]        = metadata["role"] == metadata["predicted_role"]
    metadata.to_csv(out_dir / "predictions.csv", index=False)

    print("\nDone. Run 04_visualize.py next.")


if __name__ == "__main__":
    main()
