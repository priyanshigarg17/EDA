"""
Evaluate saved loan approval pipeline/model against a CSV containing ground-truth Loan_Status.

Generates classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
and saves a report plus predictions CSV.

Usage:
    
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_pipeline_or_parts(models_dir: Path):
    """Try loading full pipeline or separately saved preprocessor + model."""
    full_path = models_dir / "full_pipeline.joblib"
    preproc_path = models_dir / "preprocessing_pipeline.joblib"
    model_path = models_dir / "logistic_model.joblib"

    if full_path.exists():
        pipe = joblib.load(full_path)
        return pipe, True

    if preproc_path.exists() and model_path.exists():
        preproc = joblib.load(preproc_path)
        model = joblib.load(model_path)
        return (preproc, model), False

    raise FileNotFoundError("No pipeline or model artifacts found in models dir")


def predict_with_artifacts(artifacts, is_full_pipeline, X: pd.DataFrame):
    if is_full_pipeline:
        pipe = artifacts
        preds = pipe.predict(X)
    else:
        preproc, model = artifacts
        X_t = preproc.transform(X)
        preds = model.predict(X_t)

    return preds


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='Y')
    rec = recall_score(y_true, y_pred, pos_label='Y')
    f1 = f1_score(y_true, y_pred, pos_label='Y')

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--target", type=str, default="Loan_Status")
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise RuntimeError(f"Target column {args.target} is missing in CSV")

    y = df[args.target].values

    artifacts, is_full = load_pipeline_or_parts(args.models)

    # load feature metadata if exists
    meta_path = args.models / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        feature_names = meta.get("feature_names", None)
    else:
        feature_names = None

    if feature_names:
        X = df[feature_names]
    else:
        X = df.drop(columns=[args.target])

    preds = predict_with_artifacts(artifacts, is_full, X)
    metrics = compute_metrics(y, preds)

    out_dir = args.out or args.models
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    report_path = out_dir / "eval_classification_metrics.json"
    with open(report_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Save predictions
    pred_path = out_dir / "loan_predictions.csv"
    df["prediction"] = preds
    df.to_csv(pred_path, index=False)

    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("Saved metrics to:", report_path)
    print("Saved predictions to:", pred_path)


if __name__ == "__main__":
    main()