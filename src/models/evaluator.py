"""
Production Evaluator – Locked Artifact Evaluation

Never use default 0.5 threshold in production evaluation.
Load the tuned threshold artifact and use it for all predictions.

WHY SEPARATE TRAIN AND EVAL SCRIPTS?
- Reproducibility: Evaluation must happen without triggering expensive retrains
- Metrics stay objectively tied to saved artifacts, not a newly fitted model
- CI/CD can run evaluation on every commit without retraining
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()


def load_production_artifacts(
    model_path: str = None,
    threshold_path: str = None,
):
    """
    Load the locked model and threshold artifact.

    CRITICAL: Both artifacts come from the Feature Store pattern.
    The model was trained once. The threshold was tuned once on training data.
    Evaluation loads them both – no retraining, no re-tuning.
    """
    if model_path is None:
        model_path = config["paths"]["model_artifact"]
    if threshold_path is None:
        threshold_path = config["paths"]["threshold_artifact"]

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading threshold artifact from {threshold_path}")
    with open(threshold_path, "r") as f:
        threshold_artifact = json.load(f)

    production_threshold = threshold_artifact["threshold"]

    logger.info(
        f"Production threshold loaded: {production_threshold:.4f} | "
        f"Tuned on: {threshold_artifact['tuned_on']} | "
        f"Recall target: {threshold_artifact['recall_target']}"
    )

    return model, production_threshold, threshold_artifact


def load_feature_store_test_data(
    test_path: str = None,
) -> tuple:
    """
    Load the locked test split from the Feature Store.

    CRITICAL: Never load train data for evaluation.
    Test data should never have been seen during training or tuning.
    """
    if test_path is None:
        test_path = config["paths"]["test_features"]

    df = pd.read_csv(test_path)

    # Verify no corrupt labels
    assert -1 not in df["label"].values, "label=-1 found in test data"

    feature_cols = [c for c in df.columns if c != "label"]
    X_test = df[feature_cols]
    y_test = df["label"]

    logger.info(f"Test data loaded | Records: {len(X_test)} | Features: {len(feature_cols)}")
    return X_test, y_test


def predict_with_threshold(model, X, threshold: float) -> np.ndarray:
    """
    Apply custom threshold to probability predictions.

    Never use model.predict() in production evaluation – it hardcodes 0.5.
    """
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)


def generate_confusion_matrix_plot(
    cm: np.ndarray,
    output_path: str = "models/artifacts/confusion_matrix.png",
):
    """Generate and save confusion matrix visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix – Production Threshold")
    plt.colorbar()

    classes = ["Normal", "High-Risk"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def generate_roc_curve(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    output_path: str = "models/artifacts/roc_curve.png",
):
    """Generate ROC curve with operating point marked."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Find the point on ROC curve closest to our threshold
    threshold_idx = np.argmin(np.abs(thresholds - threshold))
    operating_fpr = fpr[threshold_idx]
    operating_tpr = tpr[threshold_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        operating_fpr, operating_tpr, "ro", markersize=10,
        label=f"Operating Point (threshold={threshold:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Production Threshold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved to {output_path} (AUC={roc_auc:.3f})")
    return roc_auc


def generate_precision_recall_curve(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    output_path: str = "models/artifacts/pr_curve.png",
):
    """Generate Precision-Recall curve."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recalls, precisions, "b-", lw=2,
        label=f"PR curve (AP = {avg_precision:.3f})"
    )
    plt.axvline(
        x=recall_score(y_test, y_pred_proba >= threshold),
        color="red", linestyle="--", alpha=0.7,
        label=f"Operating Point (recall={recall_score(y_test, y_pred_proba >= threshold):.3f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve – Production Threshold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve saved to {output_path} (AP={avg_precision:.3f})")
    return avg_precision


def run_production_evaluation(
    model_path: str = None,
    threshold_path: str = None,
    test_path: str = None,
    evaluation_type: str = "production_threshold",
):
    """
    Complete production evaluation pipeline.

    evaluation_type locked to 'production_threshold' – distinguishes from
    L9's 'default_threshold' evaluations in MLflow.
    """
    logger.info("=" * 60)
    logger.info("PRODUCTION EVALUATION – LOCKED ARTIFACTS")
    logger.info("=" * 60)

    # Load artifacts
    model, threshold, threshold_artifact = load_production_artifacts(
        model_path, threshold_path
    )
    X_test, y_test = load_feature_store_test_data(test_path)

    # Predict using production threshold – NOT default 0.5
    y_pred = predict_with_threshold(model, X_test, threshold)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "evaluation_type": evaluation_type,
        "threshold_used": threshold,
        "threshold_tuned_on": threshold_artifact["tuned_on"],
        "recall_target": threshold_artifact["recall_target"],
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "missed_high_risk_pct": round(fn / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0,
    }

    # Generate plots
    generate_confusion_matrix_plot(cm)
    roc_auc = generate_roc_curve(y_test, y_pred_proba, threshold)
    pr_ap = generate_precision_recall_curve(y_test, y_pred_proba, threshold)

    metrics["roc_auc"] = roc_auc
    metrics["pr_ap"] = pr_ap

    # Print summary
    print("\n" + "=" * 60)
    print("PRODUCTION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Evaluation Type:     {metrics['evaluation_type']}")
    print(f"Threshold Used:      {metrics['threshold_used']:.4f}")
    print(f"Threshold Tuned On:  {metrics['threshold_tuned_on']}")
    print(f"Recall Target:       {metrics['recall_target']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AP:     {metrics['pr_ap']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}  |  FP: {fp}")
    print(f"  FN: {fn}  |  TN: {tn}")
    print(f"\nMissed high-risk patients: {fn}/{tp+fn} ({metrics['missed_high_risk_pct']}%)")

    # Recall target check
    if metrics["recall"] >= metrics["recall_target"]:
        logger.info(f"SUCCESS: Recall {metrics['recall']:.4f} meets target {metrics['recall_target']}")
    else:
        logger.warning(f"FAILURE: Recall {metrics['recall']:.4f} below target {metrics['recall_target']}")

    return metrics, cm, y_pred, y_pred_proba


if __name__ == "__main__":
    metrics, cm, y_pred, y_pred_proba = run_production_evaluation()