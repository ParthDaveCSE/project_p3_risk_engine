import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.models.trainer import load_feature_store_splits, train_random_forest

logger = get_logger(__name__)
config = load_config()


def find_optimal_threshold(model, X_train, y_train, recall_target=None):
    """Find optimal threshold achieving recall target with highest precision."""
    if recall_target is None:
        recall_target = config["model"]["recall_target"]

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model requires predict_proba for threshold tuning")

    y_pred_prob = model.predict_proba(X_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_pred_prob)

    # Align lengths (precision_recall_curve returns n+1 for precisions/recalls)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    valid_mask = recalls >= recall_target
    if not valid_mask.any():
        best_idx = np.argmax(recalls)
        best_threshold = thresholds[best_idx]
        achieved_recall = recalls[best_idx]
        achieved_precision = precisions[best_idx]
        logger.warning(f"Recall target {recall_target} unreachable. Best: {achieved_recall:.4f}")
    else:
        valid_precisions = precisions[valid_mask]
        best_idx_in_valid = np.argmax(valid_precisions)
        best_idx = np.where(valid_mask)[0][best_idx_in_valid]
        best_threshold = thresholds[best_idx]
        achieved_recall = recalls[best_idx]
        achieved_precision = precisions[best_idx]
        logger.info(f"Optimal threshold: {best_threshold:.4f} | Recall: {achieved_recall:.4f} | Precision: {achieved_precision:.4f}")

    return {
        "threshold": round(best_threshold, 4),
        "achieved_recall": round(achieved_recall, 4),
        "achieved_precision": round(achieved_precision, 4),
        "recall_target": recall_target,
        "tuned_on": "training_data",
    }


def predict_with_threshold(model, X, threshold: float):
    """
    Apply custom decision threshold to model predictions.

    Args:
        model: A trained sklearn model with predict_proba method
        X: Input features
        threshold: Decision threshold between 0 and 1

    Returns:
        Binary predictions (0 or 1) based on threshold
    """
    if not hasattr(model, "predict_proba"):
        # Fallback for models without predict_proba
        return model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]
    return (y_pred_prob >= threshold).astype(int)


def plot_precision_recall_curve(model, X_train, y_train, model_name="Model", output_path=None):
    """Plot and save precision-recall curve."""
    if output_path is None:
        output_path = f"models/artifacts/precision_recall_curve_{model_name.lower()}.png"

    y_pred_prob = model.predict_proba(X_train)[:, 1]
    precisions, recalls, _ = precision_recall_curve(y_train, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve saved: {output_path}")
    return output_path


def save_threshold_artifact(threshold_result, output_path=None):
    """Save threshold configuration as JSON artifact."""
    if output_path is None:
        output_path = "models/artifacts/threshold_config.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(threshold_result, f, indent=2)
    logger.info(f"Threshold artifact saved: {output_path}")
    return output_path


def load_threshold_artifact(path=None):
    """Load threshold configuration from JSON artifact."""
    if path is None:
        path = "models/artifacts/threshold_config.json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold artifact not found: {path}")

    with open(path, "r") as f:
        threshold_result = json.load(f)
    logger.info(f"Threshold artifact loaded: {path} | threshold={threshold_result.get('threshold')}")
    return threshold_result


def run_threshold_tuning():
    """Run complete threshold tuning pipeline."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)
    result = find_optimal_threshold(model, X_train, y_train)
    save_threshold_artifact(result)
    plot_precision_recall_curve(model, X_train, y_train, "rf_balanced")
    return result


if __name__ == "__main__":
    run_threshold_tuning()