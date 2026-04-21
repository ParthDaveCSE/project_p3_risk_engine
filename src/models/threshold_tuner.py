"""
Decision Threshold Tuning
Finds optimal threshold to meet recall target while maximizing precision.
Threshold is tuned on TRAINING data only - no leakage.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib

matplotlib.use('Agg')  # CI/CD headless guard - prevents display crashes
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.models.trainer import load_feature_store_splits, train_random_forest

logger = get_logger(__name__)
config = load_config()


def find_optimal_threshold(model, X_train, y_train, recall_target=None):
    """
    Find optimal decision threshold that meets recall target.
    Tuned on TRAINING data only - critical for preventing leakage.

    Returns:
        dict with threshold, achieved_recall, achieved_precision, and metadata
    """
    if recall_target is None:
        recall_target = config["model"]["recall_target"]

    # Get predicted probabilities on training data
    y_pred_proba = model.predict_proba(X_train)[:, 1]

    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_pred_proba)

    # Guard: Handle case where no threshold meets recall target
    valid_mask = recalls >= recall_target
    if not valid_mask.any():
        logger.warning(f"No threshold achieves recall >= {recall_target}. Using best available.")
        # Use the threshold with highest recall
        best_idx = np.argmax(recalls)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        achieved_recall = recalls[best_idx]
        achieved_precision = precisions[best_idx]
    else:
        # Find thresholds that meet recall target
        valid_indices = np.where(valid_mask)[0]
        # Among those, choose the one with highest precision
        valid_precisions = precisions[valid_indices]
        best_idx_in_valid = np.argmax(valid_precisions)
        best_idx = valid_indices[best_idx_in_valid]

        # Handle off-by-one: thresholds array is one element shorter
        if best_idx < len(thresholds):
            optimal_threshold = thresholds[best_idx]
        else:
            optimal_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5

        achieved_recall = recalls[best_idx]
        achieved_precision = precisions[best_idx]

    # Calculate metrics at default threshold for comparison
    default_recall = recall_score(y_train, y_pred_proba >= 0.5)
    default_precision = precision_score(y_train, y_pred_proba >= 0.5)

    result = {
        "threshold": float(optimal_threshold),
        "achieved_recall": float(achieved_recall),
        "achieved_precision": float(achieved_precision),
        "recall_target": float(recall_target),
        "default_threshold_recall": float(default_recall),
        "default_threshold_precision": float(default_precision),
    }

    logger.info(
        f"Optimal threshold: {optimal_threshold:.4f} | Recall: {achieved_recall:.4f} | Precision: {achieved_precision:.4f}")
    return result, (thresholds, precisions, recalls)


def plot_precision_recall_curve(thresholds, precisions, recalls, optimal_threshold, output_path="reports/pr_curve.png"):
    """
    Generate and save Precision-Recall curve visualization.
    CI/CD safe with memory cleanup.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.3f}')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Memory safety - prevents invisible memory leaks
    logger.info(f"PR curve saved to {output_path}")


def save_threshold_artifact(threshold, model_version="1.0.0", output_path=None):
    """
    Save threshold as versioned JSON artifact for inference.
    Includes traceability metadata.
    """
    if output_path is None:
        output_path = config["paths"]["threshold_artifact"]

    artifact = {
        "threshold": threshold,
        "model_version": model_version,
        "data_version": "v1.0",
        "tuned_on": "training_data",
        "random_state": config["random_forest"]["random_state"],
        "recall_target": config["model"]["recall_target"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(artifact, f, indent=2)

    logger.info(f"Threshold artifact saved to {output_path}")
    return artifact


def load_threshold_artifact(path=None):
    """Load threshold artifact for inference."""
    if path is None:
        path = config["paths"]["threshold_artifact"]

    with open(path, 'r') as f:
        artifact = json.load(f)

    logger.info(f"Threshold artifact loaded from {path}")
    return artifact


def run_threshold_tuning():
    """
    Complete threshold tuning pipeline:
    1. Load Feature Store splits
    2. Train Random Forest with balanced class weights
    3. Find optimal threshold on training data
    4. Save threshold artifact
    5. Generate PR curve
    """
    logger.info("=" * 60)
    logger.info("STARTING THRESHOLD TUNING PIPELINE")
    logger.info("=" * 60)

    # Load data (no splitting!)
    X_train, X_test, y_train, y_test = load_feature_store_splits()

    # Train model with balanced class weights
    logger.info("Training Random Forest with class_weight='balanced'")
    model = train_random_forest(X_train, y_train)

    # Find optimal threshold on training data
    logger.info("Finding optimal threshold on TRAINING data...")
    threshold_result, (thresholds, precisions, recalls) = find_optimal_threshold(model, X_train, y_train)

    # Generate PR curve
    plot_precision_recall_curve(thresholds, precisions, recalls, threshold_result["threshold"])

    # Save threshold artifact
    save_threshold_artifact(threshold_result["threshold"])

    # Validate on test data (no tuning - just evaluation)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    test_recall_at_optimal = recall_score(y_test, y_pred_proba_test >= threshold_result["threshold"])
    test_recall_at_default = recall_score(y_test, y_pred_proba_test >= 0.5)

    logger.info(f"Test Recall at default threshold (0.5): {test_recall_at_default:.4f}")
    logger.info(f"Test Recall at optimal threshold ({threshold_result['threshold']:.4f}): {test_recall_at_optimal:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING SUMMARY")
    print("=" * 60)
    print(f"\nRecall Target: {threshold_result['recall_target']}")
    print(f"\nOptimal Threshold: {threshold_result['threshold']:.4f}")
    print(f"  Training Recall at optimal: {threshold_result['achieved_recall']:.4f}")
    print(f"  Training Precision at optimal: {threshold_result['achieved_precision']:.4f}")
    print(f"\nDefault Threshold (0.5):")
    print(f"  Training Recall: {threshold_result['default_threshold_recall']:.4f}")
    print(f"  Training Precision: {threshold_result['default_threshold_precision']:.4f}")
    print(f"\nTest Set Validation (no tuning):")
    print(f"  Recall at default (0.5): {test_recall_at_default:.4f}")
    print(f"  Recall at optimal ({threshold_result['threshold']:.4f}): {test_recall_at_optimal:.4f}")

    return threshold_result, model


if __name__ == "__main__":
    run_threshold_tuning()