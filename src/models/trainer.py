"""
Model Training and Evaluation
Loads locked Feature Store splits, trains baseline and primary models,
and evaluates with defensive guards.
"""

import pandas as pd
import numpy as np
import joblib
import os
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()


def load_feature_store_splits():
    """
    Load locked train/test splits from L8 Feature Store.
    Never call train_test_split again - this is the contract.
    """
    train_path = config["paths"]["train_features"]
    test_path = config["paths"]["test_features"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Guard: Verify column alignment
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    assert train_cols == test_cols, f"Column mismatch: Train {train_cols - test_cols}, Test {test_cols - train_cols}"

    # Separate features and labels
    label_col = "label"
    feature_cols = [c for c in train_df.columns if c != label_col]

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    logger.info(f"Loaded Feature Store splits: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """Train baseline Logistic Regression model with config-driven params."""
    lr_params = config["logistic_regression"]
    model = LogisticRegression(
        class_weight=lr_params.get("class_weight"),
        solver=lr_params.get("solver", "liblinear"),
        max_iter=lr_params.get("max_iter", 1000),
        random_state=lr_params.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    logger.info(f"Logistic Regression trained | Params: {lr_params}")
    return model


def train_random_forest(X_train, y_train):
    """Train primary Random Forest model with config-driven params."""
    rf_params = config["random_forest"]
    model = RandomForestClassifier(
        class_weight=rf_params.get("class_weight"),
        n_estimators=rf_params.get("n_estimators", 100),
        max_depth=rf_params.get("max_depth", 10),
        random_state=rf_params.get("random_state", 42),
        n_jobs=rf_params.get("n_jobs", -1),
    )
    model.fit(X_train, y_train)
    logger.info(f"Random Forest trained | Params: {rf_params}")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Defensive evaluation with guards against:
    - Zero division (no positive samples in test)
    - Missing predict_proba
    """
    y_pred = model.predict(X_test)

    # Guard 1: ZeroDivision prevention
    n_positive = (y_test == 1).sum()
    if n_positive == 0:
        logger.warning(f"{model_name}: No positive samples in test set. Skipping recall/AUC.")
        recall = None
        auc = None
    else:
        recall = recall_score(y_test, y_pred)
        # Guard 2: predict_proba availability
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            logger.warning(f"{model_name}: No predict_proba method. AUC not available.")
            auc = None

    # Calculate metrics safely
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle case where only one class appears
        tn = cm[0][0] if cm.shape[0] > 0 else 0
        fp = 0
        fn = 0
        tp = cm[0][0] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4) if recall is not None else None,
        "f1_score": round(f1, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Fixed f-string formatting
    recall_str = f"{recall:.4f}" if recall is not None else "N/A"
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    logger.info(f"{model_name} | Accuracy: {accuracy:.4f} | Recall: {recall_str} | AUC: {auc_str}")
    return metrics


def check_recall_target(recall, model_name="Model"):
    """Check if recall meets the config-driven safety target."""
    target = config["model"]["recall_target"]
    if recall is None:
        logger.warning(f"{model_name}: Cannot check recall target - no positive samples")
        return False

    if recall >= target:
        logger.info(f"{model_name}: SUCCESS - Recall {recall:.4f} meets target {target}")
        return True
    else:
        logger.warning(f"{model_name}: FAILURE - Recall {recall:.4f} below target {target}")
        return False


def save_model(model, path):
    """Save model artifact to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path):
    """Load model artifact from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def run_training_pipeline():
    """
    Main training pipeline:
    1. Load locked Feature Store splits
    2. Train Logistic Regression (baseline)
    3. Train Random Forest (primary)
    4. Evaluate both
    5. Save artifacts
    6. Check recall targets
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load data (no splitting!)
    X_train, X_test, y_train, y_test = load_feature_store_splits()

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest")

    # Check recall targets
    lr_meets_target = check_recall_target(lr_metrics["recall"], "LogisticRegression")
    rf_meets_target = check_recall_target(rf_metrics["recall"], "RandomForest")

    # Save artifacts
    save_model(lr_model, config["paths"]["baseline_artifact"])
    save_model(rf_model, config["paths"]["model_artifact"])

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nLogistic Regression (Baseline):")
    print(f"  Accuracy: {lr_metrics['accuracy']}")
    print(f"  Recall:   {lr_metrics['recall']}")
    print(f"  Precision: {lr_metrics['precision']}")
    print(f"  F1:       {lr_metrics['f1_score']}")
    print(f"  AUC:      {lr_metrics['auc']}")
    print(f"  Meets recall target ({config['model']['recall_target']}): {lr_meets_target}")

    print(f"\nRandom Forest (Primary):")
    print(f"  Accuracy: {rf_metrics['accuracy']}")
    print(f"  Recall:   {rf_metrics['recall']}")
    print(f"  Precision: {rf_metrics['precision']}")
    print(f"  F1:       {rf_metrics['f1_score']}")
    print(f"  AUC:      {rf_metrics['auc']}")
    print(f"  Meets recall target ({config['model']['recall_target']}): {rf_meets_target}")

    print("\n" + "=" * 60)
    print("Confusion Matrices:")
    print(f"  Logistic Regression: TN={lr_metrics['confusion_matrix']['tn']}, FP={lr_metrics['confusion_matrix']['fp']}, FN={lr_metrics['confusion_matrix']['fn']}, TP={lr_metrics['confusion_matrix']['tp']}")
    print(f"  Random Forest:       TN={rf_metrics['confusion_matrix']['tn']}, FP={rf_metrics['confusion_matrix']['fp']}, FN={rf_metrics['confusion_matrix']['fn']}, TP={rf_metrics['confusion_matrix']['tp']}")

    return lr_metrics, rf_metrics


if __name__ == "__main__":
    run_training_pipeline()