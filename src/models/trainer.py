"""
Model Training and Evaluation with MLflow Tracking
"""

import os
import sys
import subprocess
import hashlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

import mlflow
import mlflow.sklearn
import sklearn

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.models.model_store import save_production_bundle, save_model as store_save_model

logger = get_logger(__name__)
config = load_config()


def _get_git_commit() -> str:
    """Get the current git commit hash for reproducibility."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _get_data_hash(path: str) -> str:
    """Compute MD5 hash of a data file for reproducibility logging."""
    if not os.path.exists(path):
        logger.warning(f'Data file not found for hashing: {path}')
        return 'file_not_found'
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_feature_store_splits():
    """Load locked train/test splits from L8 Feature Store."""
    train_path = config["paths"]["train_features"]
    test_path = config["paths"]["test_features"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Guard: Verify column alignment
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    assert train_cols == test_cols, f"Column mismatch: Train {train_cols - test_cols}, Test {test_cols - train_cols}"

    label_col = "label"
    feature_cols = [c for c in train_df.columns if c != label_col]

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    logger.info(f"Loaded Feature Store splits: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """Train baseline Logistic Regression model."""
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
    """Train primary Random Forest model."""
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


def evaluate_model(model, X_test, y_test, model_name="Model", threshold=0.5):
    """
    Defensive evaluation with guards against:
    - Zero division (no positive samples in test)
    - Missing predict_proba
    """
    # Guard: Check if model has predict_proba method
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Only compute AUC if there are samples from both classes
        n_positive = (y_test == 1).sum()
        n_negative = (y_test == 0).sum()
        if n_positive > 0 and n_negative > 0:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
            logger.warning(f"{model_name}: Cannot compute AUC - only one class present in test set.")
    else:
        y_pred_proba = np.zeros(len(X_test))
        auc = None
        logger.warning(f"{model_name}: No predict_proba method. AUC set to None.")

    y_pred = (y_pred_proba >= threshold).astype(int)

    n_positive = (y_test == 1).sum()
    if n_positive == 0:
        logger.warning(f"{model_name}: No positive samples in test set. Recall set to None.")
        recall = None
    else:
        recall = recall_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0][0] if cm.shape[0] > 0 else 0
        fp = 0
        fn = 0
        tp = cm[0][0] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0

    metrics = {
        "model_name": model_name,
        "evaluation_type": "production_threshold" if threshold != 0.5 else "default_threshold",
        "threshold_used": threshold,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4) if recall is not None else None,
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    recall_str = f"{recall:.4f}" if recall is not None else "N/A"
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    logger.info(f"{model_name} | Accuracy: {accuracy:.4f} | Recall: {recall_str} | AUC: {auc_str}")
    return metrics


def log_training_run_to_mlflow(
    model,
    metrics: dict,
    threshold_result: dict,
    model_name: str,
    experiment_name: str = 'clinical_risk_engine',
    extra_params: dict = None,
    artifact_paths: list = None,
) -> str:
    """Log a training run to MLflow with full reproducibility chain."""
    mlflow.set_experiment(experiment_name)

    model_cfg = config['model']
    train_path = config['paths'].get('train_features', '')
    git_commit = _get_git_commit()
    train_data_hash = _get_data_hash(train_path)

    params = {
        'model_type': model_cfg.get('type', 'unknown'),
        'n_estimators': str(model_cfg.get('n_estimators', 'N/A')),
        'max_depth': str(model_cfg.get('max_depth', 'N/A')),
        'class_weight': str(model_cfg.get('class_weight', 'None')),
        'random_state': str(model_cfg.get('random_state', 42)),
        'recall_target': str(model_cfg.get('recall_target', 0.80)),
        'default_threshold': str(model_cfg.get('default_threshold', 0.5)),
        'threshold': str(threshold_result.get('threshold', 'N/A')),
        'threshold_tuned_on': threshold_result.get('tuned_on', 'N/A'),
        'data_version': threshold_result.get('data_version', 'N/A'),
        'evaluation_type': metrics.get('evaluation_type', 'unknown'),
        'git_commit': git_commit,
        'train_data_hash': train_data_hash,
        'python_version': sys.version.split()[0],
        'sklearn_version': sklearn.__version__,
    }

    if extra_params:
        params.update({k: str(v) for k, v in extra_params.items()})

    numeric_metrics = {k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and v is not None
                       and k not in ('confusion_matrix', 'model_name', 'evaluation_type', 'threshold_used')}

    tags = {
        'stage': 'experiment',
        'model_family': model_cfg.get('type', 'unknown'),
        'use_case': 'clinical_risk',
        'recall_target': str(model_cfg.get('recall_target', 0.80)),
    }

    with mlflow.start_run(run_name=model_name) as run:
        try:
            mlflow.log_params(params)
            mlflow.log_metrics(numeric_metrics)
            mlflow.set_tags(tags)

            mlflow.sklearn.log_model(model, artifact_path='mlflow_model')

            for artifact_path in (artifact_paths or []):
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_path='plots')

            run_id = run.info.run_id
            logger.info(f"MLflow run logged: {run_id}")
            return run_id

        except Exception as e:
            mlflow.set_tag('failure_reason', str(e)[:250])
            logger.error(f'MLflow logging failed for run {model_name}: {e}')
            raise


def compare_models(metrics_list: list) -> pd.DataFrame:
    """Side-by-side comparison of models."""
    if not metrics_list:
        logger.warning("compare_models called with empty metrics list")
        return pd.DataFrame()

    df = pd.DataFrame(metrics_list)
    if 'model_name' in df.columns:
        df = df.set_index('model_name')

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(df.to_string())

    if 'recall' in df.columns:
        recall_values = df['recall'].dropna()
        if not recall_values.empty:
            best_name = recall_values.idxmax()
            best_recall = recall_values.max()
            print(f"\nBest recall: {best_name} — {best_recall:.4f}")

    return df


def run_training_pipeline():
    """Main training pipeline with MLflow tracking."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_feature_store_splits()

    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression", threshold=0.5)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest", threshold=0.5)

    from src.models.threshold_tuner import find_optimal_threshold, save_threshold_artifact, plot_precision_recall_curve

    threshold_result, _ = find_optimal_threshold(rf_model, X_train, y_train)
    optimal_threshold = threshold_result['threshold']

    rf_metrics_tuned = evaluate_model(
        rf_model, X_test, y_test, "RandomForest_Tuned",
        threshold=optimal_threshold
    )
    rf_metrics_tuned['evaluation_type'] = 'custom_threshold'

    compare_models([lr_metrics, rf_metrics, rf_metrics_tuned])

    store_save_model(lr_model, 'logistic_regression_balanced', version='1.0.0')
    store_save_model(rf_model, 'random_forest_balanced', version='1.0.0')
    save_threshold_artifact(threshold_result)

    pr_curve_path = plot_precision_recall_curve(rf_model, X_train, y_train, 'RF_Balanced')

    bundle_path = save_production_bundle(
        rf_model, optimal_threshold,
        name='rf_balanced_production',
        version='1.0.0',
    )
    logger.info(f'Production bundle saved: {bundle_path}')

    model_dir = config['paths']['model_dir']
    plot_artifacts = []
    for plot_name in [
        'confusion_matrix.png', 'roc_curve.png', 'shap_beeswarm.png',
        'fairness_recall_comparison.png',
    ]:
        full_path = os.path.join(model_dir, plot_name)
        if os.path.exists(full_path):
            plot_artifacts.append(full_path)

    if pr_curve_path and os.path.exists(pr_curve_path):
        plot_artifacts.append(pr_curve_path)

    lr_threshold_stub = {
        'threshold': config['model']['default_threshold'],
        'tuned_on': 'default_not_tuned',
        'data_version': 'feature_store_v1.0'
    }

    run_id_lr = log_training_run_to_mlflow(
        model=lr_model,
        metrics=lr_metrics,
        threshold_result=lr_threshold_stub,
        model_name='LR_Balanced',
        artifact_paths=plot_artifacts,
        extra_params={
            'solver': config['logistic_regression']['solver'],
            'max_iter': config['logistic_regression']['max_iter'],
            'lr_class_weight': str(config['logistic_regression'].get('class_weight')),
        },
    )

    run_id_rf = log_training_run_to_mlflow(
        model=rf_model,
        metrics=rf_metrics_tuned,
        threshold_result=threshold_result,
        model_name='RF_Balanced_ThresholdTuned',
        artifact_paths=plot_artifacts,
    )

    logger.info(f"MLflow runs complete | LR: {run_id_lr} | RF: {run_id_rf}")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print("\nLogistic Regression (Baseline):")
    print(f"  Accuracy: {lr_metrics['accuracy']}")
    print(f"  Recall:   {lr_metrics['recall']}")
    print(f"  Precision: {lr_metrics['precision']}")
    print(f"  F1:       {lr_metrics['f1_score']}")
    print(f"  AUC:      {lr_metrics['auc_roc']}")

    print("\nRandom Forest (Primary):")
    print(f"  Accuracy: {rf_metrics['accuracy']}")
    print(f"  Recall:   {rf_metrics['recall']}")
    print(f"  Precision: {rf_metrics['precision']}")
    print(f"  F1:       {rf_metrics['f1_score']}")
    print(f"  AUC:      {rf_metrics['auc_roc']}")

    print(f"\nRandom Forest (Threshold Tuned - {optimal_threshold:.4f}):")
    print(f"  Accuracy: {rf_metrics_tuned['accuracy']}")
    print(f"  Recall:   {rf_metrics_tuned['recall']}")
    print(f"  Precision: {rf_metrics_tuned['precision']}")
    print(f"  F1:       {rf_metrics_tuned['f1_score']}")
    print(f"  AUC:      {rf_metrics_tuned['auc_roc']}")

    print("\n" + "=" * 60)
    print("MLflow Tracking:")
    print("  Run 'uv run mlflow ui' to view results")
    print("  Experiment: clinical_risk_engine")
    print(f"  LR Run ID: {run_id_lr[:8]}...")
    print(f"  RF Run ID: {run_id_rf[:8]}...")

    return lr_metrics, rf_metrics, rf_metrics_tuned


if __name__ == "__main__":
    lr_metrics, rf_metrics, rf_tuned_metrics = run_training_pipeline()