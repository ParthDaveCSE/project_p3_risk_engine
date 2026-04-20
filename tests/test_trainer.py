"""
18 Pytest Guards for Model Training
Tests cover: Feature Store integrity, defensive evaluation, model persistence, config fidelity, leakage prevention, artifact pathing
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.trainer import (
    load_feature_store_splits,
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
    check_recall_target,
    save_model,
    load_model,
    run_training_pipeline,
)
from src.utils.config_loader import load_config


# ============================================
# Feature Store Integrity Tests
# ============================================

def test_feature_store_columns_aligned():
    """Train and test splits must have identical columns."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    assert set(X_train.columns) == set(X_test.columns)


def test_feature_store_no_label_in_features():
    """Label column must not appear in feature sets."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    assert "label" not in X_train.columns
    assert "label" not in X_test.columns


def test_feature_store_no_corrupt_labels():
    """No label=-1 records should exist in Feature Store."""
    config = load_config()
    train_path = config["paths"]["train_features"]
    test_path = config["paths"]["test_features"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert -1 not in train_df["label"].values
    assert -1 not in test_df["label"].values


def test_feature_store_shapes_consistent():
    """Train and test splits must have expected shapes."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > 0
    assert len(X_test) > 0


# ============================================
# Model Training Tests
# ============================================

def test_logistic_regression_trains():
    """Logistic Regression must train without errors."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_logistic_regression(X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")


def test_random_forest_trains():
    """Random Forest must train without errors."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")


def test_logistic_regression_uses_liblinear():
    """Logistic Regression must use liblinear solver for reproducibility."""
    config = load_config()
    solver = config["logistic_regression"]["solver"]
    assert solver == "liblinear"


def test_logistic_regression_class_weight_null():
    """Baseline must use class_weight=None from YAML."""
    config = load_config()
    class_weight = config["logistic_regression"]["class_weight"]
    assert class_weight is None


# ============================================
# Defensive Evaluation Tests
# ============================================

def test_evaluate_model_handles_zero_division():
    """ZeroDivision guard: No positive samples should not crash."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    # Create test set with no positive samples
    y_test_no_pos = pd.Series([0] * len(y_test))
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test_no_pos, "TestModel")
    assert metrics["recall"] is None
    assert metrics["auc"] is None


def test_evaluate_model_handles_missing_predict_proba():
    """Missing predict_proba guard should not crash, AUC becomes None."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    # Create a mock model without predict_proba
    class MockModel:
        def predict(self, X):
            return np.zeros(len(X))

    mock_model = MockModel()
    metrics = evaluate_model(mock_model, X_test, y_test, "MockModel")
    assert metrics["auc"] is None


def test_evaluate_model_returns_expected_metrics():
    """Evaluate model must return all expected metric keys."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "TestModel")
    expected_keys = ["model_name", "accuracy", "precision", "recall", "f1_score", "auc", "confusion_matrix"]
    for key in expected_keys:
        assert key in metrics


# ============================================
# Model Persistence Tests
# ============================================

def test_save_and_load_model_identical_predictions(tmp_path):
    """Saved and loaded model must produce identical predictions."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    original_model = train_random_forest(X_train, y_train)
    original_pred = original_model.predict(X_test)

    path = str(tmp_path / "test_model.joblib")
    save_model(original_model, path)
    loaded_model = load_model(path)
    loaded_pred = loaded_model.predict(X_test)

    np.testing.assert_array_equal(original_pred, loaded_pred)


def test_model_artifact_saved_to_correct_path():
    """Model artifact must be saved to path specified in config."""
    config = load_config()
    artifact_path = config["paths"]["model_artifact"]
    baseline_path = config["paths"]["baseline_artifact"]
    assert artifact_path.endswith(".joblib")
    assert baseline_path.endswith(".joblib")


# ============================================
# Config Fidelity Tests
# ============================================

def test_recall_target_in_config():
    """recall_target must be present in YAML config."""
    config = load_config()
    assert "recall_target" in config["model"]
    assert isinstance(config["model"]["recall_target"], float)


def test_default_threshold_in_config():
    """default_threshold must be present in YAML config."""
    config = load_config()
    assert "default_threshold" in config["model"]


def test_random_forest_params_from_config():
    """Random Forest parameters must be read from YAML."""
    config = load_config()
    rf_params = config["random_forest"]
    assert "n_estimators" in rf_params
    assert "max_depth" in rf_params
    assert "random_state" in rf_params


# ============================================
# Check Recall Target Tests
# ============================================

def test_check_recall_target_success():
    """check_recall_target returns True when recall meets target."""
    config = load_config()
    target = config["model"]["recall_target"]
    # Create a recall value above target
    recall_high = target + 0.1
    result = check_recall_target(recall_high, "TestModel")
    assert result is True


def test_check_recall_target_failure():
    """check_recall_target returns False when recall below target."""
    config = load_config()
    target = config["model"]["recall_target"]
    # Create a recall value below target
    recall_low = target - 0.1
    result = check_recall_target(recall_low, "TestModel")
    assert result is False


def test_check_recall_target_handles_none():
    """check_recall_target returns False when recall is None."""
    result = check_recall_target(None, "TestModel")
    assert result is False


# ============================================
# Training Pipeline Integration Tests
# ============================================

def test_run_training_pipeline_returns_two_metrics():
    """run_training_pipeline must return metrics for both models."""
    lr_metrics, rf_metrics = run_training_pipeline()
    assert lr_metrics["model_name"] == "LogisticRegression"
    assert rf_metrics["model_name"] == "RandomForest"


def test_run_training_pipeline_saves_artifacts():
    """Training pipeline must save both model artifacts."""
    config = load_config()
    lr_path = config["paths"]["baseline_artifact"]
    rf_path = config["paths"]["model_artifact"]

    # Run pipeline (will save/overwrite)
    run_training_pipeline()

    assert os.path.exists(lr_path)
    assert os.path.exists(rf_path)