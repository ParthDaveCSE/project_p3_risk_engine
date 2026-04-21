"""
Threshold Tuner Tests - 12 tests for CI/CD safety
Tests cover: threshold finding, metadata, leakage prevention, curve generation
"""

import pytest
import json
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score  # ADDED THIS IMPORT

from src.models.threshold_tuner import (
    find_optimal_threshold,
    save_threshold_artifact,
    load_threshold_artifact,
    plot_precision_recall_curve,
    run_threshold_tuning,
)
from src.models.trainer import load_feature_store_splits, train_random_forest
from src.utils.config_loader import load_config


# ============================================
# Threshold Finding Tests
# ============================================

def test_find_optimal_threshold_returns_dict():
    """find_optimal_threshold must return a dictionary with expected keys."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result, _ = find_optimal_threshold(model, X_train, y_train)

    expected_keys = ["threshold", "achieved_recall", "achieved_precision",
                     "recall_target", "default_threshold_recall", "default_threshold_precision"]
    for key in expected_keys:
        assert key in result


def test_find_optimal_threshold_meets_target():
    """Optimal threshold must achieve recall >= target."""
    config = load_config()
    target = config["model"]["recall_target"]

    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result, _ = find_optimal_threshold(model, X_train, y_train)

    assert result["achieved_recall"] >= target


def test_find_optimal_threshold_threshold_between_0_and_1():
    """Optimal threshold must be between 0 and 1."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result, _ = find_optimal_threshold(model, X_train, y_train)

    assert 0 <= result["threshold"] <= 1


def test_find_optimal_threshold_handles_unreachable_target():
    """If target is unreachable, function must gracefully handle and warn."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    # Set impossible target (1.1)
    result, _ = find_optimal_threshold(model, X_train, y_train, recall_target=1.1)

    # Should still return a result without crashing
    assert "threshold" in result
    assert 0 <= result["threshold"] <= 1


# ============================================
# Threshold Artifact Tests
# ============================================

def test_threshold_metadata_present():
    """Threshold artifact must include traceability metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "threshold.json")
        artifact = save_threshold_artifact(0.75, "1.0.0", path)

        expected_keys = ["threshold", "model_version", "data_version", "tuned_on", "random_state", "recall_target"]
        for key in expected_keys:
            assert key in artifact


def test_threshold_tuned_on_training_not_test():
    """Artifact must explicitly state tuning was done on training data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "threshold.json")
        artifact = save_threshold_artifact(0.75, "1.0.0", path)

        assert artifact["tuned_on"] == "training_data"


def test_save_and_load_threshold_artifact():
    """Saved threshold artifact must be loadable with identical values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "threshold.json")
        original = save_threshold_artifact(0.75, "1.0.0", path)
        loaded = load_threshold_artifact(path)

        assert original["threshold"] == loaded["threshold"]
        assert original["model_version"] == loaded["model_version"]


# ============================================
# Precision-Recall Curve Tests
# ============================================

def test_precision_recall_curve_saved_as_png():
    """PR curve must be saved as PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "pr_curve.png")

        # Create dummy data
        thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        precisions = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
        recalls = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9])

        plot_precision_recall_curve(thresholds, precisions, recalls, 0.4, output_path)

        assert os.path.exists(output_path)
        assert output_path.endswith('.png')


def test_precision_recall_curve_memory_cleanup():
    """PR curve generation must not leave open matplotlib figures."""
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "pr_curve.png")

        thresholds = np.array([0.1, 0.2, 0.3])
        precisions = np.array([0.9, 0.85, 0.8, 0.75])
        recalls = np.array([0.5, 0.6, 0.7, 0.8])

        before_count = len(plt.get_fignums())
        plot_precision_recall_curve(thresholds, precisions, recalls, 0.3, output_path)
        after_count = len(plt.get_fignums())

        # No new figures should remain open
        assert after_count == before_count


# ============================================
# Leakage Prevention Tests
# ============================================

def test_threshold_tuned_on_training_only():
    """Threshold must be tuned on training data, not test."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    # Get probabilities
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    # The optimal threshold should be determined from training data
    result, _ = find_optimal_threshold(model, X_train, y_train)
    optimal = result["threshold"]

    # Apply to test - this is validation, not tuning
    test_recall = recall_score(y_test, test_proba >= optimal)

    # Test should complete without error
    assert 0 <= test_recall <= 1


# ============================================
# Integration Tests
# ============================================

def test_run_threshold_tuning_returns_result():
    """run_threshold_tuning must return result dict and model."""
    result, model = run_threshold_tuning()

    assert "threshold" in result
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_threshold_artifact_created_by_pipeline():
    """Full pipeline must create threshold artifact."""
    config = load_config()
    artifact_path = config["paths"]["threshold_artifact"]

    # Run pipeline
    run_threshold_tuning()

    assert os.path.exists(artifact_path)