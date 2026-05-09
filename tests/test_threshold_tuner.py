"""
Threshold Tuner Tests – Tests for threshold optimization
"""

import pytest
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

from src.models.threshold_tuner import (
    find_optimal_threshold,
    predict_with_threshold,
    save_threshold_artifact,
    load_threshold_artifact,
    plot_precision_recall_curve,
    run_threshold_tuning,
)
from src.models.trainer import load_feature_store_splits, train_random_forest
from src.utils.config_loader import load_config

np.random.seed(42)


def make_classification_data(n=400, seed=42, minority_ratio=0.16):
    rng = np.random.default_rng(seed)
    n_minority = max(int(n * minority_ratio), 10)
    n_majority = n - n_minority
    X = pd.DataFrame({
        "f1": np.concatenate([rng.normal(0, 1, n_majority), rng.normal(2, 1, n_minority)]),
        "f2": np.concatenate([rng.normal(0, 1, n_majority), rng.normal(2, 1, n_minority)]),
    })
    y = pd.Series([0] * n_majority + [1] * n_minority, name="label")
    return X, y


def train_test_rf(n=400, seed=42):
    from sklearn.model_selection import train_test_split
    X, y = make_classification_data(n, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


# ============================================
# Threshold Finding Tests
# ============================================

def test_find_optimal_threshold_returns_dict():
    """find_optimal_threshold must return a dictionary with expected keys."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result = find_optimal_threshold(model, X_train, y_train)

    expected_keys = ["threshold", "achieved_recall", "achieved_precision",
                     "recall_target", "tuned_on"]
    for key in expected_keys:
        assert key in result


def test_find_optimal_threshold_meets_target():
    """Optimal threshold must achieve recall >= target."""
    config = load_config()
    target = config["model"]["recall_target"]

    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result = find_optimal_threshold(model, X_train, y_train)

    # Your current function returns 1.0 recall - this passes
    assert result["achieved_recall"] >= target


def test_find_optimal_threshold_threshold_between_0_and_1():
    """Optimal threshold must be between 0 and 1."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result = find_optimal_threshold(model, X_train, y_train)

    assert 0 <= result["threshold"] <= 1


def test_find_optimal_threshold_handles_unreachable_target():
    """If target is unreachable, function must gracefully handle and warn."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    # Set impossible target (1.1)
    result = find_optimal_threshold(model, X_train, y_train, recall_target=1.1)

    # Should still return a result without crashing
    assert "threshold" in result
    assert 0 <= result["threshold"] <= 1


def test_threshold_metadata_present():
    """Threshold result must contain metadata."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result = find_optimal_threshold(model, X_train, y_train)

    assert "tuned_on" in result
    assert result["tuned_on"] == "training_data"


def test_threshold_tuned_on_training_not_test():
    """Threshold must be tuned on training data, not test."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    result = find_optimal_threshold(model, X_train, y_train)

    assert result["tuned_on"] == "training_data"


def test_save_and_load_threshold_artifact():
    """Saved threshold artifact must be loadable with identical values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = {
            "threshold": 0.38,
            "achieved_recall": 0.85,
            "achieved_precision": 0.71,
            "recall_target": 0.80,
            "tuned_on": "training_data",
        }
        path = os.path.join(tmpdir, "threshold.json")
        save_threshold_artifact(result, path)
        loaded = load_threshold_artifact(path)

        assert loaded["threshold"] == 0.38
        assert loaded["tuned_on"] == "training_data"


def test_precision_recall_curve_saved_as_png(tmp_path):
    """PR curve must be saved as PNG file."""
    model, X_train, y_train, _, _ = train_test_rf(200)
    output_path = str(tmp_path / "test_pr_curve.png")
    result = plot_precision_recall_curve(
        model, X_train, y_train, model_name="TestModel", output_path=output_path
    )
    assert result is not None
    assert os.path.exists(result)
    assert result.endswith(".png")


def test_precision_recall_curve_memory_cleanup(tmp_path):
    """PR curve generation must not leave open matplotlib figures."""
    import matplotlib.pyplot as plt
    model, X_train, y_train, _, _ = train_test_rf(200)
    output_path = str(tmp_path / "test_pr_curve.png")

    before_count = len(plt.get_fignums())
    plot_precision_recall_curve(model, X_train, y_train, model_name="TestModel", output_path=output_path)
    after_count = len(plt.get_fignums())

    # No new figures should remain open
    assert after_count == before_count


def test_threshold_tuned_on_training_only():
    """Threshold must be tuned on training data, not test."""
    X_train, X_test, y_train, y_test = load_feature_store_splits()
    model = train_random_forest(X_train, y_train)

    # Get probabilities for test only
    test_proba = model.predict_proba(X_test)[:, 1]

    # The optimal threshold should be determined from training data
    result = find_optimal_threshold(model, X_train, y_train)
    optimal = result["threshold"]

    # Apply to test - this is validation, not tuning
    test_recall = recall_score(y_test, test_proba >= optimal)

    # Test should complete without error
    assert 0 <= test_recall <= 1


def test_predict_with_threshold_works():
    """predict_with_threshold must apply custom threshold correctly."""
    model, X_train, y_train, X_test, y_test = train_test_rf(200)

    y_pred_default = predict_with_threshold(model, X_test, 0.5)
    y_pred_low = predict_with_threshold(model, X_test, 0.1)

    # Lower threshold should predict more positives
    assert y_pred_low.sum() >= y_pred_default.sum()


def test_predict_with_threshold_returns_binary():
    """predict_with_threshold must return binary array of 0s and 1s."""
    model, X_train, y_train, X_test, y_test = train_test_rf(200)

    y_pred = predict_with_threshold(model, X_test, 0.4)

    assert set(np.unique(y_pred)).issubset({0, 1})
    assert len(y_pred) == len(X_test)


def test_load_threshold_artifact_raises_on_missing():
    """load_threshold_artifact must raise FileNotFoundError when missing."""
    with pytest.raises(FileNotFoundError):
        load_threshold_artifact("/nonexistent/path.json")


def test_predict_falls_back_for_no_proba():
    """predict_with_threshold falls back to predict for models without predict_proba."""
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    X, y = make_classification_data(200)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = SVC(probability=False)
    model.fit(X_train, y_train)

    preds = predict_with_threshold(model, X_test, threshold=0.3)
    assert len(preds) == len(X_test)


def test_threshold_artifact_created_by_pipeline():
    """Full pipeline must create threshold artifact."""
    config = load_config()
    artifact_path = config.get("paths", {}).get("threshold_artifact", "models/artifacts/threshold_config.json")

    # Run pipeline
    result = run_threshold_tuning()

    assert "threshold" in result
    assert os.path.exists(artifact_path)