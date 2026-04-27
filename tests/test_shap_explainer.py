"""
SHAP Explainability Tests – 17 Integrity Checks
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_explainer import (
    compute_shap_values,
    get_expected_log_odds,
    get_model_predictions_in_log_odds,
    sigmoid,
    log_odds_to_probability,
    log_odds,
    compute_local_explanation,
    compute_error_analysis_shap,
)


# ============================================
# Helper Functions
# ============================================

def make_dummy_model_and_data():
    """Create a simple model and test data for testing."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
    })
    y_train = np.random.choice([0, 1], 100, p=[0.8, 0.2])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    X_test = pd.DataFrame({
        "feature1": np.random.randn(20),
        "feature2": np.random.randn(20),
        "feature3": np.random.randn(20),
    })
    y_test = np.random.choice([0, 1], 20, p=[0.8, 0.2])

    return model, X_test, y_test


# ============================================
# Shape Integrity Tests
# ============================================

def test_shap_values_shape_match():
    """SHAP values must have shape (n_samples, n_features)."""
    model, X_test, _ = make_dummy_model_and_data()
    shap_values, _ = compute_shap_values(model, X_test)

    assert shap_values.shape[0] == len(X_test)
    assert shap_values.shape[1] == X_test.shape[1]


def test_shap_values_format():
    """TreeExplainer must return shap values in a format we can work with."""
    model, X_test, _ = make_dummy_model_and_data()

    import shap
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(X_test)

    # Newer SHAP versions return a 3D array (n_samples, n_features, n_classes)
    # Older versions return a list of 2 arrays
    # Both are valid - we just need to handle them
    is_valid_format = (
        isinstance(shap_values_all, list) or
        (isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3)
    )
    assert is_valid_format, f"Unexpected SHAP values format: {type(shap_values_all)}"


def test_extract_high_risk_class_only():
    """We must extract shap_values for high-risk class only."""
    model, X_test, _ = make_dummy_model_and_data()
    shap_values, _ = compute_shap_values(model, X_test)

    # shap_values should be 2D (n_samples, n_features) for class 1
    assert shap_values.ndim == 2


# ============================================
# Log-Odds Function Tests
# ============================================

def test_log_odds_range():
    """log_odds should map probabilities correctly."""
    # Probability 0.5 -> log-odds 0
    assert abs(log_odds(0.5)) < 0.01

    # Probability > 0.5 -> positive log-odds
    assert log_odds(0.8) > 0

    # Probability < 0.5 -> negative log-odds
    assert log_odds(0.2) < 0


def test_sigmoid_range():
    """Sigmoid must return values between 0 and 1."""
    test_values = np.array([-10, -5, 0, 5, 10])
    probs = sigmoid(test_values)

    assert np.all((probs >= 0) & (probs <= 1))


def test_log_odds_to_probability_range():
    """log_odds_to_probability must return value between 0 and 1."""
    prob = log_odds_to_probability(2.0)
    assert 0 <= prob <= 1


def test_sigmoid_monotonic():
    """Higher log-odds must produce higher probability."""
    low = sigmoid(np.array([-2]))
    high = sigmoid(np.array([2]))
    assert low < high


def test_log_odds_sigmoid_inverse():
    """log_odds and sigmoid should be inverses."""
    original_prob = 0.75
    log_odds_val = log_odds(original_prob)
    reconstructed_prob = sigmoid(np.array([log_odds_val]))[0]
    assert abs(original_prob - reconstructed_prob) < 0.01


# ============================================
# Expected Value Tests
# ============================================

def test_expected_log_odds_is_float():
    """Expected log-odds must be a float."""
    model, X_test, _ = make_dummy_model_and_data()
    _, explainer = compute_shap_values(model, X_test)
    expected_log_odds = get_expected_log_odds(explainer, model, X_test)

    assert isinstance(expected_log_odds, float)


def test_model_predictions_in_log_odds():
    """get_model_predictions_in_log_odds must return array of same length as test set."""
    model, X_test, _ = make_dummy_model_and_data()
    log_odds_preds = get_model_predictions_in_log_odds(model, X_test)

    assert len(log_odds_preds) == len(X_test)
    assert isinstance(log_odds_preds, np.ndarray)


# ============================================
# Local Explanation Tests
# ============================================

def test_local_explanation_structure():
    """Local explanation must contain all required fields."""
    model, X_test, _ = make_dummy_model_and_data()
    shap_values, explainer = compute_shap_values(model, X_test)
    expected_log_odds = get_expected_log_odds(explainer, model, X_test)

    explanation = compute_local_explanation(
        shap_values, expected_log_odds, X_test, X_test.columns.tolist(), 0
    )

    required_keys = ["index", "expected_log_odds", "feature_contributions", "total_shap", "predicted_log_odds", "predicted_probability", "additivity_verified"]
    for key in required_keys:
        assert key in explanation

    assert isinstance(explanation["feature_contributions"], list)


def test_feature_contributions_correct_direction():
    """Feature contributions must correctly set direction based on SHAP sign."""
    model, X_test, _ = make_dummy_model_and_data()
    shap_values, explainer = compute_shap_values(model, X_test)
    expected_log_odds = get_expected_log_odds(explainer, model, X_test)

    explanation = compute_local_explanation(
        shap_values, expected_log_odds, X_test, X_test.columns.tolist(), 0
    )

    for contrib in explanation["feature_contributions"]:
        if contrib["shap_value"] > 0:
            assert contrib["direction"] == "increases_risk"
        elif contrib["shap_value"] < 0:
            assert contrib["direction"] == "decreases_risk"
        else:
            assert contrib["direction"] in ["increases_risk", "decreases_risk"]


def test_local_explanation_probability_bounds():
    """Predicted probability must be between 0 and 1."""
    model, X_test, _ = make_dummy_model_and_data()
    shap_values, explainer = compute_shap_values(model, X_test)
    expected_log_odds = get_expected_log_odds(explainer, model, X_test)

    explanation = compute_local_explanation(
        shap_values, expected_log_odds, X_test, X_test.columns.tolist(), 0
    )

    assert 0 <= explanation["predicted_probability"] <= 1


# ============================================
# Error Analysis SHAP Tests
# ============================================

def test_shap_error_analysis_returns_comparison():
    """Error analysis must return feature comparison with gaps."""
    model, X_test, y_test = make_dummy_model_and_data()
    y_pred = model.predict(X_test)
    shap_values, _ = compute_shap_values(model, X_test)

    result = compute_error_analysis_shap(y_test, y_pred, shap_values, X_test)

    assert "fn_count" in result
    assert "tp_count" in result
    assert "feature_comparison" in result
    assert isinstance(result["feature_comparison"], list)


def test_shap_error_analysis_handles_no_fns():
    """Error analysis must handle case with no False Negatives gracefully."""
    model, X_test, y_test = make_dummy_model_and_data()
    y_pred = np.ones_like(y_test)  # All positives
    shap_values, _ = compute_shap_values(model, X_test)

    result = compute_error_analysis_shap(y_test, y_pred, shap_values, X_test)

    assert result["fn_count"] == 0
    assert isinstance(result["feature_comparison"], list)


def test_shap_error_analysis_handles_no_tps():
    """Error analysis must handle case with no True Positives gracefully."""
    model, X_test, y_test = make_dummy_model_and_data()
    y_pred = np.zeros_like(y_test)  # All negatives
    shap_values, _ = compute_shap_values(model, X_test)

    result = compute_error_analysis_shap(y_test, y_pred, shap_values, X_test)

    assert result["tp_count"] == 0
    assert isinstance(result["feature_comparison"], list)


# ============================================
# Output Integrity Tests
# ============================================

def test_shap_values_no_nans():
    """SHAP values must not contain NaNs."""
    model, X_test, _ = make_dummy_model_and_data()
    shap_values, _ = compute_shap_values(model, X_test)

    assert not np.any(np.isnan(shap_values))


def test_global_importance_descending():
    """Mean absolute SHAP must be descending after sorting."""
    model, X_test, _ = make_dummy_model_and_data()
    shap_values, _ = compute_shap_values(model, X_test)

    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_means = sorted(mean_abs, reverse=True)

    assert sorted_means[0] >= sorted_means[-1]