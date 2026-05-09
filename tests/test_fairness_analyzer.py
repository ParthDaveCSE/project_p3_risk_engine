"""
Fairness Analyzer Tests – 11 new tests for CI/CD integrity
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.explainability.fairness_analyzer import (
    compute_group_metrics,
    analyze_group_performance,
    compute_recall_gap,
    run_feature_removal_experiment,
    generate_fairness_metrics_dict,
    generate_fairness_report,
    plot_group_recall_comparison,
)


def make_classification_data(n=300, seed=42):
    """Minimal dataset with Age column. Older patients more likely positive."""
    np.random.seed(seed)
    n_young = n // 2
    n_older = n - n_young

    X = pd.DataFrame({
        "feature_1": np.concatenate([np.random.normal(0, 1, n_young), np.random.normal(1, 1, n_older)]),
        "feature_2": np.concatenate([np.random.normal(0, 1, n_young), np.random.normal(1.5, 1, n_older)]),
        "bmi": np.concatenate([np.random.normal(25, 5, n_young), np.random.normal(32, 5, n_older)]),
        "blood_pressure": np.concatenate([np.random.normal(110, 10, n_young), np.random.normal(130, 10, n_older)]),
        "age": np.concatenate([np.random.uniform(21, 35, n_young), np.random.uniform(51, 80, n_older)]),
    })

    # Higher positive rate for older patients
    y_young = np.random.choice([0, 1], n_young, p=[0.8, 0.2])
    y_older = np.random.choice([0, 1], n_older, p=[0.6, 0.4])
    y = pd.Series(np.concatenate([y_young, y_older]), name="label")

    return X, y


def train_simple_rf(X, y, n_estimators=20):
    """Train a simple Random Forest for testing."""
    return RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42).fit(X, y)


# ============================================
# Group Metrics Tests
# ============================================

def test_compute_group_metrics_returns_correct_keys():
    """compute_group_metrics must return all required keys."""
    result = compute_group_metrics(
        np.array([1, 0, 1, 1, 0]),
        np.array([1, 0, 0, 1, 0]),
        "TestGroup"
    )
    expected_keys = ["group", "n_total", "n_positive", "recall", "precision", "f1", "estimate_reliable"]
    for key in expected_keys:
        assert key in result


def test_compute_group_metrics_correct_recall():
    """compute_group_metrics must calculate recall correctly."""
    result = compute_group_metrics(
        np.array([1, 0, 1, 1, 0]),
        np.array([1, 0, 0, 1, 0]),
        "TestGroup"
    )
    # TP=2, FN=1 → recall = 2/3 = 0.6667
    assert abs(result["recall"] - 2/3) < 0.001


def test_compute_group_metrics_handles_no_positives():
    """Groups with no positive cases should have recall=None and estimate_reliable=False."""
    result = compute_group_metrics(
        np.array([0, 0, 0, 0, 0]),
        np.array([0, 1, 0, 0, 0]),
        "EmptyGroup"
    )
    assert result["recall"] is None
    assert result["estimate_reliable"] is False


def test_compute_group_metrics_low_support_flag():
    """Groups with fewer than min_positive_for_reliable positives should be flagged."""
    result = compute_group_metrics(
        np.array([1, 0, 1, 0, 0]),
        np.array([1, 0, 0, 0, 0]),
        "SmallGroup",
        min_positive_for_reliable=30
    )
    assert result["estimate_reliable"] is False
    assert result["recall"] is not None


# ============================================
# Recall Gap Tests
# ============================================

def test_compute_recall_gap_correct_calculation():
    """compute_recall_gap must calculate max - min correctly."""
    df = pd.DataFrame({
        "group": ["A", "B", "C"],
        "recall": [0.80, 0.60, 0.70],
        "estimate_reliable": [True, True, True]
    })
    gap = compute_recall_gap(df)
    assert abs(gap - 0.20) < 0.001


def test_compute_recall_gap_handles_insufficient_groups():
    """Fewer than 2 reliable groups should return NaN."""
    df = pd.DataFrame({
        "group": ["A"],
        "recall": [0.75],
        "estimate_reliable": [True]
    })
    gap = compute_recall_gap(df)
    assert np.isnan(gap)


def test_compute_recall_gap_ignores_none_recalls():
    """None recalls should be ignored in gap calculation."""
    df = pd.DataFrame({
        "group": ["A", "B", "C"],
        "recall": [0.80, None, 0.60],
        "estimate_reliable": [True, True, True]
    })
    gap = compute_recall_gap(df)
    assert abs(gap - 0.20) < 0.001


# ============================================
# Threshold Dependency Tests
# ============================================

def test_analyze_group_performance_threshold_parameter():
    """Different thresholds should produce different recall values."""
    X, y = make_classification_data()
    X_no_age = X.drop(columns=["age"])
    model = train_simple_rf(X_no_age, y)

    # Create age groups
    age_values = X["age"]

    result_default = analyze_group_performance(
        model, X_no_age, y, age_values,
        [21, 36, 51, 100], ["Young", "Middle", "Older"],
        threshold=0.5, min_positive_for_reliable=1
    )
    result_low = analyze_group_performance(
        model, X_no_age, y, age_values,
        [21, 36, 51, 100], ["Young", "Middle", "Older"],
        threshold=0.3, min_positive_for_reliable=1
    )

    # At least one group should have different recall between thresholds
    recalls_default = result_default["recall"].tolist()
    recalls_low = result_low["recall"].tolist()
    assert recalls_default != recalls_low


# ============================================
# Feature Removal Experiment Tests
# ============================================

def test_feature_removal_returns_correct_models():
    """Feature removal experiment must return models with different feature counts."""
    X, y = make_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=["age"]), y, test_size=0.25, random_state=42, stratify=y
    )
    age_test = X["age"].iloc[y_test.index].reset_index(drop=True)

    # Reset indices for all dataframes
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model_with, model_without, _, _, _, _ = run_feature_removal_experiment(
        X_train, X_test, y_train, y_test, age_test,
        feature_to_remove="Age",
        bins=[21, 36, 51, 100],
        labels=["Young", "Middle", "Older"],
        min_positive_for_reliable=1
    )

    assert model_with.n_features_in_ == X_train.shape[1]
    assert model_without.n_features_in_ == X_train.shape[1] - 2  # removing 2 proxy features


# ============================================
# Metrics Dictionary Tests
# ============================================

def test_generate_fairness_metrics_dict_returns_required_keys():
    """Generated fairness dict must contain all required keys."""
    metrics_with = pd.DataFrame({
        "group": ["Young", "Older"],
        "n_total": [60, 40],
        "n_positive": [15, 20],
        "recall": [0.60, 0.85],
        "estimate_reliable": [True, True],
    })
    metrics_without = metrics_with.copy()
    metrics_without["recall"] = [0.55, 0.80]

    result = generate_fairness_metrics_dict(
        metrics_with, metrics_without,
        overall_recall_with=0.75,
        overall_recall_without=0.72,
        feature_removed="Age"
    )

    required_keys = ["fairness_metric", "feature_removed", "recall_gap_with",
                     "recall_gap_without", "overall_recall_degraded"]
    for key in required_keys:
        assert key in result

    # Verify JSON serializable
    assert isinstance(json.dumps(result), str)


# ============================================
# Report Generation Tests
# ============================================

def test_generate_fairness_report_contains_required_sections():
    """Fairness report must contain all key sections."""
    metrics_with = pd.DataFrame({
        "group": ["Young", "Older"],
        "n_total": [60, 40],
        "n_positive": [15, 20],
        "recall": [0.60, 0.85],
        "estimate_reliable": [True, True],
    })
    metrics_without = pd.DataFrame({
        "group": ["Young", "Older"],
        "n_total": [60, 40],
        "n_positive": [15, 20],
        "recall": [0.55, 0.80],
        "estimate_reliable": [True, True],
    })

    report = generate_fairness_report(
        metrics_with, metrics_without,
        overall_recall_with=0.75,
        overall_recall_without=0.72,
        feature_removed="Age",
        documented_position="Test position."
    )

    assert "Equal Opportunity" in report
    assert "GROUP PERFORMANCE" in report
    assert "FEATURE REMOVAL" in report
    assert "DOCUMENTED POSITION" in report
    assert "Test position." in report
    assert "proxy feature" in report.lower()


# ============================================
# Plot Generation Tests
# ============================================

def test_plot_generates_file(tmp_path):
    """Fairness comparison plot must be saved as PNG."""
    metrics_with = pd.DataFrame({
        "group": ["Young", "Older"],
        "n_total": [60, 40],
        "n_positive": [15, 20],
        "recall": [0.60, 0.85],
        "estimate_reliable": [True, True],
    })
    metrics_without = pd.DataFrame({
        "group": ["Young", "Older"],
        "n_total": [60, 40],
        "n_positive": [15, 20],
        "recall": [0.55, 0.80],
        "estimate_reliable": [True, True],
    })

    output_path = str(tmp_path / "test_fairness_plot.png")
    result_path = plot_group_recall_comparison(
        metrics_with, metrics_without,
        overall_recall_with=0.75,
        overall_recall_without=0.72,
        feature_removed="Age",
        output_path=output_path
    )

    assert os.path.exists(result_path)
    assert result_path.endswith(".png")