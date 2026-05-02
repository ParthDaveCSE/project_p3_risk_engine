"""
Model Store Tests – Versioning, Bundles, and Loading
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import model_store as ms


def make_simple_model():
    """Create a simple Random Forest for testing."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def test_save_and_load_model_identical_predictions(tmp_path):
    """Saved and loaded model must produce identical predictions."""
    # Override MODEL_DIR for testing
    original_dir = ms.MODEL_DIR
    ms.MODEL_DIR = str(tmp_path)

    try:
        model = make_simple_model()

        # Generate test data
        X_test = pd.DataFrame(np.random.randn(10, 5))
        y_before = model.predict(X_test)

        ms.save_model(model, 'test_rf', version='1.0.0')
        loaded_model = ms.load_model('test_rf', version='1.0.0')
        y_after = loaded_model.predict(X_test)

        np.testing.assert_array_equal(y_before, y_after)
    finally:
        ms.MODEL_DIR = original_dir


def test_load_model_raises_on_missing(tmp_path):
    """load_model must raise ModelArtifactNotFoundError when file missing."""
    original_dir = ms.MODEL_DIR
    ms.MODEL_DIR = str(tmp_path)

    try:
        with pytest.raises(ms.ModelArtifactNotFoundError):
            ms.load_model('nonexistent_model', version='1.0.0')
    finally:
        ms.MODEL_DIR = original_dir


def test_production_bundle_saves_model_and_threshold(tmp_path):
    """Production bundle must contain model, threshold, and metadata."""
    original_dir = ms.MODEL_DIR
    ms.MODEL_DIR = str(tmp_path)

    try:
        model = make_simple_model()
        threshold = 0.38

        path = ms.save_production_bundle(model, threshold, 'prod_rf', version='1.0.0')
        bundle = joblib.load(path)

        assert 'model' in bundle
        assert 'threshold' in bundle
        assert 'metadata' in bundle
        assert abs(bundle['threshold'] - threshold) < 1e-6
        assert bundle['metadata']['name'] == 'prod_rf'
        assert bundle['metadata']['version'] == '1.0.0'
    finally:
        ms.MODEL_DIR = original_dir


def test_load_production_bundle_returns_model_and_threshold(tmp_path):
    """load_production_bundle must return model, threshold, and metadata."""
    original_dir = ms.MODEL_DIR
    ms.MODEL_DIR = str(tmp_path)

    try:
        model = make_simple_model()
        threshold = 0.38

        ms.save_production_bundle(model, threshold, 'prod_rf', version='1.0.0')
        loaded_model, loaded_threshold, metadata = ms.load_production_bundle('prod_rf', version='1.0.0')

        assert loaded_model is not None
        assert abs(loaded_threshold - threshold) < 1e-6
        assert metadata['name'] == 'prod_rf'
    finally:
        ms.MODEL_DIR = original_dir


def test_list_available_models_empty_when_none(tmp_path):
    """list_available_models must return empty list when no artifacts exist."""
    original_dir = ms.MODEL_DIR
    ms.MODEL_DIR = str(tmp_path)

    try:
        available = ms.list_available_models()
        assert available == []
    finally:
        ms.MODEL_DIR = original_dir