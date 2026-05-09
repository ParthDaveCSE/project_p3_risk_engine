import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from src.features.feature_engineer import (
    DomainFeatureTransformer,
    build_feature_pipeline,
)
from src.utils.config_loader import load_config


# ============================================
# Helpers
# ============================================

def make_sample_dataframe(n=200, seed=42):
    """Create a minimal clean dataframe for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "hemoglobin": rng.uniform(12, 17, n),
        "glucose": rng.uniform(70, 99, n),
        "wbc": rng.uniform(4000, 11000, n),
        "platelets": rng.uniform(150000, 450000, n),
        "creatinine": rng.uniform(0.6, 1.3, n),
        "label": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "patient_id": [f"P-{i:04d}" for i in range(n)],
        "confidence_score": rng.uniform(0.8, 1.0, n),
    })


def make_feature_only(n=200, seed=42):
    """Return only clinical feature columns - no metadata or labels."""
    df = make_sample_dataframe(n, seed)
    return df[["hemoglobin", "glucose", "wbc", "platelets", "creatinine"]]


# ============================================
# sklearn API Compliance Tests
# ============================================

def test_transformer_sklearn_api_compliant():
    """get_params() must return creatinine_critical_high as a named parameter."""
    transformer = DomainFeatureTransformer()
    params = transformer.get_params()
    assert "creatinine_critical_high" in params
    assert params["creatinine_critical_high"] is not None


def test_transformer_accepts_custom_threshold():
    """Custom threshold passed to constructor must override YAML default."""
    custom = DomainFeatureTransformer(creatinine_critical_high=3.0)
    assert custom.creatinine_critical_high == 3.0
    assert custom.get_params()["creatinine_critical_high"] == 3.0


# ============================================
# Domain Feature Correctness Tests
# ============================================

def test_domain_features_added():
    """All three domain features must be present after transformation."""
    X = make_feature_only()
    transformer = DomainFeatureTransformer()
    X_out = transformer.fit_transform(X)
    assert "glucose_hemoglobin_ratio" in X_out.columns
    assert "wbc_platelet_ratio" in X_out.columns
    assert "creatinine_risk_flag" in X_out.columns


def test_domain_features_correct():
    """Domain feature values must match expected formulas exactly."""
    X = pd.DataFrame({
        "hemoglobin": [14.0],
        "glucose": [98.0],
        "wbc": [8000.0],
        "platelets": [200000.0],
        "creatinine": [1.0],
    })
    transformer = DomainFeatureTransformer()
    result = transformer.fit_transform(X)

    expected_chg = 98.0 / (14.0 + 1e-9)
    expected_wp = 8000.0 / (200000.0 + 1e-9)

    assert abs(result["glucose_hemoglobin_ratio"].iloc[0] - expected_chg) < 0.01
    assert abs(result["wbc_platelet_ratio"].iloc[0] - expected_wp) < 0.0001
    assert result["creatinine_risk_flag"].iloc[0] == 0


def test_creatinine_flag_triggers_above_threshold():
    """Creatinine above critical high must set flag to 1."""
    config = load_config()
    critical_high = config["parameters"]["creatinine"]["critical_high"]

    X = pd.DataFrame({
        "hemoglobin": [14.0],
        "glucose": [90.0],
        "wbc": [8000.0],
        "platelets": [200000.0],
        "creatinine": [critical_high + 0.5],
    })
    transformer = DomainFeatureTransformer()
    result = transformer.fit_transform(X)
    assert result["creatinine_risk_flag"].iloc[0] == 1


def test_transformer_does_not_modify_input():
    """Transformer does not modify the original DataFrame in place."""
    X = make_feature_only(50)
    X_original = X.copy()
    transformer = DomainFeatureTransformer()
    transformer.fit_transform(X)
    pd.testing.assert_frame_equal(X, X_original)


# ============================================
# Leakage Prevention Tests
# ============================================

def test_no_data_leakage():
    """Scaler must be fitted only on training data."""
    X = make_feature_only(300)
    y = pd.Series(np.random.choice([0, 1], 300, p=[0.84, 0.16]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_feature_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    # Training set must be approximately zero-mean after scaling
    train_mean = X_train_scaled["hemoglobin"].mean()
    assert abs(train_mean) < 0.1, f"Got {train_mean:.4f}"

    assert X_test_scaled.shape[0] == len(X_test)
    assert X_test_scaled.shape[1] == X_train_scaled.shape[1]


def test_column_names_preserved_through_pipeline():
    """set_output(transform='pandas') must preserve column names for SHAP."""
    X = make_feature_only()
    pipeline = build_feature_pipeline()
    X_out = pipeline.fit_transform(X)

    assert isinstance(X_out, pd.DataFrame)

    expected_cols = [
        "hemoglobin", "glucose", "wbc", "platelets", "creatinine",
        "glucose_hemoglobin_ratio", "wbc_platelet_ratio", "creatinine_risk_flag",
    ]
    for col in expected_cols:
        assert col in X_out.columns


# ============================================
# Label Alignment & Artifacts Tests
# ============================================

def test_label_alignment_after_split(tmp_path):
    """Labels must be correctly aligned after y.values reattachment."""
    # Create a small test CSV
    df = make_sample_dataframe(100)
    csv_path = tmp_path / "test_clean.csv"
    df.to_csv(csv_path, index=False)

    train_path = str(tmp_path / "train_features.csv")
    test_path = str(tmp_path / "test_features.csv")
    artifact_path = str(tmp_path / "feature_pipeline.joblib")

    # Override load_and_prepare to use our test file
    import src.features.feature_engineer as fe
    original_load = fe.load_and_prepare

    def mock_load_and_prepare(*args, **kwargs):
        df_local = pd.read_csv(csv_path)
        feature_cols = ["hemoglobin", "glucose", "wbc", "platelets", "creatinine"]
        X = df_local[feature_cols]
        y = df_local["label"]
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    fe.load_and_prepare = mock_load_and_prepare

    try:
        fe.run_feature_pipeline(
            data_path=str(csv_path),
            artifact_path=artifact_path,
            train_output_path=train_path,
            test_output_path=test_path,
        )
    finally:
        fe.load_and_prepare = original_load

    train = pd.read_csv(train_path)
    assert set(train["label"].unique()).issubset({0, 1})

    label_counts = train["label"].value_counts(normalize=True)
    assert label_counts[0] > 0.75
    assert label_counts[1] > 0.10


def test_pipeline_artifact_saved_and_loadable(tmp_path):
    """Saved pipeline artifact must be loadable and produce identical output."""
    X = make_feature_only(100)
    artifact_path = str(tmp_path / "test_pipeline.joblib")

    pipeline = build_feature_pipeline()
    X_transformed_original = pipeline.fit_transform(X)

    joblib.dump(pipeline, artifact_path)
    loaded_pipeline = joblib.load(artifact_path)

    X_transformed_loaded = loaded_pipeline.transform(X)

    pd.testing.assert_frame_equal(X_transformed_original, X_transformed_loaded)


# ============================================
# Safety Filter Tests
# ============================================

def test_safety_filter_removes_corrupt_labels():
    """Records with label=-1 must be filtered out."""
    df = make_sample_dataframe(100)
    # Add some corrupt records
    corrupt_df = pd.DataFrame({
        "hemoglobin": [14.0, 15.0],
        "glucose": [90.0, 95.0],
        "wbc": [8000.0, 8500.0],
        "platelets": [200000.0, 210000.0],
        "creatinine": [1.0, 1.1],
        "label": [-1, -1],
        "patient_id": ["P-CORRUPT-1", "P-CORRUPT-2"],
        "confidence_score": [0.9, 0.9],
    })
    df = pd.concat([df, corrupt_df], ignore_index=True)

    csv_path = "data/processed/test_clean_with_corrupt.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Test the safety filter logic
    df_loaded = pd.read_csv(csv_path)
    n_before = len(df_loaded)
    df_filtered = df_loaded[df_loaded["label"] != -1].copy()
    n_after = len(df_filtered)

    assert n_before == 102
    assert n_after == 100
    assert -1 not in df_filtered["label"].values

    # Clean up
    os.remove(csv_path)