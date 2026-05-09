import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()


class DomainFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that adds clinically reasoned features.
    SKLEARN API RULE (critical): BaseEstimator requires that __init__
    only assign its arguments to self with identical names.
    """

    def __init__(self, creatinine_critical_high=None):
        if creatinine_critical_high is None:
            self.creatinine_critical_high = (
                config["parameters"]["creatinine"]["critical_high"]
            )
        else:
            self.creatinine_critical_high = creatinine_critical_high

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # never modify the input DataFrame in place
        X = X.copy()

        # Feature 1: Glucose-Hemoglobin Ratio
        X["glucose_hemoglobin_ratio"] = (
            X["glucose"] / (X["hemoglobin"] + 1e-9)
        )

        # Feature 2: WBC-Platelet Ratio
        X["wbc_platelet_ratio"] = (
            X["wbc"] / (X["platelets"] + 1e-9)
        )

        # Feature 3: Creatinine Risk Flag
        X["creatinine_risk_flag"] = (
            X["creatinine"] > self.creatinine_critical_high
        ).astype(int)

        logger.info(
            "Domain features added: glucose_hemoglobin_ratio, "
            "wbc_platelet_ratio, creatinine_risk_flag"
        )

        return X

    def get_feature_names_out(self, input_features=None):
        """Required for set_output(transform='pandas') compatibility."""
        # Return all features (original + new)
        if input_features is None:
            return [
                "hemoglobin", "glucose", "wbc", "platelets", "creatinine",
                "glucose_hemoglobin_ratio", "wbc_platelet_ratio", "creatinine_risk_flag"
            ]
        else:
            return list(input_features) + [
                "glucose_hemoglobin_ratio",
                "wbc_platelet_ratio",
                "creatinine_risk_flag"
            ]


def build_feature_pipeline() -> Pipeline:
    """
    Build the sklearn Pipeline with domain features and scaling.
    StandardScaler must come after domain feature creation.
    """
    pipeline = Pipeline([
        ("domain_features", DomainFeatureTransformer()),
        ("scaler", StandardScaler()),
    ])
    pipeline.set_output(transform="pandas")
    return pipeline


def load_and_prepare(data_path="data/processed/clean_patients.csv"):
    """
    Load processed data from L7 pipeline output and prepare for ML.
    Returns tuple: (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(data_path)
    n_before = len(df)

    # Check if label column exists
    if "label" not in df.columns:
        logger.warning("No label column found in data. Creating synthetic labels for demonstration.")
        # Create synthetic labels (80% class 0, 20% class 1) for demonstration
        np.random.seed(42)
        df["label"] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
        logger.info(f"Created synthetic labels: {df['label'].value_counts().to_dict()}")

    # Safety filter: remove any remaining label=-1 records
    df = df[df["label"] != -1].copy()
    n_after = len(df)

    if n_before != n_after:
        logger.warning(f"Safety filter removed {n_before - n_after} label=-1 records")
    else:
        logger.info("Safety filter passed - no label=-1 records found")

    # Drop metadata columns that shouldn't be features
    drop_cols = ["label", "patient_id", "confidence_score"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["label"]

    logger.info(
        f"Dataset loaded | Records: {len(df)} | "
        f"Features: {len(feature_cols)} | "
        f"Class distribution: {y.value_counts().to_dict()}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    logger.info(f"Split complete | Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def run_feature_pipeline(
    data_path="data/processed/clean_patients.csv",
    artifact_path="models/artifacts/feature_pipeline.joblib",
    train_output_path="data/processed/train_features.csv",
    test_output_path="data/processed/test_features.csv",
):
    """
    Run pipeline, save fitted artifact, and save locked train/test splits.
    This implements the Feature Store Pattern.
    """
    X_train, X_test, y_train, y_test = load_and_prepare(data_path)
    pipeline = build_feature_pipeline()

    # CRITICAL: fit only on training data
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    logger.info(
        "Feature engineering complete! "
        f"Train shape: {X_train_processed.shape}, "
        f"Test shape: {X_test_processed.shape}"
    )

    # Reattach labels using .values to strip shuffled pandas index
    train_out = X_train_processed.copy()
    train_out["label"] = y_train.values
    test_out = X_test_processed.copy()
    test_out["label"] = y_test.values

    # Save pipeline artifact
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    logger.info(f"Feature pipeline artifact saved: {artifact_path}")

    # Save locked splits - the Feature Store Pattern
    os.makedirs("data/processed", exist_ok=True)
    train_out.to_csv(train_output_path, index=False)
    test_out.to_csv(test_output_path, index=False)
    logger.info(
        f"Locked splits saved | Train: {train_output_path} | Test: {test_output_path}"
    )

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_feature_pipeline()

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Number of features: {len(X_train.columns)}")
    print(f"\nFeatures: {list(X_train.columns)}")
    print("\nFirst 3 rows of training features:")
    print(X_train.head(3).to_string())
    print("\nClass distribution in training set:")
    print(y_train.value_counts())