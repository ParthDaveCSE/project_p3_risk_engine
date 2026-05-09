"""
PIMA Diabetes Validation - Architecture Reusability Demonstration

WHAT THIS DEMONSTRATES:
- Architecture reusability: same code pattern works on a completely different domain
- Config-driven absolute impossibility filtering (pima_rules.yaml)
- Leakage-safe imputation (split first, impute using training statistics only)

WHAT THIS DOES NOT DEMONSTRATE:
- Model generalization - PIMA has completely different features (Glucose/BMI/Age)
- A fresh model is trained on PIMA features each time - weights do NOT transfer

The value: This proves the ARCHITECTURE is domain-agnostic, not that the model generalizes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, confusion_matrix

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns in PIMA that use 0 as a missing value placeholder
PIMA_ZERO_COLS = ["Glucose", "BloodPressure", "BMI", "Insulin"]


def rename_pima_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename PIMA columns to match config parameter names.
    Original PIMA columns:
    0. Pregnancies (not used - no clinical bounds in config)
    1. Glucose
    2. BloodPressure
    3. SkinThickness
    4. Insulin
    5. BMI
    6. DiabetesPedigreeFunction
    7. Age
    8. Outcome (label)
    """
    df.columns = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree",
        "age",
        "Outcome",
    ]
    return df


def filter_absolute_impossibilities(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Use pima_rules.yaml to remove biologically impossible records.

    This is the concrete proof that pima_rules.yaml does real work.
    Same filtering logic, different config, same Python code.

    Note: PIMA uses 0 as a missing value placeholder in some columns.
    absolute_min is set to 0.0 so the filter does NOT remove zero
    placeholders — those are handled by leakage-safe imputation below.
    """
    original_len = len(df)
    params = config["parameters"]

    for col, rules in params.items():
        if col not in df.columns:
            continue
        df = df[
            (df[col] >= rules["absolute_min"]) &
            (df[col] <= rules["absolute_max"])
        ]

    removed = original_len - len(df)
    logger.info(
        f"Absolute filter removed {removed} impossible records "
        f"using pima_rules.yaml"
    )
    return df.copy()


def impute_missing_values_leakage_safe(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols: list,
) -> tuple:
    """
    Impute zero-as-missing using training statistics only.

    CRITICAL — split first, then impute.
    Computing median from full dataset before splitting = data leakage.

    Compute median from X_train rows where value > 0 only
    (excluding zeros so they do not skew the median downward).
    Apply that training median to both sets.
    """
    for col in cols:
        if col not in X_train.columns:
            continue

        # Calculate median from training data only (excluding zeros)
        train_median = X_train.loc[X_train[col] > 0, col].median()

        # If all values are zero, median will be NaN - use 0 as fallback
        if pd.isna(train_median):
            train_median = 0.0
            logger.warning(f"{col}: all training values are 0 - using 0 as imputation value")

        # Impute zeros in both train and test using training median
        X_train.loc[X_train[col] == 0, col] = train_median
        X_test.loc[X_test[col] == 0, col] = train_median

        logger.info(
            f"Imputed {col}: training median = {train_median:.2f} "
            f"(applied to both train and test)"
        )

    return X_train, X_test


def run_pima_validation(
    data_path: str = "data/raw/pima_diabetes.csv",
    config_path: str = "config/pima_rules.yaml",
) -> tuple:
    """
    Demonstrate architecture reusability on PIMA Diabetes dataset.

    WHAT THIS IS:
    Architecture reusability — same training code, different config and data.

    WHAT THIS IS NOT:
    Model generalization — the clinical risk engine's trained weights
    cannot transfer to PIMA because the feature spaces are completely
    different (Hemoglobin/WBC/Platelets vs Glucose/BMI/Age).
    A fresh model is trained each time on the new feature set.

    What is demonstrated:
    - Same load_config() works with pima_rules.yaml
    - Same absolute filter runs with different clinical boundaries
    - Same leakage-safe imputation applies to different missing data
    - Same Random Forest training code runs without modification
    - Only YAML and dataset changed — zero Python code changes
    """
    logger.info(f"Loading domain rules from {config_path}")
    config = load_config(config_path)

    logger.info(f"Loading PIMA dataset from {data_path}")
    df = pd.read_csv(data_path, header=None)
    df = rename_pima_columns(df)

    logger.info(f"PIMA dataset loaded | Records: {len(df)} | Columns: {list(df.columns)}")

    # Step 1: Config-driven absolute filter — proves config does real work
    df = filter_absolute_impossibilities(df, config)

    # Define features (exclude pregnancies and Outcome)
    feature_cols = [
        "glucose", "blood_pressure", "skin_thickness", "insulin",
        "bmi", "diabetes_pedigree", "age"
    ]
    X = df[feature_cols]
    y = df["Outcome"]

    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    # Step 2: SPLIT FIRST — before any imputation or scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split complete | Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 3: Leakage-safe imputation using training statistics only
    X_train, X_test = impute_missing_values_leakage_safe(
        X_train, X_test, PIMA_ZERO_COLS
    )

    # Step 4: Scale using training statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train — identical code to clinical pipeline
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Step 6: Evaluate
    y_pred = model.predict(X_test_scaled)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"PIMA validation complete | Recall (diabetic): {recall:.4f}")

    # Print results
    print("\n" + "=" * 60)
    print("PIMA DIABETES VALIDATION - ARCHITECTURE REUSABILITY DEMO")
    print("=" * 60)
    print("\nDataset: PIMA Indians Diabetes (UCI)")
    print(f"Total records after filtering: {len(df)}")
    print(f"Features used: {feature_cols}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Non-diabetic", "Diabetic"]
    ))

    print("\n" + "=" * 60)
    print("ARCHITECTURE REUSABILITY DEMONSTRATED")
    print("=" * 60)
    print(
        "  Config:   config/pima_rules.yaml\n"
        "  Dataset:  PIMA Diabetes (completely different feature space)\n"
        "  Code:     identical training logic — no Python changes\n"
        "  Leakage:  imputation uses training statistics only\n"
        "\n"
        "  Note: This is architecture reusability, NOT model generalization.\n"
        "  A fresh model is trained on PIMA features — the clinical risk\n"
        "  engine's weights do not transfer across different feature spaces.\n"
        "  To validate model generalization, external data with the same\n"
        "  five clinical parameters would be required."
    )

    return model, recall


if __name__ == "__main__":
    model, recall = run_pima_validation()