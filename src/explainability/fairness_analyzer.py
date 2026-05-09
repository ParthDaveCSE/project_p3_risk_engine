"""
Fairness Analyzer – Group Fairness Analysis and Feature Removal Paradox

Equal Opportunity (Recall) is our chosen fairness metric because:
- Missing a sick patient (false negative) is the dominant clinical risk
- This cost asymmetry applies equally across all demographic groups

The Feature Removal Paradox:
- Removing a demographic feature (Age) does NOT make a model fair
- The model reconstructs age-related patterns through correlated proxies (BMI, BP)
- Fairness cannot be achieved by feature removal alone – it requires structural changes
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.models.pima_trainer import (
    filter_absolute_impossibilities,
    impute_missing_values_leakage_safe,
    PIMA_ZERO_COLS,
    rename_pima_columns,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)


def load_and_prepare_pima(
    data_path: str = "data/raw/pima_diabetes.csv",
    config_path: str = "config/pima_rules.yaml",
):
    """
    Load PIMA dataset, apply absolute filter, impute missing values,
    and return train/test splits with original Age values (not scaled).

    CRITICAL: Age is extracted BEFORE scaling. Scaled z-scores are meaningless
    for clinical age groups (e.g., "under 35" cannot be defined on z-scores).
    """
    logger.info(f"Loading PIMA data from {data_path}")
    config = load_config(config_path)

    # Load and rename columns
    df = pd.read_csv(data_path, header=None)
    df = rename_pima_columns(df)

    # Apply absolute impossibility filter
    df = filter_absolute_impossibilities(df, config)

    # Define features (exclude pregnancies)
    feature_cols = [
        "glucose", "blood_pressure", "skin_thickness", "insulin",
        "bmi", "diabetes_pedigree", "age"
    ]
    X = df[feature_cols].copy()
    y = df["Outcome"].copy()

    # SPLIT FIRST – before any imputation or scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Extract Age BEFORE scaling (for fairness analysis)
    age_train = X_train["age"].copy().reset_index(drop=True)
    age_test = X_test["age"].copy().reset_index(drop=True)

    # Drop Age from training features (we don't want it in the model)
    X_train = X_train.drop(columns=["age"])
    X_test = X_test.drop(columns=["age"])

    # Reset indices for all DataFrames to ensure alignment
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Leakage-safe imputation using training statistics only
    X_train, X_test = impute_missing_values_leakage_safe(
        X_train, X_test, PIMA_ZERO_COLS
    )

    # Scale using training statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    logger.info(f"Data prepared | Train: {len(X_train_scaled)} | Test: {len(X_test_scaled)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, age_train, age_test, config


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_name: str,
    min_positive_for_reliable: int = 30,
) -> dict:
    """
    Compute recall, precision, f1 for a single group.
    Groups with fewer than min_positive_for_reliable positives are flagged as unreliable.
    """
    n_positive = int((y_true == 1).sum())

    if n_positive == 0:
        return {
            "group": group_name,
            "n_total": len(y_true),
            "n_positive": 0,
            "recall": None,
            "precision": None,
            "f1": None,
            "estimate_reliable": False,
        }

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "group": group_name,
        "n_total": len(y_true),
        "n_positive": n_positive,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "estimate_reliable": n_positive >= min_positive_for_reliable,
    }


def analyze_group_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    age_values: pd.Series,
    bins: list,
    labels: list,
    threshold: float = 0.5,
    min_positive_for_reliable: int = 30,
) -> pd.DataFrame:
    """
    Analyze model performance across age groups.

    CRITICAL: threshold parameter must support production tuned values (0.38),
    not just default 0.5. A model fair at 0.5 may be unfair at deployment threshold.
    """
    # Ensure all inputs have same length and reset indices
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    age_values = age_values.reset_index(drop=True)

    # Get predictions at specified threshold
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Create age groups
    age_groups = pd.cut(age_values, bins=bins, labels=labels, right=False)

    results = []
    for group_name in labels:
        mask = age_groups == group_name
        if not mask.any():
            continue

        y_true_group = y_test[mask]
        y_pred_group = y_pred[mask]

        metrics = compute_group_metrics(
            y_true_group, y_pred_group, group_name, min_positive_for_reliable
        )
        results.append(metrics)

    return pd.DataFrame(results)


def compute_recall_gap(metrics_df: pd.DataFrame) -> float:
    """
    Compute the recall gap (max recall - min recall) across groups.
    Returns NaN if fewer than 2 groups have valid recall estimates.
    """
    if metrics_df.empty:
        return np.nan

    valid_recalls = metrics_df[
        (metrics_df["recall"].notna()) & metrics_df["estimate_reliable"]
    ]["recall"].tolist()

    if len(valid_recalls) < 2:
        return np.nan

    return round(max(valid_recalls) - min(valid_recalls), 4)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest model with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def run_feature_removal_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    age_test: pd.Series,
    feature_to_remove: str = "Age",
    bins: list = None,
    labels: list = None,
    threshold: float = 0.5,
    min_positive_for_reliable: int = 30,
):
    """
    Run the Feature Removal Paradox experiment.
    """
    if bins is None:
        bins = [21, 36, 51, 100]
    if labels is None:
        labels = ["Young (21–35)", "Middle (36–50)", "Older (51+)"]

    # Reset indices to ensure alignment
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    age_test = age_test.reset_index(drop=True)

    # Model WITH all features (Age is already excluded from feature matrix)
    model_with = train_model(X_train, y_train)

    # Remove proxy features correlated with Age (BMI and BloodPressure)
    proxy_features = ["bmi", "blood_pressure"]
    X_train_without = X_train.drop(columns=proxy_features, errors="ignore")
    X_test_without = X_test.drop(columns=proxy_features, errors="ignore")

    model_without = train_model(X_train_without, y_train)

    # Analyze group performance at specified threshold
    metrics_with = analyze_group_performance(
        model_with, X_test, y_test, age_test,
        bins, labels, threshold, min_positive_for_reliable
    )
    metrics_without = analyze_group_performance(
        model_without, X_test_without, y_test, age_test,
        bins, labels, threshold, min_positive_for_reliable
    )

    # Overall recall
    y_pred_proba_with = model_with.predict_proba(X_test)[:, 1]
    y_pred_with = (y_pred_proba_with >= threshold).astype(int)
    overall_recall_with = recall_score(y_test, y_pred_with)

    y_pred_proba_without = model_without.predict_proba(X_test_without)[:, 1]
    y_pred_without = (y_pred_proba_without >= threshold).astype(int)
    overall_recall_without = recall_score(y_test, y_pred_without)

    return (
        model_with, model_without,
        metrics_with, metrics_without,
        overall_recall_with, overall_recall_without,
    )


def generate_fairness_metrics_dict(
    metrics_with: pd.DataFrame,
    metrics_without: pd.DataFrame,
    overall_recall_with: float,
    overall_recall_without: float,
    feature_removed: str,
) -> dict:
    """Generate machine-readable fairness metrics dictionary for MLflow logging."""
    gap_with = compute_recall_gap(metrics_with)
    gap_without = compute_recall_gap(metrics_without)

    def df_to_dict(df):
        if df.empty:
            return []
        return [
            {
                "group": str(row["group"]),
                "n_total": int(row["n_total"]),
                "n_positive": int(row["n_positive"]),
                "recall": float(row["recall"]) if pd.notna(row["recall"]) else None,
                "reliable": bool(row.get("estimate_reliable", True)),
            }
            for _, row in df.iterrows()
        ]

    return {
        "fairness_metric": "equal_opportunity_recall_gap",
        "feature_removed": feature_removed,
        "recall_gap_with": gap_with if not np.isnan(gap_with) else None,
        "recall_gap_without": gap_without if not np.isnan(gap_without) else None,
        "recall_gap_delta": round(gap_without - gap_with, 4) if not np.isnan(gap_with) and not np.isnan(gap_without) else None,
        "overall_recall_with": round(overall_recall_with, 4),
        "overall_recall_without": round(overall_recall_without, 4),
        "overall_recall_delta": round(overall_recall_without - overall_recall_with, 4),
        "gap_worsened_after_removal": bool(gap_without > gap_with) if not np.isnan(gap_with) and not np.isnan(gap_without) else None,
        "overall_recall_degraded": bool(overall_recall_without < overall_recall_with),
        "groups_with": df_to_dict(metrics_with),
        "groups_without": df_to_dict(metrics_without),
        "dataset_limitation": "PIMA is a small, historically biased dataset. Results may not generalize.",
    }


def plot_group_recall_comparison(
    metrics_with: pd.DataFrame,
    metrics_without: pd.DataFrame,
    overall_recall_with: float,
    overall_recall_without: float,
    feature_removed: str = "Age",
    output_path: str = None,
) -> str:
    """Generate bar chart comparing recall by group before/after feature removal."""
    if output_path is None:
        output_path = "models/artifacts/fairness_recall_comparison.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    groups = metrics_with["group"].tolist()
    recall_with = [r if pd.notna(r) else 0 for r in metrics_with["recall"].tolist()]
    recall_without = [r if pd.notna(r) else 0 for r in metrics_without["recall"].tolist()]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, recall_with, width, label="With Proxy Features", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, recall_without, width, label="Without Proxy Features", color="#e74c3c", alpha=0.85)

    ax.axhline(y=overall_recall_with, color="#2980b9", linestyle="-", label=f"Overall Recall (With): {overall_recall_with:.3f}")
    ax.axhline(y=overall_recall_without, color="#c0392b", linestyle="-", label=f"Overall Recall (Without): {overall_recall_without:.3f}")

    ax.set_xlabel("Age Group")
    ax.set_ylabel("Recall (Diabetic Class)")
    ax.set_title("Recall by Age Group — Effect of Removing Proxy Features")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Fairness recall comparison plot saved to {output_path}")
    return output_path


def generate_fairness_report(
    metrics_with: pd.DataFrame,
    metrics_without: pd.DataFrame,
    overall_recall_with: float,
    overall_recall_without: float,
    feature_removed: str,
    documented_position: str = "",
) -> str:
    """Generate human-readable fairness report with documented position."""
    gap_with = compute_recall_gap(metrics_with)
    gap_without = compute_recall_gap(metrics_without)
    gap_delta = gap_without - gap_with if gap_with is not None and gap_without is not None else None
    recall_delta = overall_recall_without - overall_recall_with

    lines = []
    lines.append("=" * 70)
    lines.append("FAIRNESS ANALYSIS REPORT")
    lines.append("Metric: Equal Opportunity (Recall Gap)")
    lines.append("=" * 70)

    lines.append("\n1. GROUP PERFORMANCE - WITH PROXY FEATURES")
    lines.append(f"   Recall gap: {gap_with:.4f}" if gap_with is not None else "   Recall gap: N/A")
    lines.append(f"   Overall recall: {overall_recall_with:.4f}")
    for _, row in metrics_with.iterrows():
        reliable_flag = "" if row.get("estimate_reliable", True) else " ▲ low support"
        lines.append(f"   {row['group']:<22} n={int(row['n_total']):>4} pos={int(row['n_positive']):>3} recall={row['recall']:.4f}{reliable_flag}")

    lines.append("\n2. GROUP PERFORMANCE - WITHOUT PROXY FEATURES")
    lines.append(f"   Recall gap: {gap_without:.4f}" if gap_without is not None else "   Recall gap: N/A")
    lines.append(f"   Overall recall: {overall_recall_without:.4f}")
    for _, row in metrics_without.iterrows():
        reliable_flag = "" if row.get("estimate_reliable", True) else " ▲ low support"
        lines.append(f"   {row['group']:<22} n={int(row['n_total']):>4} pos={int(row['n_positive']):>3} recall={row['recall']:.4f}{reliable_flag}")

    lines.append("\n3. FEATURE REMOVAL ANALYSIS")
    if gap_delta is not None and gap_delta < 0:
        lines.append(f"   Recall gap IMPROVED: {gap_with:.4f} → {gap_without:.4f} (Δ {gap_delta:+.4f})")
        if recall_delta < 0:
            lines.append(f"   TRADEOFF: Lower overall recall ({overall_recall_with:.4f} → {overall_recall_without:.4f}) may not be worth the fairness gain.")
        else:
            lines.append("   Genuine improvement without utility loss.")
    else:
        lines.append(f"   Recall gap WORSENED: {gap_with:.4f} → {gap_without:.4f} (Δ {gap_delta:+.4f})" if gap_delta is not None else "   Recall gap analysis not available")
        lines.append("   This is the proxy feature problem. Fairness cannot be achieved by feature removal alone.")

    lines.append("\n4. DOCUMENTED POSITION")
    if documented_position:
        lines.append(documented_position)
    else:
        lines.append("   In production, I would:")
        lines.append("   1. Add recall gap to MLflow tracking for continuous monitoring")
        lines.append("   2. Set automated alerts if any group falls below threshold")
        lines.append("   3. NOT deploy feature removal as a fairness fix")
        lines.append("   4. Document known gaps in model card for regulatory review")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def run_fairness_analysis(threshold: float = 0.5):
    """
    Complete fairness analysis pipeline.
    threshold parameter supports production tuned values (default 0.5 for baseline).
    """
    logger.info("=" * 70)
    logger.info("FAIRNESS ANALYSIS PIPELINE")
    logger.info("=" * 70)

    # Load data
    X_train, X_test, y_train, y_test, age_train, age_test, config = load_and_prepare_pima()

    # Get fairness config
    fairness_config = config.get("fairness", {})
    bins = fairness_config.get("age_bins", [21, 36, 51, 100])
    labels = fairness_config.get("age_labels", ["Young (21–35)", "Middle (36–50)", "Older (51+)"])
    min_pos = fairness_config.get("min_group_size_for_reliable_estimate", 30)

    # Run experiment
    model_with, model_without, metrics_with, metrics_without, overall_recall_with, overall_recall_without = run_feature_removal_experiment(
        X_train, X_test, y_train, y_test, age_test,
        feature_to_remove="Age",
        bins=bins,
        labels=labels,
        threshold=threshold,
        min_positive_for_reliable=min_pos,
    )

    # Generate artifacts
    metrics_dict = generate_fairness_metrics_dict(
        metrics_with, metrics_without,
        overall_recall_with, overall_recall_without,
        "Age"
    )

    print("\n" + json.dumps(metrics_dict, indent=2))

    # Generate report with documented position
    documented_position = (
        "In production, I would:\n"
        "  1. Add recall gap to MLflow tracking for continuous monitoring\n"
        "  2. Set automated alerts if any group falls below threshold\n"
        "  3. NOT deploy feature removal as a fairness fix\n"
        "  4. Document known gaps in model card for regulatory review\n"
        "  5. Collect more representative data for underrepresented groups"
    )

    report = generate_fairness_report(
        metrics_with, metrics_without,
        overall_recall_with, overall_recall_without,
        "Age",
        documented_position
    )
    print(report)

    # Generate plot
    plot_group_recall_comparison(
        metrics_with, metrics_without,
        overall_recall_with, overall_recall_without,
        "Age"
    )

    # Save metrics as JSON
    output_path = "models/artifacts/fairness_metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"Fairness metrics saved to {output_path}")

    return metrics_dict, report


if __name__ == "__main__":
    # Run with default threshold (0.5)
    metrics_dict, report = run_fairness_analysis(threshold=0.5)