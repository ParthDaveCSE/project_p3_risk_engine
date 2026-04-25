"""
Error Analyzer – False Negative Diagnosis

Not all False Negatives are the same. Segment by probability to diagnose:
- Boundary errors: model was uncertain, needs threshold tuning
- Confident wrong: model fundamentally misunderstood, needs feature engineering

The Confidence Score Paradox:
High confidence score ≠ correct prediction. It only means the input data
was biologically valid. Predictive completeness is separate.
"""

import os
import pandas as pd
import numpy as np
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.models.evaluator import (
    load_production_artifacts,
    load_feature_store_test_data,
    predict_with_threshold,
)

logger = get_logger(__name__)
config = load_config()


def extract_false_negatives(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    X_test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Extract all False Negative records with their metadata.

    False Negative = actual high-risk (1) but predicted normal (0)
    """
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    if len(fn_indices) == 0:
        logger.info("No False Negatives found – model is perfect on this test set")
        return pd.DataFrame()

    fn_records = X_test.iloc[fn_indices].copy()
    fn_records["true_label"] = y_test[fn_mask]
    fn_records["predicted_label"] = y_pred[fn_mask]
    fn_records["predicted_probability"] = y_pred_proba[fn_mask]
    fn_records["original_index"] = fn_indices

    logger.info(f"Extracted {len(fn_records)} False Negative records")
    return fn_records


def segment_false_negatives(
    fn_df: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    Segment False Negatives by probability.

    Boundary Errors: probability between 0.5 * threshold and threshold
    Confident Wrong: probability below 0.5 * threshold

    This segmentation converts a count into a diagnosis.
    """
    if fn_df.empty:
        return {
            "boundary_errors": pd.DataFrame(),
            "confident_wrong": pd.DataFrame(),
            "boundary_count": 0,
            "confident_count": 0,
            "boundary_pct": 0,
            "confident_pct": 0,
        }

    boundary_threshold = threshold * 0.5

    boundary_mask = (fn_df["predicted_probability"] >= boundary_threshold) & (fn_df["predicted_probability"] < threshold)
    confident_mask = fn_df["predicted_probability"] < boundary_threshold

    boundary_errors = fn_df[boundary_mask].copy()
    confident_wrong = fn_df[confident_mask].copy()

    result = {
        "boundary_errors": boundary_errors,
        "confident_wrong": confident_wrong,
        "boundary_count": len(boundary_errors),
        "confident_count": len(confident_wrong),
        "boundary_pct": len(boundary_errors) / len(fn_df) * 100 if len(fn_df) > 0 else 0,
        "confident_pct": len(confident_wrong) / len(fn_df) * 100 if len(fn_df) > 0 else 0,
    }

    logger.info(
        f"False Negative segmentation: "
        f"Boundary errors: {result['boundary_count']} ({result['boundary_pct']:.1f}%) | "
        f"Confident wrong: {result['confident_count']} ({result['confident_pct']:.1f}%)"
    )
    return result


def calculate_segment_statistics(
    fn_df: pd.DataFrame,
    clinical_cols: list = None,
) -> dict:
    """
    Calculate summary statistics for False Negative segments.
    Helps identify patterns: Are missed patients different from caught ones?
    """
    if clinical_cols is None:
        clinical_cols = ["hemoglobin", "glucose", "wbc", "platelets", "creatinine"]

    stats = {
        "count": len(fn_df),
        "mean_probability": fn_df["predicted_probability"].mean() if len(fn_df) > 0 else None,
        "std_probability": fn_df["predicted_probability"].std() if len(fn_df) > 0 else None,
        "clinical_summary": {},
    }

    for col in clinical_cols:
        if col in fn_df.columns:
            stats["clinical_summary"][col] = {
                "mean": fn_df[col].mean(),
                "std": fn_df[col].std(),
                "min": fn_df[col].min(),
                "max": fn_df[col].max(),
            }

    return stats


def generate_error_report(
    fn_df: pd.DataFrame,
    segmentation: dict,
    threshold: float,
    recall_target: float,
    total_high_risk: int,
) -> str:
    """
    Generate a structured error analysis report.

    The output is NOT just a report – it's a prioritized list of system improvements.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ERROR ANALYSIS REPORT – FALSE NEGATIVE DIAGNOSIS")
    report_lines.append("=" * 60)

    report_lines.append(f"\n1. PROBLEM QUANTIFICATION")
    report_lines.append(f"   Total high-risk patients in test: {total_high_risk}")
    report_lines.append(f"   False Negatives (missed): {len(fn_df)}")
    missed_pct = len(fn_df) / total_high_risk * 100 if total_high_risk > 0 else 0
    report_lines.append(f"   Missed percentage: {missed_pct:.1f}%")
    report_lines.append(f"   Recall target: {recall_target}")
    report_lines.append(f"   Production threshold: {threshold:.4f}")

    report_lines.append(f"\n2. FALSE NEGATIVE SEGMENTATION")
    report_lines.append(f"   Boundary errors (uncertain): {segmentation['boundary_count']} ({segmentation['boundary_pct']:.1f}%)")
    report_lines.append(f"   Confident wrong: {segmentation['confident_count']} ({segmentation['confident_pct']:.1f}%)")

    if segmentation["boundary_count"] > segmentation["confident_count"]:
        report_lines.append(f"\n   → DIAGNOSIS: MOSTLY BOUNDARY ERRORS")
        report_lines.append(f"   → PRESCRIPTION: Lower threshold or tune on different metric")
    else:
        report_lines.append(f"\n   → DIAGNOSIS: SIGNIFICANT CONFIDENT WRONG PREDICTIONS")
        report_lines.append(f"   → PRESCRIPTION: Add features or collect more training data")

    # Confidence Score Paradox section
    report_lines.append(f"\n3. CONFIDENCE SCORE PARADOX")
    if len(fn_df) > 0:
        report_lines.append(f"   Mean predicted probability of FNs: {fn_df['predicted_probability'].mean():.4f}")
    else:
        report_lines.append(f"   No FNs to analyze")
    report_lines.append(f"   High confidence score ≠ correct prediction.")
    report_lines.append(f"   Confidence score validates input integrity, not predictive completeness.")

    # Clinical pattern summary
    if len(fn_df) > 0:
        report_lines.append(f"\n4. CLINICAL PATTERNS (Missed Patients)")
        for col in ["hemoglobin", "glucose", "creatinine"]:
            if col in fn_df.columns:
                mean_val = fn_df[col].mean()
                report_lines.append(f"   {col}: mean = {mean_val:.2f}")

    report_lines.append(f"\n5. PRESCRIPTION")
    report_lines.append(f"   Based on analysis, prioritize:")
    if segmentation["boundary_count"] > 0:
        report_lines.append(f"   - [ ] Re-tune decision threshold (targeting boundary errors)")
    if segmentation["confident_count"] > 0:
        report_lines.append(f"   - [ ] Add interaction features (targeting confident wrong)")
    report_lines.append(f"   - [ ] Review clinical features for additional signals")

    report_lines.append("\n" + "=" * 60)
    report_lines.append("END OF ERROR ANALYSIS")
    report_lines.append("=" * 60)

    report = "\n".join(report_lines)
    logger.info(report)
    return report


def run_error_analysis(
    model_path: str = None,
    threshold_path: str = None,
    test_path: str = None,
) -> dict:
    """
    Complete error analysis pipeline.
    """
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # Load artifacts and data
    model, threshold, threshold_artifact = load_production_artifacts(
        model_path, threshold_path
    )
    X_test, y_test = load_feature_store_test_data(test_path)

    # Predict
    y_pred = predict_with_threshold(model, X_test, threshold)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    total_high_risk = (y_test == 1).sum()
    fn_df = extract_false_negatives(y_test, y_pred, y_pred_proba, X_test, threshold)

    if fn_df.empty:
        logger.info("No False Negatives to analyze – model meets clinical safety target")
        return {"fn_count": 0, "fn_df": fn_df}

    segmentation = segment_false_negatives(fn_df, threshold)
    boundary_stats = calculate_segment_statistics(segmentation["boundary_errors"])
    confident_stats = calculate_segment_statistics(segmentation["confident_wrong"])

    report = generate_error_report(
        fn_df, segmentation, threshold,
        threshold_artifact["recall_target"],
        total_high_risk
    )

    # Save FN records for inspection
    output_path = "models/artifacts/false_negatives.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fn_df.to_csv(output_path, index=False)
    logger.info(f"False Negative records saved to {output_path}")

    return {
        "fn_count": len(fn_df),
        "fn_df": fn_df,
        "segmentation": segmentation,
        "boundary_stats": boundary_stats,
        "confident_stats": confident_stats,
        "report": report,
    }


if __name__ == "__main__":
    results = run_error_analysis()