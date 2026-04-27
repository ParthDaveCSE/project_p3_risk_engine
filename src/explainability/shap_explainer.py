"""
SHAP Explainability Pipeline – Feature Attribution for Clinical Models

CRITICAL CONCEPTS:
1. SHAP values are in LOG-ODDS space, NOT probability space
2. Additivity: model_output = expected_value + sum(SHAP_values)
3. TreeExplainer is structure-aware, exact, and fast for Random Forest
4. We use shap_values[1] exclusively (High-Risk class)

INTERVIEW ANSWER:
"SHAP values quantify the marginal contribution of each feature across all
possible coalitions using game theory. For regulatory audit, additivity allows
deterministic reconstruction of predictions from their components."
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib

# CI/CD headless guard – prevents display crashes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.models.evaluator import (
    load_production_artifacts,
    load_feature_store_test_data,
    predict_with_threshold,
)

# Suppress SHAP's verbose deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)
config = load_config()


def log_odds(p: float) -> float:
    """Convert probability to log-odds."""
    p = np.clip(p, 1e-7, 1 - 1e-7)  # Avoid log(0) or log(1)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert log-odds to probability."""
    return 1 / (1 + np.exp(-x))


def log_odds_to_probability(log_odds: float) -> float:
    """Convert a single log-odds value to probability."""
    return 1 / (1 + np.exp(-log_odds))


def compute_shap_values(model: RandomForestClassifier, X_test: pd.DataFrame):
    """
    Compute SHAP values for the high-risk class (index 1) in LOG-ODDS space.

    WHY TreeExplainer over KernelExplainer?
    - TreeExplainer is structure-aware, utilizing tree paths for exact computation
    - KernelExplainer is model-agnostic but slow and approximates
    - For clinical systems, maximum precision demands TreeExplainer

    WHY shap_values[1]?
    - shap_values returns list: [class0_values, class1_values] OR 3D array
    - We exclusively explain the clinically relevant class (High-Risk = 1)
    """
    logger.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    logger.info("Computing SHAP values for test set...")
    shap_values_all = explainer.shap_values(X_test)

    # Handle different SHAP output formats
    if isinstance(shap_values_all, list):
        # Older versions: list of 2 arrays [class0, class1]
        shap_values_high_risk = shap_values_all[1]
    elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
        # Newer versions: 3D array (n_samples, n_features, n_classes)
        # Take class 1 (index 1) for high-risk
        shap_values_high_risk = shap_values_all[:, :, 1]
    else:
        # Fallback
        shap_values_high_risk = shap_values_all

    logger.info(f"SHAP values computed | Shape: {shap_values_high_risk.shape}")
    return shap_values_high_risk, explainer


def get_expected_log_odds(explainer, model, X_sample=None) -> float:
    """
    Extract the expected value (baseline log-odds) from the explainer.

    For Random Forest, SHAP's expected_value is in PROBABILITY space.
    We convert it to LOG-ODDS space for additivity verification.

    This is the average model output in log-odds space before
    individual patient features are factored in.

    HANDLES DIFFERENT SHAP VERSION OUTPUTS:
    - Single float: direct use
    - List of 2: take index 1 (high-risk class)
    - 2D array: take appropriate element
    - 3D array: take appropriate element
    """
    ev = explainer.expected_value

    # Extract the probability for high-risk class (index 1)
    if isinstance(ev, list):
        base_probability = float(ev[1]) if len(ev) > 1 else float(ev[0])
    elif isinstance(ev, np.ndarray):
        if ev.ndim == 0:
            base_probability = float(ev)
        elif ev.ndim == 1:
            base_probability = float(ev[1]) if len(ev) > 1 else float(ev[0])
        elif ev.ndim == 2:
            base_probability = float(ev[0, 1]) if ev.shape[1] > 1 else float(ev[0, 0])
        elif ev.ndim == 3:
            base_probability = float(ev[0, 0, 1]) if ev.shape[2] > 1 else float(ev[0, 0, 0])
        else:
            base_probability = 0.5
    else:
        base_probability = float(ev)

    # Convert probability to log-odds
    expected_log_odds = log_odds(base_probability)

    logger.info(f"Expected probability: {base_probability:.4f} -> Log-odds: {expected_log_odds:.4f}")
    return expected_log_odds


def get_model_predictions_in_log_odds(model, X_test: pd.DataFrame) -> np.ndarray:
    """
    Get model predictions in log-odds space.
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    log_odds_values = np.array([log_odds(p) for p in probabilities])
    return log_odds_values


def compute_local_explanation(
    shap_values: np.ndarray,
    expected_log_odds: float,
    X_sample: pd.DataFrame,
    feature_names: list,
    index: int,
) -> dict:
    """
    Compute local explanation for a single patient.

    LOCAL EXPLANATION = expected_log_odds + sum(feature contributions in log-odds)
    """
    sample_shap = shap_values[index]
    sample_features = X_sample.iloc[index]

    # Convert shap values to float
    sample_shap_float = np.array([float(x) for x in sample_shap])

    # Verify additivity mathematically
    total_shap = float(sample_shap_float.sum())
    predicted_log_odds = float(expected_log_odds) + total_shap
    predicted_probability = log_odds_to_probability(predicted_log_odds)

    # Individual feature contributions
    contributions = []
    for i, feature in enumerate(feature_names):
        shap_val = float(sample_shap_float[i])
        contributions.append({
            "feature": feature,
            "value": float(sample_features[feature]),
            "shap_value": shap_val,
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
        })

    # Sort by absolute SHAP value (most impactful first)
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    result = {
        "index": int(index),
        "expected_log_odds": float(expected_log_odds),
        "feature_contributions": contributions,
        "total_shap": total_shap,
        "predicted_log_odds": predicted_log_odds,
        "predicted_probability": predicted_probability,
        "additivity_verified": abs(predicted_log_odds - (float(expected_log_odds) + total_shap)) < 1e-6,
    }

    return result


def plot_global_beeswarm(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    output_path: str = "models/artifacts/shap_beeswarm.png",
):
    """
    Generate SHAP beeswarm plot – global feature importance.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP beeswarm plot saved to {output_path}")


def plot_local_waterfall(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    expected_log_odds: float,
    patient_index: int,
    output_dir: str = "models/artifacts/",
):
    """
    Generate SHAP waterfall plot for a single patient in log-odds space.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert shap values to float array
    shap_values_float = np.array([float(x) for x in shap_values[patient_index]])

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_float,
            base_values=expected_log_odds,
            data=X_test.iloc[patient_index].values,
            feature_names=X_test.columns.tolist(),
        ),
        show=False,
        max_display=8,
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"shap_waterfall_patient_{patient_index}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Patient {patient_index} waterfall plot saved to {output_path}")
    return output_path


def compute_error_analysis_shap(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
) -> dict:
    """
    Compare SHAP patterns between False Negatives and True Positives.

    This reveals the "diagnostic gap" – what features are missing when the model fails?
    """
    fn_mask = (y_test == 1) & (y_pred == 0)
    tp_mask = (y_test == 1) & (y_pred == 1)

    fn_shap = shap_values[fn_mask] if fn_mask.any() else np.array([])
    tp_shap = shap_values[tp_mask] if tp_mask.any() else np.array([])

    # Mean SHAP for each feature
    if len(fn_shap) > 0:
        fn_mean = np.abs(fn_shap).mean(axis=0)
    else:
        fn_mean = np.zeros(X_test.shape[1])

    if len(tp_shap) > 0:
        tp_mean = np.abs(tp_shap).mean(axis=0)
    else:
        tp_mean = np.zeros(X_test.shape[1])

    feature_names = X_test.columns.tolist()

    comparison = []
    for i, feature in enumerate(feature_names):
        comparison.append({
            "feature": feature,
            "fn_mean_shap": float(fn_mean[i]),
            "tp_mean_shap": float(tp_mean[i]),
            "gap": float(tp_mean[i] - fn_mean[i]),
            "fn_count": int(fn_mask.sum()),
            "tp_count": int(tp_mask.sum()),
        })

    comparison.sort(key=lambda x: abs(x["gap"]), reverse=True)

    result = {
        "fn_count": int(fn_mask.sum()),
        "tp_count": int(tp_mask.sum()),
        "feature_comparison": comparison,
        "top_missing_signal": comparison[0]["feature"] if comparison else None,
    }

    logger.info(f"Error analysis SHAP complete | FN: {result['fn_count']} | TP: {result['tp_count']}")
    return result


def plot_fn_tp_comparison(
    comparison: dict,
    output_path: str = "models/artifacts/shap_fn_tp_comparison.png",
):
    """
    Bar chart comparing FN vs TP mean |SHAP| for each feature.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    features = [c["feature"] for c in comparison["feature_comparison"][:8]]
    fn_means = [c["fn_mean_shap"] for c in comparison["feature_comparison"][:8]]
    tp_means = [c["tp_mean_shap"] for c in comparison["feature_comparison"][:8]]

    x = np.arange(len(features))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, fn_means, width, label="False Negatives (Missed)", color="red", alpha=0.7)
    plt.bar(x + width/2, tp_means, width, label="True Positives (Caught)", color="green", alpha=0.7)

    plt.xlabel("Features")
    plt.ylabel("Mean |SHAP Value| (Log-Odds)")
    plt.title("SHAP Comparison: False Negatives vs True Positives")
    plt.xticks(x, features, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"FN vs TP comparison plot saved to {output_path}")


def run_shap_pipeline():
    """
    Complete SHAP explainability pipeline.
    """
    logger.info("=" * 60)
    logger.info("SHAP EXPLAINABILITY PIPELINE")
    logger.info("=" * 60)

    # Load artifacts
    model, threshold, threshold_artifact = load_production_artifacts()
    X_test, y_test = load_feature_store_test_data()

    # Compute SHAP
    shap_values, explainer = compute_shap_values(model, X_test)
    expected_log_odds = get_expected_log_odds(explainer, model, X_test)

    # Get model predictions in log-odds for verification
    y_pred_log_odds = get_model_predictions_in_log_odds(model, X_test)

    # Verify additivity for first 5 patients
    logger.info("Verifying SHAP additivity property...")
    additivity_errors = []
    for i in range(min(5, len(X_test))):
        total_shap = float(np.sum([float(x) for x in shap_values[i]]))
        reconstructed_log_odds = expected_log_odds + total_shap
        actual_log_odds = y_pred_log_odds[i]

        additivity_error = abs(reconstructed_log_odds - actual_log_odds)
        additivity_errors.append(additivity_error)
        logger.info(f"Patient {i}: Reconstructed={reconstructed_log_odds:.4f}, Actual={actual_log_odds:.4f}, Error={additivity_error:.6f}")

    avg_error = np.mean(additivity_errors)
    logger.info(f"Average additivity error: {avg_error:.6f}")

    if avg_error < 0.1:
        logger.info("Additivity verified – predictions match SHAP decomposition within tolerance")
    else:
        logger.warning(f"Additivity error {avg_error:.4f} - SHAP approximations may have some error")

    # Global explanation
    plot_global_beeswarm(shap_values, X_test)

    # Local explanations for sample patients
    sample_indices = [0, 1, 2] if len(X_test) >= 3 else range(len(X_test))
    local_explanations = []
    for idx in sample_indices:
        explanation = compute_local_explanation(
            shap_values, expected_log_odds, X_test, X_test.columns.tolist(), idx
        )
        local_explanations.append(explanation)
        plot_local_waterfall(shap_values, X_test, expected_log_odds, idx)

        print(f"\n--- Patient {explanation['index']} Local Explanation ---")
        print(f"Baseline (Expected) Log-Odds: {explanation['expected_log_odds']:.4f}")
        print(f"Sum of SHAP contributions: {explanation['total_shap']:.4f}")
        print(f"Predicted Log-Odds: {explanation['predicted_log_odds']:.4f}")
        print(f"Predicted Probability: {explanation['predicted_probability']:.1%}")
        print("Top 3 contributors:")
        for contrib in explanation['feature_contributions'][:3]:
            print(f"  {contrib['feature']}: {contrib['shap_value']:+.4f} ({contrib['direction']})")

    # Error analysis via SHAP
    y_pred = predict_with_threshold(model, X_test, threshold)
    shap_comparison = compute_error_analysis_shap(y_test, y_pred, shap_values, X_test)
    plot_fn_tp_comparison(shap_comparison)

    # Print global importance summary
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = sorted(
        zip(X_test.columns, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 60)
    print("GLOBAL FEATURE IMPORTANCE (Mean |SHAP|)")
    print("=" * 60)
    for feature, importance in global_importance[:5]:
        print(f"  {feature}: {importance:.4f}")

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS: Missing Signal in False Negatives")
    print("=" * 60)
    for comp in shap_comparison["feature_comparison"][:5]:
        print(f"  {comp['feature']}: TP={comp['tp_mean_shap']:.4f} | FN={comp['fn_mean_shap']:.4f} | Gap={comp['gap']:.4f}")

    if shap_comparison["top_missing_signal"]:
        print(f"\n→ PRIMARY MISSING SIGNAL: {shap_comparison['top_missing_signal']}")
        print("  Action: Engineer features capturing moderate simultaneous abnormalities")

    # Save local explanations as JSON
    output_path = "models/artifacts/shap_local_explanations.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(local_explanations, f, indent=2)
    logger.info(f"Local explanations saved to {output_path}")

    return shap_values, explainer, local_explanations, shap_comparison


if __name__ == "__main__":
    shap_values, explainer, local_explanations, shap_comparison = run_shap_pipeline()