"""
Model Comparison Helper
"""

import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()


def compare_models(metrics_list: list) -> pd.DataFrame:
    """Side-by-side comparison of models."""
    if not metrics_list:
        logger.warning("compare_models called with empty metrics list")
        return pd.DataFrame()

    df = pd.DataFrame(metrics_list)
    if 'model_name' in df.columns:
        df = df.set_index('model_name')

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(df.to_string())

    if 'recall' in df.columns:
        best_name = df['recall'].idxmax() if not df['recall'].isna().all() else 'N/A'
        best_recall = df['recall'].max()
        print(f"\nBest recall: {best_name} — {best_recall:.4f}" if best_recall is not None else "\nBest recall: N/A")

    return df