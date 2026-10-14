"""Validation metrics: Kolmogorov-Smirnov test for distribution alignment,
and nearest-neighbor privacy disclosure distance.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def evaluate_statistical_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes KS-test statistic and p-values across continuous features."""
    results = {}
    num_cols = real_df.select_dtypes(include=[np.number]).columns.intersection(synth_df.columns)
    
    for col in num_cols:
        r_vals = real_df[col].dropna()
        s_vals = synth_df[col].dropna()
        stat, pval = ks_2samp(r_vals, s_vals)
        results[col] = {
            "ks_statistic": float(stat),
            "p_value": float(pval),
            "passed": bool(pval > 0.05)
        }
    return results

def evaluate_privacy_distance(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, float]:
    """Measures minimum normalized Euclidean distance from synthetic points to real points to detect copying."""
    num_cols = real_df.select_dtypes(include=[np.number]).columns.intersection(synth_df.columns)
    if len(num_cols) == 0:
        return {"min_distance": 0.0, "mean_min_distance": 0.0}

    real_norm = (real_df[num_cols] - real_df[num_cols].min()) / (real_df[num_cols].max() - real_df[num_cols].min() + 1e-6)
    synth_norm = (synth_df[num_cols] - real_df[num_cols].min()) / (real_df[num_cols].max() - real_df[num_cols].min() + 1e-6)

    min_dists = []
    for _, s_row in synth_norm.iterrows():
        dists = np.linalg.norm(real_norm.values - s_row.values, axis=1)
        min_dists.append(np.min(dists))

    return {
        "min_distance_to_real": float(np.min(min_dists)),
        "mean_min_distance_to_real": float(np.mean(min_dists)),
        "exact_copies_detected": int(np.sum(np.array(min_dists) < 1e-5))
    }
