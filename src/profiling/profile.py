"""Data profiling utilities.
Computes marginal distributions, correlations, missingness, and identifies
sensitive columns based on a configurable list.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

SENSITIVE_FIELDS = ["ssn", "name", "email", "phone", "address", "customerid"]

def profile(df: pd.DataFrame) -> dict:
    """Return a profiling report dictionary for the given DataFrame."""
    report = {}
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing"] = df.isnull().mean().to_dict()
    
    # Calculate Spearman correlation only on numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr(method="spearman").fillna(0).to_dict()
    else:
        corr = {}
    report["correlation"] = corr

    # Histograms for numeric columns
    hist = {}
    for col in numeric_df.columns:
        counts, bins = np.histogram(numeric_df[col].dropna(), bins=10)
        hist[col] = {"bins": bins.tolist(), "counts": counts.tolist()}
    report["histograms"] = hist

    # Sensitive column detection
    report["sensitive"] = [c for c in df.columns if c.lower() in SENSITIVE_FIELDS]
    return report

def save_report(report: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile a CSV dataset")
    parser.add_argument("input_csv", type=str, help="Path to raw CSV file")
    parser.add_argument("output_json", type=str, help="Where to write the profile report")
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    prof = profile(df)
    save_report(prof, Path(args.output_json))
    print(f"Profile saved to {args.output_json}")
