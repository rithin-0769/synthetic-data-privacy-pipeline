"""Downstream ML Utility & Train-on-Synthetic Test-on-Real (TSTR) Benchmark.
Evaluates model performance drop when training on synthetic vs real data.
Uses PyTorch Logistic Regression / MLP for classifier robustness.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TabularClassifier(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def _prepare_tensors(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[torch.Tensor, torch.Tensor, list]:
    feature_cols = [c for c in df.columns if c != target_col and c.lower() != "customerid"]
    
    encoded_cols = []
    for c in feature_cols:
        if df[c].dtype == object or df[c].dtype.name == "category":
            # Binary or category map
            vals = sorted(df[c].dropna().unique())
            mapping = {v: idx for idx, v in enumerate(vals)}
            encoded = df[c].map(mapping).fillna(0).values.reshape(-1, 1)
            encoded_cols.append(encoded)
        else:
            c_min, c_max = float(df[c].min()), float(df[c].max())
            denom = c_max - c_min if c_max != c_min else 1.0
            normed = (df[c].values - c_min) / denom
            encoded_cols.append(normed.reshape(-1, 1))

    X_mat = np.hstack(encoded_cols).astype(np.float32)
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    y_arr = y_raw.map({"yes": 1.0, "1": 1.0, "true": 1.0, "no": 0.0, "0": 0.0, "false": 0.0}).fillna(0.0).values
    y_mat = y_arr.astype(np.float32).reshape(-1, 1)

    return torch.tensor(X_mat), torch.tensor(y_mat), feature_cols

def _train_and_evaluate(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, epochs: int = 150) -> Dict[str, float]:
    model = TabularClassifier(in_features=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        pred_labels = (test_preds >= 0.5).float()
        accuracy = (pred_labels == y_test).float().mean().item()
        
        # Calculate approximate ROC AUC
        y_true = y_test.numpy().flatten()
        y_score = test_preds.numpy().flatten()
        
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) > 0 and len(neg) > 0:
            auc = float(np.mean([p > n for p in pos for n in neg] + [0.5 * (p == n) for p in pos for n in neg]))
        else:
            auc = 0.5

        # F1 score calculation
        tp = float(((pred_labels == 1) & (y_test == 1)).sum())
        fp = float(((pred_labels == 1) & (y_test == 0)).sum())
        fn = float(((pred_labels == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"accuracy": float(accuracy), "roc_auc": float(auc), "f1_score": float(f1)}

def benchmark_tstr(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, target_col: str = "Churn", test_ratio: float = 0.3) -> Dict[str, Any]:
    """Runs Train-on-Real Test-on-Real (TRTR) vs Train-on-Synthetic Test-on-Real (TSTR)."""
    # Split real into train and test
    n_total = len(real_df)
    indices = np.random.RandomState(42).permutation(n_total)
    split_idx = max(1, int(n_total * (1 - test_ratio)))
    real_train = real_df.iloc[indices[:split_idx]].reset_index(drop=True)
    real_test = real_df.iloc[indices[split_idx:]].reset_index(drop=True)

    X_real_train, y_real_train, _ = _prepare_tensors(real_train, target_col)
    X_real_test, y_real_test, _ = _prepare_tensors(real_test, target_col)
    X_synth_train, y_synth_train, _ = _prepare_tensors(synthetic_df, target_col)

    # 1. TRTR (Train on Real, Test on Real) - baseline upper bound
    trtr_metrics = _train_and_evaluate(X_real_train, y_real_train, X_real_test, y_real_test)

    # 2. TSTR (Train on Synthetic, Test on Real) - utility retention test
    tstr_metrics = _train_and_evaluate(X_synth_train, y_synth_train, X_real_test, y_real_test)

    # Utility degradation delta
    delta_accuracy = trtr_metrics["accuracy"] - tstr_metrics["accuracy"]
    delta_auc = trtr_metrics["roc_auc"] - tstr_metrics["roc_auc"]
    retention_rate = (tstr_metrics["accuracy"] / max(trtr_metrics["accuracy"], 1e-4)) * 100.0

    return {
        "TRTR_baseline (Train Real -> Test Real)": trtr_metrics,
        "TSTR_evaluation (Train Synth -> Test Real)": tstr_metrics,
        "utility_delta": {
            "accuracy_drop": round(float(delta_accuracy), 4),
            "auc_drop": round(float(delta_auc), 4),
            "accuracy_retention_pct": round(float(retention_rate), 2)
        }
    }
