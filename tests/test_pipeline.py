"""Unit and integration tests for Synthetic Data Privacy Pipeline."""
import os
import pytest
import pandas as pd
import numpy as np

from src.profiling.profile import profile
from src.preprocessing.pipeline import clean_and_split_features, identify_columns
from src.models.generator import TabularSynthesizer, DPNoiseMechanism
from src.validation.metrics import evaluate_statistical_similarity, evaluate_privacy_distance
from src.validation.utility_benchmark import benchmark_tstr
from src.attack.privacy_attacks import simulate_membership_inference, simulate_attribute_inference

@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "CustomerID": [101, 102, 103, 104, 105, 106, 107, 108],
        "Age": [22, 45, 33, 50, 28, 41, 36, 55],
        "Gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "MonthlySpend": [30.0, 85.5, 45.0, 110.0, 39.0, 78.0, 52.0, 95.0],
        "TenureMonths": [6, 36, 12, 48, 8, 24, 18, 40],
        "SupportTickets": [1, 4, 0, 5, 2, 3, 1, 6],
        "Churn": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"]
    })

def test_profiling_detects_sensitive_fields(sample_raw_df):
    report = profile(sample_raw_df)
    assert "CustomerID" in report["sensitive"]
    assert "columns" in report
    assert "correlation" in report
    assert len(report["columns"]) == 7

def test_preprocessing_strips_identifiers(sample_raw_df):
    clean_df, dropped_ids, cat_cols, num_cols = clean_and_split_features(sample_raw_df, target_col="Churn")
    assert "CustomerID" in dropped_ids
    assert "CustomerID" not in clean_df.columns
    assert "Gender" in cat_cols
    assert "Age" in num_cols

def test_dp_noise_mechanism():
    dp = DPNoiseMechanism(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    assert dp.sigma > 0
    import torch
    dummy_tensor = torch.zeros(10, 10)
    perturbed = dp.add_noise(dummy_tensor)
    assert not torch.equal(dummy_tensor, perturbed)

def test_copula_synthesizer_generates_samples(sample_raw_df):
    clean_df, _, _, _ = clean_and_split_features(sample_raw_df, target_col="Churn")
    model = TabularSynthesizer(model_type="copula")
    model.fit(clean_df)
    sampled = model.sample(num_rows=15)
    assert len(sampled) == 15
    assert set(sampled.columns) == set(clean_df.columns)
    assert set(sampled["Churn"].unique()).issubset({"No", "Yes"})

def test_ctgan_synthesizer_generates_samples(sample_raw_df):
    clean_df, _, _, _ = clean_and_split_features(sample_raw_df, target_col="Churn")
    model = TabularSynthesizer(model_type="ctgan", epochs=10, epsilon=2.0)
    model.fit(clean_df)
    sampled = model.sample(num_rows=10)
    assert len(sampled) == 10
    assert set(sampled.columns) == set(clean_df.columns)

def test_privacy_metrics_detects_no_exact_copies(sample_raw_df):
    clean_df, _, _, _ = clean_and_split_features(sample_raw_df, target_col="Churn")
    model = TabularSynthesizer(model_type="copula")
    model.fit(clean_df)
    synth = model.sample(num_rows=20)
    
    priv = evaluate_privacy_distance(clean_df, synth)
    assert "min_distance_to_real" in priv
    assert "exact_copies_detected" in priv

def test_tstr_utility_benchmark(sample_raw_df):
    clean_df, _, _, _ = clean_and_split_features(sample_raw_df, target_col="Churn")
    model = TabularSynthesizer(model_type="copula")
    model.fit(clean_df)
    synth = model.sample(num_rows=25)
    
    bench = benchmark_tstr(clean_df, synth, target_col="Churn")
    assert "TRTR_baseline (Train Real -> Test Real)" in bench
    assert "TSTR_evaluation (Train Synth -> Test Real)" in bench
    assert "utility_delta" in bench

def test_membership_inference_simulation(sample_raw_df):
    clean_df, _, _, _ = clean_and_split_features(sample_raw_df, target_col="Churn")
    model = TabularSynthesizer(model_type="copula")
    model.fit(clean_df)
    synth = model.sample(num_rows=20)
    
    mia = simulate_membership_inference(clean_df, synth, target_col="Churn")
    assert "mia_adversary_auc" in mia
    assert "defense_passed" in mia
    assert isinstance(mia["defense_passed"], bool)

def test_attribute_inference_simulation(sample_raw_df):
    clean_df, _, _, _ = clean_and_split_features(sample_raw_df, target_col="Churn")
    synth = clean_df.copy()
    aia = simulate_attribute_inference(clean_df, synth, sensitive_attr="MonthlySpend")
    assert aia["macro_deviation_pct"] == 0.0
    assert aia["individual_attribute_protected"] is True
