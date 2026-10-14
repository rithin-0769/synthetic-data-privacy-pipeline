"""End-to-end pipeline execution runner."""
import os
import json
import argparse
import pandas as pd

from src.profiling.profile import profile, save_report
from src.preprocessing.pipeline import clean_and_split_features
from src.models.generator import TabularSynthesizer
from src.validation.metrics import evaluate_statistical_similarity, evaluate_privacy_distance

def run_pipeline(input_csv: str, output_dir: str = "data/output", sample_size: int = 50, model_type: str = "copula"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/profile", exist_ok=True)
    
    print(f"[1/5] Loading and profiling dataset: {input_csv}")
    raw_df = pd.read_csv(input_csv)
    prof_report = profile(raw_df)
    save_report(prof_report, "data/profile/report.json")
    print(f"      Profile saved. Detected sensitive fields: {prof_report['sensitive']}")

    print("[2/5] Preprocessing and dropping direct identifiers...")
    processed_df, id_cols, cat_cols, num_cols = clean_and_split_features(raw_df, target_col="Churn")
    print(f"      Dropped direct identifiers: {id_cols}")

    print(f"[3/5] Training generative model ({model_type})...")
    synthesizer = TabularSynthesizer(model_type=model_type)
    synthesizer.fit(processed_df)

    print(f"[4/5] Generating {sample_size} synthetic records...")
    synth_df = synthesizer.sample(num_rows=sample_size)
    synth_path = os.path.join(output_dir, "synthetic_churn_data.csv")
    synth_df.to_csv(synth_path, index=False)
    print(f"      Synthetic data written to {synth_path}")

    print("[5/5] Running validation (Utility + Privacy checks)...")
    stat_metrics = evaluate_statistical_similarity(processed_df, synth_df)
    priv_metrics = evaluate_privacy_distance(processed_df, synth_df)
    
    print("      Running Downstream ML Utility Benchmark (TSTR vs TRTR)...")
    from src.validation.utility_benchmark import benchmark_tstr
    utility_metrics = benchmark_tstr(processed_df, synth_df, target_col="Churn")

    print("      Simulating Privacy Attacks (MIA & AIA)...")
    from src.attack.privacy_attacks import simulate_membership_inference, simulate_attribute_inference
    mia_results = simulate_membership_inference(processed_df, synth_df, target_col="Churn")
    aia_results = simulate_attribute_inference(processed_df, synth_df, sensitive_attr="MonthlySpend")

    eval_summary = {
        "statistical_similarity": stat_metrics,
        "privacy_metrics": priv_metrics,
        "downstream_ml_utility": utility_metrics,
        "privacy_attacks": {
            "membership_inference_attack": mia_results,
            "attribute_inference_attack": aia_results
        }
    }
    eval_path = os.path.join(output_dir, "evaluation_report.json")
    with open(eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("      Generating Compliance Audit Report (HTML/PDF)...")
    from src.validation.report_generator import generate_audit_html_report
    html_report_path = os.path.join(output_dir, "privacy_compliance_report.html")
    generate_audit_html_report(eval_path, "data/profile/report.json", html_report_path)

    print("\n=== PIPELINE RUN COMPLETE ===")
    print(f"Privacy disclosure distance (mean): {priv_metrics['mean_min_distance_to_real']:.4f}")
    print(f"Exact copies detected: {priv_metrics['exact_copies_detected']}")
    print(f"Downstream Accuracy Retention: {utility_metrics['utility_delta']['accuracy_retention_pct']}%")
    print(f"Membership Inference Adversary AUC: {mia_results['mia_adversary_auc']} (Passed: {mia_results['defense_passed']})")
    print(f"Evaluation report saved to {eval_path}")
    print(f"HTML Compliance Certificate: {html_report_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Privacy-Preserving Synthetic Pipeline")
    parser.add_argument("--input", default="data/raw/train.csv", help="Input raw CSV path")
    parser.add_argument("--output_dir", default="data/output", help="Output directory")
    parser.add_argument("--samples", type=int, default=50, help="Number of synthetic samples to generate")
    parser.add_argument("--model", default="copula", choices=["copula", "ctgan"], help="Synthesizer model")
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, sample_size=args.samples, model_type=args.model)
