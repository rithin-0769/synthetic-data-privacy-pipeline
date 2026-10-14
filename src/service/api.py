"""FastAPI Microservice for on-demand privacy-preserving synthetic data generation & auditing."""
import os
import json
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from src.models.generator import TabularSynthesizer
from src.preprocessing.pipeline import clean_and_split_features
from src.profiling.profile import profile

app = FastAPI(
    title="Privacy-Preserving Synthetic Data Service",
    description="API to generate audited synthetic datasets and run privacy/utility checks on demand.",
    version="1.0.0"
)

RAW_DATA_PATH = "data/raw/train.csv"
OUTPUT_DIR = "data/output"

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Synthetic Data Privacy Pipeline API",
        "endpoints": ["/generate", "/evaluation", "/profile", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/profile")
def get_profile():
    """Returns dataset profiling analysis."""
    if not os.path.exists(RAW_DATA_PATH):
        raise HTTPException(status_code=404, detail="Raw dataset not found")
    df = pd.read_csv(RAW_DATA_PATH)
    return profile(df)

@app.post("/generate")
def generate_synthetic_data(
    samples: int = Query(25, ge=5, le=1000, description="Number of synthetic records to generate"),
    model_type: str = Query("ctgan", pattern="^(ctgan|copula)$", description="Generative model algorithm"),
    epsilon: float = Query(1.5, ge=0.1, le=10.0, description="Differential Privacy budget epsilon")
):
    """Generates synthetic records with privacy preservation."""
    if not os.path.exists(RAW_DATA_PATH):
        raise HTTPException(status_code=404, detail="Raw dataset not found")

    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df, dropped_ids, _, _ = clean_and_split_features(raw_df, target_col="Churn")

    synthesizer = TabularSynthesizer(model_type=model_type, epsilon=epsilon)
    synthesizer.fit(clean_df)
    synth_df = synthesizer.sample(num_rows=samples)

    return {
        "model": model_type,
        "privacy_budget_epsilon": epsilon,
        "dropped_direct_identifiers": dropped_ids,
        "sample_count": len(synth_df),
        "records": synth_df.to_dict(orient="records")
    }

@app.get("/evaluation")
def get_latest_evaluation():
    """Retrieves the latest privacy, utility, and attack simulation evaluation report."""
    eval_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="No evaluation report found. Run pipeline first.")
    
    with open(eval_path, "r") as f:
        return json.load(f)
