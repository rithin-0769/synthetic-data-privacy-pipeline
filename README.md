# Synthetic Data Pipelines for Privacy

Production-grade pipeline that generates high-quality synthetic datasets to replace or augment sensitive real data, preserving mathematical and empirical privacy while retaining downstream predictive utility.

---

## Architecture Overview

```
                      Raw Sensitive Data (data/raw/train.csv)
                                       │
                                       ▼
                     [1. Data Profiler & Sensitive Field Tagging]
                     (Correlation, missingness, schema & direct IDs)
                                       │
                                       ▼
                     [2. Anonymization & Feature Preprocessor]
                     (Drop CustomerID, One-Hot / Min-Max scalers)
                                       │
                                       ▼
                     [3. Generative Modeling Engine]
                     (DP-CTGAN with Gradient Perturbation / Copula)
                                       │
                                       ▼
                     [4. Multi-Pillar Validation Suite]
                     ├── A. Statistical Similarity (KS-Test)
                     ├── B. Empirical Privacy (Nearest Neighbor Distance)
                     ├── C. Downstream ML Utility (TSTR vs TRTR)
                     └── D. Privacy Attack Simulator (Membership & Attribute)
                                       │
                                       ▼
           Synthetic Dataset + JSON Audit Report (data/output/)
                                       │
                                       ▼
                     [5. Serving & Orchestration Layer]
                     ├── FastAPI Microservice (src/service/api.py)
                     └── Periodic Orchestration (src/orchestration/scheduler.py)
```

---

## Quickstart

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Executes profiling, anonymization, DP-CTGAN training, synthetic sampling, and the full 4-pillar privacy/utility validation:
```powershell
.venv\Scripts\python.exe -m src.run_pipeline --input data/raw/train.csv --output_dir data/output --samples 30 --model ctgan
```

### 3. Launch the Interactive Dashboard (Streamlit UI)
Inspect synthetic data, compare marginal distributions, and audit privacy/utility metrics visually:
```powershell
.venv\Scripts\python.exe -m streamlit run src/ui/app.py
```

### 4. Launch the FastAPI Microservice
Starts the REST API server:
```powershell
.venv\Scripts\python.exe -m uvicorn src.service.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs: `http://localhost:8000/docs`

Available endpoints:
* `POST /generate`: Synthesizes $N$ records with Differential Privacy budget parameter $\varepsilon$.
* `GET /evaluation`: Retrieves current evaluation metrics, nearest-neighbor distance, and attack defenses.
* `GET /profile`: Returns feature correlations, missingness, and detected sensitive columns.
* `GET /health`: Microservice health status.

### 5. Run Scheduled Retraining / Pipeline Orchestration
```powershell
.venv\Scripts\python.exe -m src.orchestration.scheduler --once
```

### 6. Run Automated Test Suite
Executes all 13 unit, integration, and security tests:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Automated CI/CD Workflow (`.github/workflows/ci.yml`)

The repository includes a GitHub Actions CI matrix (`Python 3.10, 3.11, 3.12`):
1. Runs full `pytest` suite across API and synthesis components.
2. Executes end-to-end DP-CTGAN training and synthetic generation.
3. Enforces an **Automated Privacy Gate**:
   * Asserts `exact_copies_detected == 0` (build fails if any record is memorized).
   * Asserts `membership_inference_attack.defense_passed == True` (build fails if MIA vulnerability is detected).
4. Packages and uploads generated synthetic samples and privacy audit JSONs as downloadable CI artifacts.

---

## Evaluation & Privacy Metrics

The generated audit report is saved to [`data/output/evaluation_report.json`](file:///c:/Users/Rithin%20Ravoori/Downloads/prac/project1/data/output/evaluation_report.json):
* **Direct Identifiers Stripped**: `CustomerID` removed automatically before training.
* **Exact Copies Detected**: `0` (no sample memorization).
* **Mean Nearest-Neighbor Distance**: `> 0.48` normalized distance from all real records.
* **Downstream ML Retention (TSTR)**: High predictive utility retained relative to the real-data baseline.
* **Membership Inference Defense**: Attack AUC $< 0.50$ (adversary cannot infer membership).
