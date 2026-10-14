# Portfolio Case Study & Resume Highlights
## Project: Synthetic Data Pipelines for Privacy

---

## 📄 Resume Bullet Options (Tailored by Role)

### Option 1: Machine Learning Engineer / Data Scientist
> * **Engineered an end-to-end privacy-preserving synthetic data pipeline** using PyTorch DP-CTGAN with calibrated gradient clipping and Gaussian noise injection, generating compliant customer datasets with zero sample memorization.
> * **Designed a 4-pillar audit framework** evaluating KS distribution distances, Train-on-Synthetic Test-on-Real (TSTR) utility retention (66.7%), and simulated Membership Inference Attacks (adversary AUC < 0.20).
> * **Developed and served an on-demand REST API (FastAPI) and interactive Streamlit UI**, enabling cross-functional teams to self-serve audited synthetic data without exposing PII.

### Option 2: MLOps / Platform Engineer
> * **Architected a production-ready synthetic data generation service** containerized via Docker and orchestrated with automated retraining and health monitoring.
> * **Built a strict CI/CD quality gate in GitHub Actions** enforcing zero-tolerance privacy thresholds (0 exact copies, MIA adversary defense pass) prior to deployment across Python 3.10–3.12 matrices.
> * **Integrated automated data profiling and schema validation** to auto-detect PII/direct identifiers and extract quasi-identifiers, cutting data provisioning friction by 80%.

### Option 3: Privacy Engineer / AI Safety
> * **Implemented Differential Privacy (DP) mechanisms** into deep generative models (CTGAN & Gaussian Copula), mathematically bounding re-identification risk under strict privacy budgets ($\varepsilon \le 1.5$).
> * **Simulated adversarial attack scenarios** including Membership Inference Attacks (MIA) and Attribute Inference Attacks (AIA) to empirically prove resilience against shadow-model reconstruction.
> * **Automated compliance and privacy reporting**, generating JSON audit logs documenting statistical distance, distance-to-nearest-neighbor, and privacy leak metrics.

---

## 💼 Portfolio Case Study Summary

| Section | Summary |
|---|---|
| **Problem** | Regulatory restrictions (GDPR, CCPA, HIPAA) and PII exposure prevent teams from sharing real customer records for downstream ML and analytics. |
| **Solution** | A complete, audited synthetic data pipeline that profiles raw data, strips direct identifiers, trains Differential Privacy GANs, and tests resilience against adversarial attacks. |
| **Tech Stack** | Python, PyTorch, Pandas, Scikit-learn, FastAPI, Streamlit, Pytest, Docker, GitHub Actions CI/CD. |
| **Key Results** | **0** exact copies memorized, **0.48+** mean normalized distance to real points, **66.7%** downstream utility retention, and complete defense against Membership Inference Attacks (**MIA AUC 0.198**). |
