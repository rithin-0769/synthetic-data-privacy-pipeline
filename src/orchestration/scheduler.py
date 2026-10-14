"""Automated Pipeline Orchestration & Scheduling Task.
Supports running end-to-end retraining, profiling, auditing, and packaging periodically.
"""
import time
import argparse
from datetime import datetime
from src.run_pipeline import run_pipeline

def scheduled_job(interval_hours: int = 24, run_once: bool = True):
    print(f"[{datetime.now().isoformat()}] Starting scheduled synthetic pipeline run...")
    run_pipeline(
        input_csv="data/raw/train.csv",
        output_dir="data/output",
        sample_size=30,
        model_type="ctgan"
    )
    print(f"[{datetime.now().isoformat()}] Pipeline run completed successfully.")
    
    if not run_once:
        print(f"Sleeping for {interval_hours} hours until next trigger...")
        while True:
            time.sleep(interval_hours * 3600)
            print(f"[{datetime.now().isoformat()}] Triggering scheduled synthetic retraining...")
            run_pipeline("data/raw/train.csv", "data/output", sample_size=30, model_type="ctgan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator Scheduler")
    parser.add_argument("--interval", type=int, default=24, help="Retraining interval in hours")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit")
    args = parser.parse_args()
    scheduled_job(interval_hours=args.interval, run_once=args.once)
