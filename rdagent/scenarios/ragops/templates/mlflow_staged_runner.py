#!/usr/bin/env python3
"""MLflow integration for staged RAG evaluation.

This script demonstrates how to use the staged evaluation script
with MLflow to log checkpoint metrics with step parameter.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import mlflow
from mlflow import log_metric, log_param, log_artifact


def run_checkpoint_evaluation(
    checkpoint: int,
    resume_state: Optional[str] = None,
    documents_dir: Optional[str] = None,
    test_data: Optional[str] = None,
    baseline_metrics: Optional[str] = None
) -> Dict[str, Any]:
    """Run evaluation at a specific checkpoint."""
    cmd = [
        sys.executable,
        "evaluate_rag_staged.py",
        "--checkpoint-docs", str(checkpoint)
    ]
    
    if resume_state:
        cmd.extend(["--resume-from-state", resume_state])
    if documents_dir:
        cmd.extend(["--documents-dir", documents_dir])
    if test_data:
        cmd.extend(["--test-data", test_data])
    if baseline_metrics:
        cmd.extend(["--baseline-metrics", baseline_metrics])
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr}")
    
    # Load checkpoint results
    results_file = f"checkpoint_{checkpoint}_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    
    return {}


def log_checkpoint_metrics(metrics: Dict[str, float], checkpoint: int):
    """Log metrics to MLflow with checkpoint as step."""
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            log_metric(metric_name, metric_value, step=checkpoint)


def run_staged_experiment(
    experiment_name: str,
    checkpoints: List[int],
    documents_dir: Optional[str] = None,
    test_data: Optional[str] = None,
    baseline_metrics: Optional[str] = None,
    early_stopping_threshold: float = 0.95
):
    """Run a staged RAG evaluation experiment with MLflow tracking."""
    
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log experiment parameters
        log_param("checkpoints", ",".join(map(str, checkpoints)))
        log_param("early_stopping_threshold", early_stopping_threshold)
        
        # Log configuration from environment
        config_params = [
            "LLM_MODEL", "EMBEDDING_MODEL", "CHUNK_SIZE", 
            "CHUNK_OVERLAP_SIZE", "RETRIEVAL_MODE", "TOP_K"
        ]
        for param in config_params:
            if param in os.environ:
                log_param(param.lower(), os.environ[param])
        
        state_file = "checkpoint_state.json"
        
        # Run evaluation at each checkpoint
        for i, checkpoint in enumerate(checkpoints):
            print(f"\n--- Running checkpoint {checkpoint} ---")
            
            try:
                # Run checkpoint evaluation
                results = run_checkpoint_evaluation(
                    checkpoint=checkpoint,
                    resume_state=state_file if i > 0 else None,
                    documents_dir=documents_dir,
                    test_data=test_data,
                    baseline_metrics=baseline_metrics
                )
                
                # Log metrics
                if "metrics" in results:
                    log_checkpoint_metrics(results["metrics"], checkpoint)
                    
                    # Log checkpoint details
                    log_metric("documents_ingested", checkpoint, step=checkpoint)
                    
                    # Check for early stopping
                    if os.path.exists("early_stopping.json"):
                        with open("early_stopping.json", 'r') as f:
                            early_stop_data = json.load(f)
                        
                        log_param("early_stopped", True)
                        log_param("stopped_at_checkpoint", early_stop_data["stopped_at_checkpoint"])
                        log_artifact("early_stopping.json")
                        
                        print(f"Early stopping triggered at checkpoint {checkpoint}")
                        break
                
                # Log checkpoint results file
                results_file = f"checkpoint_{checkpoint}_results.json"
                if os.path.exists(results_file):
                    log_artifact(results_file)
                
            except Exception as e:
                print(f"Error at checkpoint {checkpoint}: {e}")
                log_param(f"error_checkpoint_{checkpoint}", str(e))
        
        # Log final results and state
        if os.path.exists("results.json"):
            log_artifact("results.json")
        
        if os.path.exists(state_file):
            log_artifact(state_file)
        
        # Create a summary plot if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            # Load all checkpoint results
            checkpoint_metrics = {}
            for checkpoint in checkpoints:
                results_file = f"checkpoint_{checkpoint}_results.json"
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        if "metrics" in data:
                            checkpoint_metrics[checkpoint] = data["metrics"]
            
            if checkpoint_metrics:
                # Plot accuracy over checkpoints
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                checkpoints_actual = sorted(checkpoint_metrics.keys())
                accuracies = [checkpoint_metrics[cp]["accuracy"] for cp in checkpoints_actual]
                latencies = [checkpoint_metrics[cp]["latency_ms"] for cp in checkpoints_actual]
                
                # Accuracy plot
                ax1.plot(checkpoints_actual, accuracies, 'b-o', linewidth=2, markersize=8)
                ax1.set_xlabel("Documents Ingested")
                ax1.set_ylabel("Accuracy")
                ax1.set_title("RAG Accuracy vs Document Count")
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1.1)
                
                # Add baseline if available
                if baseline_metrics and os.path.exists(baseline_metrics):
                    with open(baseline_metrics, 'r') as f:
                        baseline_data = json.load(f)
                    if "metrics" in baseline_data and "accuracy" in baseline_data["metrics"]:
                        baseline_acc = baseline_data["metrics"]["accuracy"]
                        ax1.axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline')
                        ax1.axhline(y=baseline_acc * early_stopping_threshold, 
                                   color='orange', linestyle=':', label='Early Stop Threshold')
                        ax1.legend()
                
                # Latency plot
                ax2.plot(checkpoints_actual, latencies, 'g-o', linewidth=2, markersize=8)
                ax2.set_xlabel("Documents Ingested")
                ax2.set_ylabel("Latency (ms)")
                ax2.set_title("Query Latency vs Document Count")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig("checkpoint_metrics.png", dpi=150)
                log_artifact("checkpoint_metrics.png")
                plt.close()
                
        except ImportError:
            print("Matplotlib not available, skipping plots")


def main():
    """Example usage of staged evaluation with MLflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run staged RAG evaluation with MLflow")
    parser.add_argument("--experiment-name", default="rag-staged-evaluation", 
                       help="MLflow experiment name")
    parser.add_argument("--checkpoints", default="5,20,100,300,500",
                       help="Comma-separated checkpoint sizes")
    parser.add_argument("--documents-dir", help="Directory containing documents")
    parser.add_argument("--test-data", help="JSON file with test Q&A pairs")
    parser.add_argument("--baseline-metrics", help="JSON file with baseline metrics")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.95,
                       help="Threshold for early stopping")
    
    args = parser.parse_args()
    
    # Parse checkpoints
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    
    # Run staged experiment
    run_staged_experiment(
        experiment_name=args.experiment_name,
        checkpoints=checkpoints,
        documents_dir=args.documents_dir,
        test_data=args.test_data,
        baseline_metrics=args.baseline_metrics,
        early_stopping_threshold=args.early_stopping_threshold
    )


if __name__ == "__main__":
    main()