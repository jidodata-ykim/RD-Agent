#!/usr/bin/env python3
"""Test script for MLflowRAGOPSRunner to demonstrate usage."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rdagent.core.experiment import Experiment
from rdagent.scenarios.ragops.mlflow_runner import MLflowRAGOPSRunner
from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.workspace import RAGOPSWorkspace


def test_mlflow_runner():
    """Test the MLflowRAGOPSRunner with a sample experiment."""
    
    # Enable MLflow via environment variable
    os.environ["ENABLE_MLFLOW"] = "true"
    
    # You can also set MLflow tracking URI if needed
    # os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    
    # Create scenario
    scenario = RAGOPSScenario()
    
    # Create runner
    runner = MLflowRAGOPSRunner(scenario)
    
    # Create a test experiment
    exp = Experiment()
    exp.experiment_workspace = RAGOPSWorkspace(scenario)
    
    # Create a sample .env configuration
    env_content = """# RAG Configuration
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
RETRIEVAL_MODE=hybrid
TOP_K=60
EMBEDDING_DIM=1536

# API Keys (these would be real in production)
OPENAI_API_KEY=sk-test-key
"""
    
    # Write the configuration
    env_path = exp.experiment_workspace.workspace_path / "experiment.env"
    env_path.write_text(env_content)
    
    # Run the experiment with MLflow tracking
    print("Running experiment with MLflow tracking...")
    try:
        result_exp = runner.develop(exp)
        print(f"Experiment completed. Result: {result_exp.result}")
        if hasattr(result_exp, 'metrics'):
            print(f"Metrics: {result_exp.metrics}")
    except Exception as e:
        print(f"Error running experiment: {e}")


if __name__ == "__main__":
    # Check if MLflow is available
    try:
        import mlflow
        print(f"MLflow version: {mlflow.__version__}")
    except ImportError:
        print("MLflow is not installed. Install it with: pip install mlflow")
        print("The runner will still work but without tracking.")
    
    test_mlflow_runner()