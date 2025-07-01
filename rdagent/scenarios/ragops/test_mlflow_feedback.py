"""Test script for MLflow-integrated RAGOPS feedback."""

import os
import json
from pathlib import Path

from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.mlflow_feedback import MLflowRAGOPSExperiment2Feedback
from rdagent.core.proposal import Trace
from rdagent.log import rdagent_logger as logger


def test_mlflow_feedback():
    """Test the MLflow feedback integration."""
    
    # Initialize scenario
    scen = RAGOPSScenario()
    
    # Initialize MLflow feedback with optional tracking URI
    # You can set MLFLOW_TRACKING_URI environment variable or pass it here
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    feedback_analyzer = MLflowRAGOPSExperiment2Feedback(
        scen=scen,
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    
    # Create a mock trace for testing
    trace = Trace(hist=[])
    
    # Test getting SOTA metrics
    logger.info("Testing SOTA metrics retrieval...")
    sota_metrics = feedback_analyzer._get_sota_metrics(trace)
    
    print("\n=== SOTA Metrics ===")
    print(json.dumps(sota_metrics, indent=2))
    
    # Test cache functionality
    logger.info("\nTesting cache functionality...")
    
    # First call should query MLflow
    sota_1 = feedback_analyzer._get_sota_metrics(trace)
    print(f"\nFirst call source: {sota_1.get('source', 'unknown')}")
    
    # Second call should use cache
    sota_2 = feedback_analyzer._get_sota_metrics(trace)
    print(f"Second call source: {sota_2.get('source', 'unknown')}")
    
    # Invalidate cache
    feedback_analyzer.invalidate_cache()
    
    # Third call should query again
    sota_3 = feedback_analyzer._get_sota_metrics(trace)
    print(f"Third call (after invalidation) source: {sota_3.get('source', 'unknown')}")
    
    # Test getting all runs
    if feedback_analyzer.mlflow_available:
        logger.info("\nTesting retrieval of all runs...")
        all_runs = feedback_analyzer.get_all_runs(limit=10)
        print(f"\nRetrieved {len(all_runs)} runs from MLflow")
        
        if all_runs:
            print("\n=== Top 3 Runs ===")
            for i, run in enumerate(all_runs[:3]):
                print(f"\nRun {i+1}:")
                print(f"  ID: {run['run_id']}")
                print(f"  Name: {run.get('run_name', 'N/A')}")
                print(f"  Accuracy: {run['metrics'].get('accuracy', 0.0):.4f}")
                print(f"  Latency: {run['metrics'].get('latency_ms', 'N/A')} ms")
                print(f"  Cost: ${run['metrics'].get('cost_per_query', 'N/A')}")
    else:
        print("\nMLflow not available - skipping run retrieval test")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_mlflow_feedback()