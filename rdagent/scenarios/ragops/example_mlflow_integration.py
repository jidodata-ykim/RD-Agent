"""Example of integrating MLflow feedback into RAGOPS workflow."""

import os
from pathlib import Path

from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.mlflow_feedback import MLflowRAGOPSExperiment2Feedback
from rdagent.scenarios.ragops.feedback import RAGOPSExperiment2Feedback
from rdagent.core.proposal import Trace
from rdagent.core.experiment import Experiment
from rdagent.log import rdagent_logger as logger


def create_feedback_analyzer(use_mlflow: bool = True):
    """Create appropriate feedback analyzer based on configuration.
    
    Args:
        use_mlflow: Whether to use MLflow integration (default: True)
        
    Returns:
        Feedback analyzer instance
    """
    scen = RAGOPSScenario()
    
    if use_mlflow:
        try:
            # Try to create MLflow-integrated analyzer
            analyzer = MLflowRAGOPSExperiment2Feedback(scen)
            
            if analyzer.mlflow_available:
                logger.info("Using MLflow-integrated feedback analyzer")
                return analyzer
            else:
                logger.warning("MLflow not available, falling back to standard analyzer")
        except Exception as e:
            logger.error(f"Failed to create MLflow analyzer: {e}")
    
    # Fall back to standard analyzer
    logger.info("Using standard file-based feedback analyzer")
    return RAGOPSExperiment2Feedback(scen)


def main():
    """Example workflow showing MLflow integration."""
    
    # Check if MLflow should be used (can be configured via environment)
    use_mlflow = os.environ.get("USE_MLFLOW", "true").lower() == "true"
    
    # Create feedback analyzer
    feedback_analyzer = create_feedback_analyzer(use_mlflow=use_mlflow)
    
    # Show which analyzer is being used
    analyzer_type = type(feedback_analyzer).__name__
    logger.info(f"Feedback analyzer type: {analyzer_type}")
    
    # Example: Get current SOTA metrics
    trace = Trace(hist=[])
    sota_metrics = feedback_analyzer._get_sota_metrics(trace)
    
    print("\n=== Current SOTA Metrics ===")
    print(f"Source: {sota_metrics.get('source', 'unknown')}")
    print(f"Accuracy: {sota_metrics['metrics'].get('accuracy', 0.0):.4f}")
    print(f"Latency: {sota_metrics['metrics'].get('latency_ms', 'N/A')} ms")
    print(f"Cost: ${sota_metrics['metrics'].get('cost_per_query', 'N/A')}")
    
    # If using MLflow, show additional capabilities
    if isinstance(feedback_analyzer, MLflowRAGOPSExperiment2Feedback):
        print("\n=== MLflow Features Available ===")
        print("- Automatic historical data retrieval")
        print("- Caching of SOTA metrics")
        print("- Access to all historical runs")
        
        # Example: Get recent runs
        recent_runs = feedback_analyzer.get_all_runs(limit=5)
        if recent_runs:
            print(f"\nFound {len(recent_runs)} recent runs in MLflow")
            
            # Show accuracy trend
            print("\nAccuracy trend (most recent 5 runs):")
            for i, run in enumerate(recent_runs):
                accuracy = run['metrics'].get('accuracy', 0.0)
                timestamp = run.get('start_time', 'N/A')
                print(f"  {i+1}. {accuracy:.4f} (Run: {run['run_id'][:8]}...)")
    
    print("\n=== Integration Complete ===")


if __name__ == "__main__":
    main()