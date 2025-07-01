"""Example of how to integrate staged evaluation into the RD-Agent runner.

This shows the key modifications needed to runner.py to support staged evaluation.
"""

def _inject_staged_evaluation_script(self, exp: Experiment) -> None:
    """Inject the staged evaluation script into the workspace."""
    # Read the staged evaluation script template
    template_path = Path(__file__).parent / "templates" / "evaluate_rag_staged.py"
    with open(template_path, 'r') as f:
        evaluation_script = f.read()
    
    exp.experiment_workspace.inject_files(**{"evaluate_rag_staged.py": evaluation_script})


def _run_staged_evaluation(self, exp: Experiment, checkpoints: List[int] = None) -> None:
    """Run staged evaluation with checkpoints."""
    if checkpoints is None:
        checkpoints = [5, 20, 100, 300, 500]
    
    # Prepare baseline metrics if available
    baseline_file = None
    if hasattr(exp, 'baseline_metrics'):
        baseline_file = "baseline_metrics.json"
        exp.experiment_workspace.inject_files(**{
            baseline_file: json.dumps(exp.baseline_metrics, indent=2)
        })
    
    # Run evaluation at each checkpoint
    for checkpoint in checkpoints:
        cmd = [
            "python", "evaluate_rag_staged.py",
            "--checkpoint-docs", str(checkpoint),
            "--test-data", "test_data.json"
        ]
        
        if baseline_file:
            cmd.extend(["--baseline-metrics", baseline_file])
        
        if checkpoint > checkpoints[0]:
            cmd.extend(["--resume-from-state", "checkpoint_state.json"])
        
        # Execute evaluation
        result = exp.experiment_workspace.execute(cmd)
        
        # Load checkpoint results
        results_file = f"checkpoint_{checkpoint}_results.json"
        if exp.experiment_workspace.exists(results_file):
            with exp.experiment_workspace.open(results_file, 'r') as f:
                checkpoint_results = json.load(f)
            
            # Log metrics with MLflow using checkpoint as step
            if "metrics" in checkpoint_results:
                for metric_name, value in checkpoint_results["metrics"].items():
                    mlflow.log_metric(metric_name, value, step=checkpoint)
            
            # Check for early stopping
            if exp.experiment_workspace.exists("early_stopping.json"):
                with exp.experiment_workspace.open("early_stopping.json", 'r') as f:
                    early_stop_data = json.load(f)
                
                mlflow.log_param("early_stopped", True)
                mlflow.log_param("stopped_at_checkpoint", early_stop_data["stopped_at_checkpoint"])
                logger.info(f"Early stopping triggered at checkpoint {checkpoint}")
                break


def modified_generate_exp_with_mlflow(self, exp: Experiment) -> Experiment:
    """Modified version that uses staged evaluation."""
    # ... existing setup code ...
    
    # Inject staged evaluation script
    self._inject_staged_evaluation_script(exp)
    
    # Create experiment.env
    env_vars = self._create_experiment_env(exp)
    exp.experiment_workspace.inject_files(**{"experiment.env": env_vars})
    
    # Inject test data
    test_data = self._prepare_test_data(exp)
    exp.experiment_workspace.inject_files(**{"test_data.json": json.dumps(test_data, indent=2)})
    
    with mlflow.start_run():
        # Log parameters
        for key, value in exp.hypothesis.component_configs.items():
            mlflow.log_param(key, value)
        
        # Enable staged evaluation based on hypothesis
        if exp.hypothesis.get("use_staged_evaluation", False):
            checkpoints = exp.hypothesis.get("checkpoints", [5, 20, 100, 300, 500])
            mlflow.log_param("evaluation_mode", "staged")
            mlflow.log_param("checkpoints", ",".join(map(str, checkpoints)))
            
            # Run staged evaluation
            self._run_staged_evaluation(exp, checkpoints)
        else:
            # Run standard evaluation
            mlflow.log_param("evaluation_mode", "standard")
            result = exp.experiment_workspace.execute(["python", "evaluate_rag.py"])
        
        # Process final results
        if exp.experiment_workspace.exists("results.json"):
            with exp.experiment_workspace.open("results.json", 'r') as f:
                results = json.load(f)
            
            exp.result = RAGOpsExperimentResult(
                success=True,
                metrics=results.get("metrics", {}),
                errors=results.get("errors", [])
            )
            
            # Log final metrics
            for metric_name, value in exp.result.metrics.items():
                mlflow.log_metric(f"final_{metric_name}", value)
        
        # Log artifacts
        mlflow.log_artifact(exp.experiment_workspace.path / "results.json")
        
        # Log checkpoint visualization if available
        if exp.experiment_workspace.exists("checkpoint_metrics.png"):
            mlflow.log_artifact(exp.experiment_workspace.path / "checkpoint_metrics.png")
    
    return exp


# Example hypothesis that triggers staged evaluation
example_hypothesis = {
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small", 
    "chunk_size": 1200,
    "chunk_overlap": 100,
    "retrieval_mode": "hybrid",
    "top_k": 60,
    "use_staged_evaluation": True,
    "checkpoints": [10, 50, 100, 250, 500],
    "early_stopping_threshold": 0.95
}