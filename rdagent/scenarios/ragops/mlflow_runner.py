"""MLflow-enabled RAGOPS Runner for experiment tracking."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from rdagent.core.experiment import Experiment
from rdagent.scenarios.ragops.runner import RAGOPSRunner
from rdagent.scenarios.ragops.scenario import RAGOPSScenario

logger = logging.getLogger(__name__)

# Try to import mlflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. MLflowRAGOPSRunner will fall back to RAGOPSRunner behavior.")


class MLflowRAGOPSRunner(RAGOPSRunner):
    """RAGOPS Runner with MLflow experiment tracking capabilities.
    
    This runner extends the base RAGOPSRunner to add MLflow tracking for:
    - Parameters from .env configuration
    - Metrics from results.json
    - Artifacts (config files, results, logs)
    
    The runner follows a runner-centric control pattern where the evaluation
    script remains MLflow-agnostic.
    """
    
    def __init__(self, scen: RAGOPSScenario):
        super().__init__(scen)
        self.mlflow_enabled = self._check_mlflow_enabled()
        self.experiment_name = "ragops/lightrag/cuad"
        
        # Set up MLflow if enabled
        if self.mlflow_enabled:
            try:
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"MLflow experiment set to: {self.experiment_name}")
            except Exception as e:
                logger.error(f"Failed to set MLflow experiment: {e}")
                self.mlflow_enabled = False
    
    def _check_mlflow_enabled(self) -> bool:
        """Check if MLflow is enabled and available."""
        # Check environment variable
        if os.getenv("ENABLE_MLFLOW", "false").lower() == "false":
            return False
            
        # Check if MLflow is available
        if not MLFLOW_AVAILABLE:
            return False
            
        return True
    
    def _parse_env_file(self, env_path: Path) -> Dict[str, Any]:
        """Parse .env file to extract parameters."""
        params = {}
        
        if not env_path.exists():
            return params
            
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        
                        # Try to convert to appropriate type
                        if value.lower() in ('true', 'false'):
                            params[key] = value.lower() == 'true'
                        elif value.isdigit():
                            params[key] = int(value)
                        else:
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value
        except Exception as e:
            logger.error(f"Error parsing env file: {e}")
            
        return params
    
    def _log_mlflow_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.mlflow_enabled:
            return
            
        try:
            # Log each parameter
            for key, value in params.items():
                # MLflow has a limit on parameter value length
                if isinstance(value, str) and len(value) > 250:
                    value = value[:247] + "..."
                mlflow.log_param(key, value)
                
            # Log additional metadata
            mlflow.log_param("experiment_type", "ragops")
            mlflow.log_param("runner_class", self.__class__.__name__)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"Error logging parameters to MLflow: {e}")
    
    def _log_mlflow_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to MLflow."""
        if not self.mlflow_enabled:
            return
            
        try:
            # Flatten nested metrics if necessary
            flat_metrics = self._flatten_dict(metrics, parent_key='')
            
            # Log each metric
            for key, value in flat_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, value)
                    
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _log_mlflow_artifacts(self, exp: Experiment) -> None:
        """Log artifacts to MLflow."""
        if not self.mlflow_enabled:
            return
            
        try:
            workspace_path = exp.experiment_workspace.workspace_path
            
            # Log key files as artifacts
            artifact_files = [
                "experiment.env",
                "results.json",
                "evaluation_log.txt",
                "evaluate_rag.py"
            ]
            
            for filename in artifact_files:
                file_path = workspace_path / filename
                if file_path.exists():
                    mlflow.log_artifact(str(file_path))
                    
            # Log the entire workspace as an artifact directory if it's not too large
            # (This is optional and can be commented out for large workspaces)
            workspace_size = sum(f.stat().st_size for f in workspace_path.rglob('*') if f.is_file())
            if workspace_size < 100 * 1024 * 1024:  # 100MB limit
                mlflow.log_artifacts(str(workspace_path), artifact_path="workspace")
                
        except Exception as e:
            logger.error(f"Error logging artifacts to MLflow: {e}")
    
    def develop(self, exp: Experiment) -> Experiment:
        """Execute the RAGOPS experiment with MLflow tracking.
        
        This method wraps the parent's develop method with MLflow tracking:
        1. Starts an MLflow run
        2. Logs parameters from the .env file
        3. Executes the experiment
        4. Logs metrics and artifacts
        5. Sets the run status based on success/failure
        """
        # If MLflow is not enabled, fall back to parent behavior
        if not self.mlflow_enabled:
            return super().develop(exp)
        
        # Start MLflow run
        run = None
        try:
            run = mlflow.start_run()
            logger.info(f"Started MLflow run: {run.info.run_id}")
            
            # Parse and log parameters from .env file
            env_path = exp.experiment_workspace.workspace_path / "experiment.env"
            params = self._parse_env_file(env_path)
            self._log_mlflow_params(params)
            
            # Execute the experiment
            exp = super().develop(exp)
            
            # Log metrics if available
            if hasattr(exp, 'metrics') and exp.metrics:
                self._log_mlflow_metrics(exp.metrics)
            
            # Parse and log metrics from results.json if not already in exp.metrics
            results_path = exp.experiment_workspace.workspace_path / "results.json"
            if results_path.exists():
                try:
                    with open(results_path, 'r') as f:
                        results_data = json.load(f)
                        
                    # Log metrics from results.json
                    if 'metrics' in results_data:
                        self._log_mlflow_metrics(results_data['metrics'])
                        
                    # Log any additional metadata
                    if 'configuration' in results_data:
                        for key, value in results_data['configuration'].items():
                            mlflow.log_param(f"config.{key}", value)
                            
                except Exception as e:
                    logger.error(f"Error parsing results.json: {e}")
            
            # Log artifacts
            self._log_mlflow_artifacts(exp)
            
            # Set run status
            if hasattr(exp, 'metrics') and exp.metrics and exp.metrics.get('status') == 'failed':
                mlflow.set_tag("mlflow.runStatus", "FAILED")
            else:
                mlflow.set_tag("mlflow.runStatus", "FINISHED")
                
        except Exception as e:
            logger.error(f"Error during MLflow tracking: {e}")
            if run:
                mlflow.set_tag("mlflow.runStatus", "FAILED")
                mlflow.log_param("error", str(e))
            raise
            
        finally:
            if run:
                mlflow.end_run()
                logger.info(f"Ended MLflow run: {run.info.run_id}")
        
        return exp