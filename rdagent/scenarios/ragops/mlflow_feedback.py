"""RAGOPS Experiment Feedback with MLflow Integration."""

import json
import time
from typing import Dict, Any, Optional, List
from functools import lru_cache
from datetime import datetime, timedelta

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from rdagent.log import rdagent_logger as logger
from rdagent.core.proposal import Trace
from rdagent.core.experiment import Experiment
from rdagent.scenarios.ragops.feedback import RAGOPSExperiment2Feedback
from rdagent.scenarios.ragops.scenario import RAGOPSScenario


class MLflowRAGOPSExperiment2Feedback(RAGOPSExperiment2Feedback):
    """RAGOPS Experiment Feedback with MLflow integration for historical data."""
    
    # Cache configuration
    CACHE_TTL_SECONDS = 300  # 5 minutes cache TTL
    EXPERIMENT_NAME = "ragops/lightrag/cuad"
    
    def __init__(self, scen: RAGOPSScenario, mlflow_tracking_uri: Optional[str] = None):
        """Initialize with MLflow client.
        
        Args:
            scen: RAGOPSScenario instance
            mlflow_tracking_uri: Optional MLflow tracking URI. If None, uses default.
        """
        super().__init__(scen)
        
        # Initialize MLflow client
        try:
            self.mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
            self.mlflow_available = True
            self._experiment_id = self._get_experiment_id()
            logger.info(f"MLflow client initialized successfully. Experiment ID: {self._experiment_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow client: {e}. Falling back to local files.")
            self.mlflow_client = None
            self.mlflow_available = False
            self._experiment_id = None
        
        # Cache for SOTA metrics
        self._sota_cache = None
        self._cache_timestamp = None
    
    def _get_experiment_id(self) -> Optional[str]:
        """Get the experiment ID from MLflow."""
        if not self.mlflow_available:
            return None
            
        try:
            experiment = self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME)
            if experiment:
                return experiment.experiment_id
            else:
                logger.warning(f"Experiment '{self.EXPERIMENT_NAME}' not found in MLflow")
                return None
        except Exception as e:
            logger.error(f"Error getting experiment ID: {e}")
            return None
    
    def _is_cache_valid(self) -> bool:
        """Check if the cached SOTA metrics are still valid."""
        if self._sota_cache is None or self._cache_timestamp is None:
            return False
        
        cache_age = time.time() - self._cache_timestamp
        return cache_age < self.CACHE_TTL_SECONDS
    
    def _query_mlflow_sota(self) -> Optional[Dict[str, Any]]:
        """Query MLflow for the best performing run.
        
        Returns:
            Dictionary with SOTA metrics and configuration, or None if query fails.
        """
        if not self.mlflow_available or not self._experiment_id:
            return None
        
        try:
            # Search for successful runs, ordered by accuracy (descending)
            runs = self.mlflow_client.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string="status = 'FINISHED' and metrics.accuracy > 0",
                order_by=["metrics.accuracy DESC"],
                max_results=1
            )
            
            if not runs:
                logger.info("No successful runs found in MLflow")
                return None
            
            best_run = runs[0]
            
            # Extract metrics
            metrics = {
                "accuracy": best_run.data.metrics.get("accuracy", 0.0),
                "latency_ms": best_run.data.metrics.get("latency_ms", float('inf')),
                "cost_per_query": best_run.data.metrics.get("cost_per_query", float('inf')),
                "comprehensiveness": best_run.data.metrics.get("comprehensiveness", 0.0),
                "diversity": best_run.data.metrics.get("diversity", 0.0),
                "empowerment": best_run.data.metrics.get("empowerment", 0.0),
            }
            
            # Extract configuration from parameters or tags
            configuration = {}
            
            # Try to get configuration from parameters
            for key, value in best_run.data.params.items():
                if key.startswith("config."):
                    config_key = key.replace("config.", "")
                    try:
                        # Try to parse JSON values
                        configuration[config_key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        configuration[config_key] = value
            
            # If no configuration in parameters, try to get from tags
            if not configuration and "configuration" in best_run.data.tags:
                try:
                    configuration = json.loads(best_run.data.tags["configuration"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to parse configuration from MLflow tags")
            
            sota_data = {
                "metrics": metrics,
                "configuration": configuration,
                "run_id": best_run.info.run_id,
                "run_name": best_run.info.run_name or "Unknown",
                "timestamp": best_run.info.start_time,
                "source": "mlflow"
            }
            
            logger.info(f"Retrieved SOTA metrics from MLflow run: {best_run.info.run_id}")
            return sota_data
            
        except MlflowException as e:
            logger.error(f"MLflow error while querying SOTA: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while querying MLflow: {e}")
            return None
    
    def _get_sota_metrics(self, trace: Trace) -> Dict[str, Any]:
        """Get the best metrics from MLflow or fall back to local trace.
        
        This method:
        1. Checks if cached SOTA metrics are valid
        2. If not, queries MLflow for best run
        3. Falls back to parent implementation if MLflow unavailable
        4. Caches the result
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug("Using cached SOTA metrics")
            return self._sota_cache
        
        # Try to get from MLflow
        mlflow_sota = self._query_mlflow_sota()
        
        if mlflow_sota:
            # Update cache
            self._sota_cache = mlflow_sota
            self._cache_timestamp = time.time()
            return mlflow_sota
        
        # Fall back to parent implementation (local trace)
        logger.info("Falling back to local trace for SOTA metrics")
        local_sota = super()._get_sota_metrics(trace)
        
        # Add source indicator
        local_sota["source"] = "local"
        
        # Cache the result
        self._sota_cache = local_sota
        self._cache_timestamp = time.time()
        
        return local_sota
    
    def _extract_metrics(self, exp: Experiment) -> Dict[str, Any]:
        """Extract metrics from experiment results.
        
        Overrides parent to also log metrics to MLflow if available.
        """
        metrics = super()._extract_metrics(exp)
        
        # Optionally log to MLflow if client is available
        if self.mlflow_available and metrics["status"] == "success":
            try:
                # This is just for tracking - actual logging should be done by the experiment runner
                logger.debug(f"Experiment metrics extracted: accuracy={metrics['metrics'].get('accuracy', 0.0)}")
            except Exception as e:
                logger.warning(f"Failed to process MLflow metrics: {e}")
        
        return metrics
    
    def invalidate_cache(self):
        """Manually invalidate the SOTA cache.
        
        Useful when you know new experiments have been logged to MLflow.
        """
        self._sota_cache = None
        self._cache_timestamp = None
        logger.info("SOTA cache invalidated")
    
    def get_all_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all runs from MLflow for analysis.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            List of run dictionaries with metrics and parameters
        """
        if not self.mlflow_available or not self._experiment_id:
            logger.warning("MLflow not available, cannot retrieve runs")
            return []
        
        try:
            runs = self.mlflow_client.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["metrics.accuracy DESC"],
                max_results=limit
            )
            
            run_data = []
            for run in runs:
                run_dict = {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "start_time": run.info.start_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags)
                }
                run_data.append(run_dict)
            
            logger.info(f"Retrieved {len(run_data)} runs from MLflow")
            return run_data
            
        except Exception as e:
            logger.error(f"Error retrieving runs from MLflow: {e}")
            return []