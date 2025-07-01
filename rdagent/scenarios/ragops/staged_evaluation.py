"""
Staged evaluation mechanism for RAGOPS scenario with early stopping.
"""

import logging
from typing import List, Dict, Any, Optional
import mlflow

logger = logging.getLogger(__name__)

class StagedEvaluator:
    """
    Evaluator that runs at specific checkpoints and supports early stopping.
    """
    
    def __init__(
        self,
        checkpoints: List[int] = None,
        early_stopping_threshold: float = 0.65,
        metric_name: str = "composite_score"
    ):
        """
        Initialize staged evaluator.
        
        Args:
            checkpoints: Document counts at which to evaluate
            early_stopping_threshold: Minimum score to continue
            metric_name: Metric to use for early stopping
        """
        self.checkpoints = checkpoints or [5, 20, 100, 300, 500]
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_name = metric_name
        self.current_checkpoint_idx = 0
        
    def should_evaluate(self, num_docs: int) -> bool:
        """
        Check if evaluation should run at current document count.
        
        Args:
            num_docs: Current number of documents processed
            
        Returns:
            True if evaluation should run
        """
        return num_docs in self.checkpoints
    
    def should_stop(self, metrics: Dict[str, float]) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            metrics: Current evaluation metrics
            
        Returns:
            True if processing should stop
        """
        score = metrics.get(self.metric_name, 0.0)
        
        if score < self.early_stopping_threshold:
            logger.warning(
                f"Early stopping triggered: {self.metric_name}={score:.3f} "
                f"< threshold={self.early_stopping_threshold}"
            )
            return True
        
        return False
    
    def log_checkpoint_metrics(
        self,
        metrics: Dict[str, float],
        num_docs: int,
        run_id: Optional[str] = None
    ):
        """
        Log metrics for a checkpoint.
        
        Args:
            metrics: Metrics to log
            num_docs: Current document count (used as step)
            run_id: Optional MLflow run ID
        """
        # Log to MLflow with step
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value, step=num_docs)
        
        # Log checkpoint info
        mlflow.log_param(f"checkpoint_{num_docs}_evaluated", True)
        
        # Check early stopping
        if self.should_stop(metrics):
            mlflow.set_tag("early_stopping", "true")
            mlflow.set_tag("early_stopping_checkpoint", num_docs)
            mlflow.set_tag("early_stopping_reason", 
                          f"{self.metric_name} < {self.early_stopping_threshold}")
        
        logger.info(f"Checkpoint {num_docs}: {metrics}")
    
    def get_next_checkpoint(self, current_docs: int) -> Optional[int]:
        """
        Get the next checkpoint after current document count.
        
        Args:
            current_docs: Current number of documents
            
        Returns:
            Next checkpoint or None if no more checkpoints
        """
        for checkpoint in self.checkpoints:
            if checkpoint > current_docs:
                return checkpoint
        return None