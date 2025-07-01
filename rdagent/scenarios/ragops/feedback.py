"""RAGOPS Experiment Feedback Analysis."""

import json
from typing import Dict, Any

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.ragops.scenario import RAGOPSScenario


class RAGOPSExperiment2Feedback(Experiment2Feedback):
    """Analyze RAGOPS experiment results and generate feedback."""
    
    def __init__(self, scen: RAGOPSScenario):
        super().__init__(scen)
        self.scen = scen

    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """Generate feedback by analyzing experiment results against SOTA.
        
        This method:
        1. Compares current experiment metrics with best previous results
        2. Analyzes whether the hypothesis was validated
        3. Suggests new directions for exploration
        """
        logger.info("Generating RAGOPS feedback...")
        
        # Extract current experiment results
        current_metrics = self._extract_metrics(exp)
        
        # Get SOTA (state-of-the-art) results from trace
        sota_metrics = self._get_sota_metrics(trace)
        
        # Generate analysis using LLM
        system_prompt = """You are an expert in optimizing RAG (Retrieval-Augmented Generation) pipelines.
Your task is to analyze experiment results and provide feedback on whether the hypothesis was validated.

Consider these metrics when evaluating results:
- Accuracy: How well the system answers questions (most important)
- Latency: Response time in milliseconds (lower is better)
- Cost: Estimated cost per query in USD (lower is better)
- Comprehensiveness, Diversity, Empowerment: Secondary quality metrics

When comparing results:
1. A result is better if it improves the primary metric (accuracy) without severely degrading others
2. If accuracy is equal, prefer lower latency
3. If both accuracy and latency are equal, prefer lower cost

Generate structured feedback that helps guide the next experiment."""

        user_prompt = f"""Current Experiment Results:
{json.dumps(current_metrics, indent=2)}

Previous Best (SOTA) Results:
{json.dumps(sota_metrics, indent=2)}

Hypothesis tested: {exp.hypothesis.hypothesis if hasattr(exp, 'hypothesis') else 'Unknown'}

Configuration changes made:
{self._get_config_changes(exp)}

Analyze these results and provide feedback. Return a JSON object with:
- "observations": What patterns or insights can be drawn from the results?
- "hypothesis_evaluation": Was the hypothesis validated? Why or why not?
- "new_hypothesis": What should be tested next based on these results?
- "reason": Detailed reasoning for your conclusions
- "decision": true if current results are better than SOTA, false otherwise
"""

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
        
        feedback_data = json.loads(response)
        
        # Create HypothesisFeedback object
        feedback = HypothesisFeedback(
            observations=feedback_data.get("observations", "No observations"),
            hypothesis_evaluation=feedback_data.get("hypothesis_evaluation", "No evaluation"),
            new_hypothesis=feedback_data.get("new_hypothesis", "No new hypothesis"),
            reason=feedback_data.get("reason", "No reason provided"),
            decision=feedback_data.get("decision", False)
        )
        
        # Update trace with new SOTA if current is better
        if feedback.decision:
            logger.info("Current experiment improved SOTA results!")
        
        return feedback

    def _extract_metrics(self, exp: Experiment) -> Dict[str, Any]:
        """Extract metrics from experiment results."""
        # Check if results.json exists in workspace
        results_path = exp.experiment_workspace.workspace_path / "results.json"
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
                return {
                    "metrics": results.get("metrics", {}),
                    "configuration": results.get("configuration", {}),
                    "errors": results.get("errors", []),
                    "status": "success" if not results.get("errors") else "failed"
                }
        
        # Fallback if no results file
        return {
            "metrics": {
                "accuracy": 0.0,
                "latency_ms": -1,
                "cost_per_query": -1,
            },
            "configuration": {},
            "errors": ["No results.json found"],
            "status": "failed"
        }

    def _get_sota_metrics(self, trace: Trace) -> Dict[str, Any]:
        """Get the best metrics from previous experiments."""
        if not trace.hist:
            # No previous experiments
            return {
                "metrics": {
                    "accuracy": 0.0,
                    "latency_ms": float('inf'),
                    "cost_per_query": float('inf'),
                },
                "configuration": {},
                "experiment_id": None
            }
        
        best_accuracy = 0.0
        best_metrics = None
        best_config = None
        
        # Find the experiment with best accuracy
        for hypothesis, experiment in trace.hist:
            metrics = self._extract_metrics(experiment)
            if metrics["status"] == "success":
                accuracy = metrics["metrics"].get("accuracy", 0.0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_metrics = metrics["metrics"]
                    best_config = metrics["configuration"]
        
        if best_metrics:
            return {
                "metrics": best_metrics,
                "configuration": best_config,
            }
        else:
            # No successful experiments yet
            return {
                "metrics": {
                    "accuracy": 0.0,
                    "latency_ms": float('inf'),
                    "cost_per_query": float('inf'),
                },
                "configuration": {},
            }

    def _get_config_changes(self, exp: Experiment) -> str:
        """Extract configuration changes from the experiment."""
        config_summary_path = exp.experiment_workspace.workspace_path / "config_summary.json"
        
        if config_summary_path.exists():
            with open(config_summary_path, 'r') as f:
                config_summary = json.load(f)
                modifications = config_summary.get("modifications", {})
                if modifications:
                    return json.dumps(modifications, indent=2)
        
        # Fallback: try to parse from task description
        if exp.tasks and exp.tasks[0].description:
            try:
                task_data = json.loads(exp.tasks[0].description)
                return json.dumps(task_data.get("env", {}), indent=2)
            except:
                pass
        
        return "No configuration changes detected"