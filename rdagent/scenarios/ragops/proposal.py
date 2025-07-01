"""RAGOPS Hypothesis Generation and Experiment Proposal."""

import json
from typing import List, Tuple

from rdagent.core.proposal import (
    Hypothesis,
    HypothesisGen,
    Hypothesis2Experiment,
    Trace,
)
from rdagent.core.experiment import Experiment, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.ragops.scenario import RAGOPSScenario


class RAGOPSHypothesis(Hypothesis):
    """Hypothesis for RAGOPS parameter optimization."""
    
    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        parameters: dict,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation,
            concise_justification, concise_knowledge
        )
        self.parameters = parameters  # Dict of parameter changes to test


class RAGOPSHypothesisGen(HypothesisGen):
    """Generate hypotheses for RAGOPS parameter optimization."""

    def __init__(self, scen: RAGOPSScenario):
        super().__init__(scen)
        self.scen = scen
        
        # Define parameter space for exploration
        self.parameter_options = {
            "LLM_MODEL": ["gpt-4o-mini", "gpt-4o", "claude-3-haiku", "claude-3-sonnet"],
            "EMBEDDING_MODEL": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            "CHUNK_SIZE": ["600", "800", "1200", "1600", "2000"],
            "CHUNK_OVERLAP_SIZE": ["50", "100", "200", "300"],
            "RETRIEVAL_MODE": ["local", "global", "hybrid", "naive"],
            "TOP_K": ["20", "40", "60", "80", "100"],
            "COSINE_THRESHOLD": ["0.1", "0.2", "0.3", "0.4"],
        }

    def gen(self, trace: Trace) -> Tuple[Hypothesis, ...]:
        """Generate hypotheses based on the trace of past experiments."""
        
        # Analyze past results to inform hypothesis generation
        analysis = self._analyze_trace(trace)
        
        # Generate hypothesis based on analysis
        system_prompt = """You are an expert in optimizing RAG (Retrieval-Augmented Generation) pipelines.
Your task is to analyze past experiment results and propose new parameter configurations that might improve performance.

Consider the following metrics when generating hypotheses:
- Accuracy: How well the RAG system answers questions correctly
- Latency: Response time (lower is better)
- Cost: Per-query cost (lower is better)
- Comprehensiveness: How complete the answers are
- Diversity: How varied the retrieved content is

Generate hypotheses that explore the parameter space intelligently, learning from past results."""

        user_prompt = f"""Based on the following analysis of past experiments:

{analysis}

Available parameters to tune:
{json.dumps(self.parameter_options, indent=2)}

Generate 3 different hypotheses for parameter configurations to test next. Each hypothesis should:
1. Target a specific performance metric (accuracy, latency, or cost)
2. Explain the reasoning based on past results
3. Suggest specific parameter changes

Return a JSON array with 3 hypothesis objects, each containing:
- "hypothesis": A clear statement of what you expect to happen
- "reason": Detailed reasoning based on past results
- "parameters": Dict of parameter changes (e.g., {{"LLM_MODEL": "claude-3-haiku", "CHUNK_SIZE": "800"}})
- "target_metric": Which metric this hypothesis aims to improve (accuracy/latency/cost)
"""

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
        
        hypotheses_data = json.loads(response)
        hypotheses = []
        
        for hyp_data in hypotheses_data[:3]:  # Limit to 3 hypotheses
            hypothesis = RAGOPSHypothesis(
                hypothesis=hyp_data["hypothesis"],
                reason=hyp_data["reason"],
                concise_reason=f"Testing {hyp_data['target_metric']} improvement",
                concise_observation=analysis[:200] + "..." if len(analysis) > 200 else analysis,
                concise_justification=f"Parameters: {hyp_data['parameters']}",
                concise_knowledge="RAG optimization through parameter tuning",
                parameters=hyp_data["parameters"],
            )
            hypotheses.append(hypothesis)
        
        return tuple(hypotheses)

    def _analyze_trace(self, trace: Trace) -> str:
        """Analyze the trace to understand past experiment results."""
        if not trace.hist:
            return """No previous experiments found. Starting with baseline configuration.
Suggested approach: Test variations of key parameters like LLM model and chunk size."""
        
        # Collect results from past experiments
        results = []
        for hypothesis, experiment in trace.hist:
            if hasattr(experiment, "result") and experiment.result:
                try:
                    # Parse result file if available
                    result_data = json.loads(experiment.result)
                    config = result_data.get("configuration", {})
                    metrics = result_data.get("metrics", {})
                    results.append({
                        "hypothesis": str(hypothesis.hypothesis if hasattr(hypothesis, "hypothesis") else hypothesis),
                        "config": config,
                        "metrics": metrics,
                    })
                except:
                    pass
        
        if not results:
            return "Previous experiments exist but no results were successfully parsed. Consider checking evaluation output format."
        
        # Analyze trends
        analysis = f"Analyzed {len(results)} past experiments:\n\n"
        
        # Find best performers for each metric
        if results:
            best_accuracy = max(results, key=lambda x: x["metrics"].get("accuracy", 0))
            best_latency = min(results, key=lambda x: x["metrics"].get("latency_ms", float('inf')))
            best_cost = min(results, key=lambda x: x["metrics"].get("cost_per_query", float('inf')))
            
            analysis += f"Best accuracy ({best_accuracy['metrics'].get('accuracy', 'N/A')}): "
            analysis += f"{best_accuracy['config'].get('LLM_MODEL', 'unknown')} with chunk_size={best_accuracy['config'].get('CHUNK_SIZE', 'unknown')}\n"
            
            analysis += f"Best latency ({best_latency['metrics'].get('latency_ms', 'N/A')}ms): "
            analysis += f"{best_latency['config'].get('LLM_MODEL', 'unknown')} with retrieval_mode={best_latency['config'].get('RETRIEVAL_MODE', 'unknown')}\n"
            
            analysis += f"Best cost (${best_cost['metrics'].get('cost_per_query', 'N/A')}): "
            analysis += f"{best_cost['config'].get('LLM_MODEL', 'unknown')}\n"
            
            # Identify patterns
            analysis += "\nObserved patterns:\n"
            chunk_sizes = [r["config"].get("CHUNK_SIZE", "1200") for r in results]
            analysis += f"- Tested chunk sizes: {sorted(set(chunk_sizes))}\n"
            
            models = [r["config"].get("LLM_MODEL", "unknown") for r in results]
            analysis += f"- Tested models: {sorted(set(models))}\n"
            
            modes = [r["config"].get("RETRIEVAL_MODE", "hybrid") for r in results]
            analysis += f"- Tested retrieval modes: {sorted(set(modes))}\n"
        
        return analysis


class RAGOPSHypothesis2Experiment(Hypothesis2Experiment[RAGOPSHypothesis, Experiment]):
    """Convert RAGOPS hypotheses into experiments."""

    def __init__(self, scen: RAGOPSScenario):
        super().__init__(scen)
        self.scen = scen

    def convert(self, hypothesis: RAGOPSHypothesis, trace: Trace) -> Experiment:
        """Convert a hypothesis into an executable experiment."""
        
        # Create a task with the parameter configuration
        task_description = json.dumps({
            "env": hypothesis.parameters,
            "hypothesis": hypothesis.hypothesis,
            "target_metric": hypothesis.concise_reason,
        })
        
        task = Task(
            description=task_description,
            task_type="ragops_config",
        )
        
        # Create experiment with the task
        exp = Experiment([task])
        
        return exp