"""RAGOPS Runner that executes experiments in Docker environment."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any

from rdagent.components.runner import CachedRunner
from rdagent.core.experiment import Experiment
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.workspace import RAGOPSWorkspace


class RAGOPSRunner(CachedRunner[Experiment]):
    """Runner that executes RAGOPS experiments in isolated Docker environments."""

    def __init__(self, scen: RAGOPSScenario):
        super().__init__(scen)
        self.scen = scen

    def get_cache_key(self, exp: Experiment) -> str:
        """Generate cache key based on configuration content."""
        # Get base cache key
        base_key = super().get_cache_key(exp)
        
        # Add configuration content to cache key
        env_path = exp.experiment_workspace.workspace_path / "experiment.env"
        if env_path.exists():
            env_content = env_path.read_text()
            return md5_hash(base_key + env_content)
        
        return base_key

    def assign_cached_result(self, exp: Experiment, cached_res: Experiment) -> Experiment:
        """Assign cached results to the experiment."""
        exp = super().assign_cached_result(exp, cached_res)
        
        # Copy result files from cached experiment
        if cached_res.experiment_workspace.workspace_path.exists():
            for result_file in ["results.json", "evaluation_log.txt", "experiment.env"]:
                src_file = cached_res.experiment_workspace.workspace_path / result_file
                if src_file.exists():
                    shutil.copy(src_file, exp.experiment_workspace.workspace_path)
        
        return exp

    @cache_with_pickle(get_cache_key, assign_cached_result)
    def develop(self, exp: Experiment) -> Experiment:
        """Execute the RAGOPS experiment.
        
        This method:
        1. Ensures the evaluation script is present in the workspace
        2. Executes the experiment in a Docker container
        3. Parses and stores the results
        """
        # Ensure workspace is properly set up
        if not isinstance(exp.experiment_workspace, RAGOPSWorkspace):
            raise ValueError("RAGOPSRunner requires a RAGOPSWorkspace")
        
        # Inject the evaluation script if not already present
        if not (exp.experiment_workspace.workspace_path / "evaluate_rag.py").exists():
            self._inject_evaluation_script(exp)
        
        # Check that configuration was written by developer
        env_path = exp.experiment_workspace.workspace_path / "experiment.env"
        if not env_path.exists():
            raise FileNotFoundError(
                "experiment.env not found. RAGOPSDeveloper should have created this file."
            )
        
        # Execute the experiment
        try:
            result = exp.experiment_workspace.execute()
            exp.result = result
            
            # Parse results.json if available
            results_path = exp.experiment_workspace.workspace_path / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    exp.metrics = json.load(f)
            else:
                exp.metrics = None
                
        except Exception as e:
            # Store error in result
            exp.result = f"Experiment failed: {str(e)}"
            exp.metrics = {
                "error": str(e),
                "status": "failed"
            }
            
            # Write error to results.json for feedback analysis
            error_result = {
                "metrics": {
                    "accuracy": 0.0,
                    "latency_ms": -1,
                    "cost_per_query": -1,
                    "comprehensiveness": 0.0,
                    "diversity": 0.0,
                    "empowerment": 0.0
                },
                "configuration": {},
                "errors": [str(e)],
                "status": "failed"
            }
            results_path = exp.experiment_workspace.workspace_path / "results.json"
            with open(results_path, 'w') as f:
                json.dump(error_result, f, indent=2)
        
        return exp

    def _inject_evaluation_script(self, exp: Experiment) -> None:
        """Inject a template evaluation script into the workspace.
        
        This script will be the main entry point for running the RAG evaluation.
        """
        evaluation_script = '''#!/usr/bin/env python3
"""RAG Pipeline Evaluation Script.

This script:
1. Loads configuration from experiment.env
2. Initializes LightRAG with the specified parameters
3. Runs evaluation on the test dataset
4. Outputs metrics to results.json
"""

import json
import os
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import LightRAG and configure it
import sys
sys.path.append("/app/LightRAG")

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete, openai_embed
import dotenv

# Load environment variables from experiment.env
dotenv.load_dotenv("experiment.env")


def load_test_data() -> List[Dict[str, str]]:
    """Load the Q&A test dataset."""
    # For now, use a small hardcoded dataset
    # In production, this would load from a file
    return [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "context": "France is a country in Europe. Its capital city is Paris."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "context": "Romeo and Juliet is a famous play written by William Shakespeare."
        },
        {
            "question": "What is the speed of light?",
            "answer": "299,792,458 meters per second",
            "context": "The speed of light in vacuum is exactly 299,792,458 meters per second."
        }
    ]


def initialize_rag() -> LightRAG:
    """Initialize LightRAG with configuration from environment."""
    # Create cache directory
    cache_dir = Path("./lightrag_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Initialize LightRAG with environment parameters
    rag = LightRAG(
        working_dir=str(cache_dir),
        llm_model_func=openai_complete,
        embedding_func=openai_embed,
        chunk_token_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
        llm_model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
    )
    
    return rag


def evaluate_rag(rag: LightRAG, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluate the RAG pipeline on test data."""
    results = {
        "correct": 0,
        "total": len(test_data),
        "latencies": [],
        "responses": []
    }
    
    # First, insert contexts into RAG
    for item in test_data:
        if "context" in item:
            rag.insert(item["context"])
    
    # Query and evaluate
    query_param = QueryParam(
        mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
        top_k=int(os.getenv("TOP_K", "60")),
    )
    
    for item in test_data:
        start_time = time.time()
        
        try:
            response = rag.query(item["question"], param=query_param)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Simple accuracy check - does response contain the answer?
            is_correct = item["answer"].lower() in response.lower()
            if is_correct:
                results["correct"] += 1
            
            results["latencies"].append(latency)
            results["responses"].append({
                "question": item["question"],
                "expected": item["answer"],
                "response": response[:200],  # Truncate for storage
                "correct": is_correct
            })
        except Exception as e:
            results["responses"].append({
                "question": item["question"],
                "error": str(e)
            })
            results["latencies"].append(-1)
    
    return results


def calculate_metrics(eval_results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate final metrics from evaluation results."""
    accuracy = eval_results["correct"] / eval_results["total"] if eval_results["total"] > 0 else 0.0
    
    valid_latencies = [l for l in eval_results["latencies"] if l > 0]
    avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else -1
    
    # Estimate cost based on model and usage
    # These are rough estimates - adjust based on actual pricing
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    cost_per_1k_tokens = {
        "gpt-4o-mini": 0.00015,
        "gpt-4o": 0.005,
        "claude-3-haiku": 0.00025,
        "claude-3-sonnet": 0.003,
    }
    base_cost = cost_per_1k_tokens.get(model, 0.001)
    # Assume ~500 tokens per query on average
    cost_per_query = base_cost * 0.5
    
    return {
        "accuracy": accuracy,
        "latency_ms": avg_latency,
        "cost_per_query": cost_per_query,
        "comprehensiveness": accuracy * 0.8,  # Simplified metric
        "diversity": 0.7,  # Placeholder
        "empowerment": accuracy * 0.9  # Simplified metric
    }


def main():
    """Main evaluation function."""
    print("Starting RAG evaluation...")
    
    try:
        # Load test data
        test_data = load_test_data()
        print(f"Loaded {len(test_data)} test examples")
        
        # Initialize RAG
        print("Initializing LightRAG...")
        rag = initialize_rag()
        
        # Run evaluation
        print("Running evaluation...")
        eval_results = evaluate_rag(rag, test_data)
        
        # Calculate metrics
        metrics = calculate_metrics(eval_results)
        
        # Prepare final results
        results = {
            "metrics": metrics,
            "configuration": {
                "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                "chunk_size": int(os.getenv("CHUNK_SIZE", "1200")),
                "chunk_overlap": int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
                "retrieval_mode": os.getenv("RETRIEVAL_MODE", "hybrid"),
                "top_k": int(os.getenv("TOP_K", "60")),
            },
            "errors": [],
            "timestamp": datetime.datetime.now().isoformat(),
            "details": eval_results
        }
        
        # Write results
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation complete. Accuracy: {metrics['accuracy']:.2%}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Write error result
        error_result = {
            "metrics": {
                "accuracy": 0.0,
                "latency_ms": -1,
                "cost_per_query": -1,
                "comprehensiveness": 0.0,
                "diversity": 0.0,
                "empowerment": 0.0
            },
            "configuration": {},
            "errors": [str(e)],
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open("results.json", "w") as f:
            json.dump(error_result, f, indent=2)
        raise


if __name__ == "__main__":
    main()
'''
        
        exp.experiment_workspace.inject_files(**{"evaluate_rag.py": evaluation_script})