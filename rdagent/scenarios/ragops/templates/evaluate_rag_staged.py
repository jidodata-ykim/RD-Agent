#!/usr/bin/env python3
"""RAG Pipeline Staged Evaluation Script.

This script supports incremental evaluation with checkpoints for early stopping.
It can:
1. Load configuration from experiment.env
2. Initialize LightRAG with the specified parameters
3. Ingest documents incrementally with checkpoints
4. Run evaluation at each checkpoint
5. Support state persistence between stages
6. Output checkpoint-specific metrics
"""

import json
import os
import time
import datetime
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import hashlib

# Import LightRAG and configure it
import sys
sys.path.append("/app/LightRAG")

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete, openai_embed
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default checkpoint sizes
DEFAULT_CHECKPOINTS = [5, 20, 100, 300, 500]


class CheckpointState:
    """Manages checkpoint state persistence."""
    
    def __init__(self, state_file: str = "checkpoint_state.json"):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load checkpoint state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
        
        return {
            "ingested_documents": [],
            "checkpoint_results": {},
            "current_checkpoint": 0,
            "completed_checkpoints": []
        }
    
    def save_state(self):
        """Save current state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def add_ingested_document(self, doc_id: str, content: str):
        """Record an ingested document."""
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        self.state["ingested_documents"].append({
            "id": doc_id,
            "hash": doc_hash,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def get_ingested_count(self) -> int:
        """Get count of ingested documents."""
        return len(self.state["ingested_documents"])
    
    def is_checkpoint_completed(self, checkpoint: int) -> bool:
        """Check if a checkpoint has been completed."""
        return checkpoint in self.state["completed_checkpoints"]
    
    def record_checkpoint_result(self, checkpoint: int, metrics: Dict[str, float]):
        """Record results for a checkpoint."""
        self.state["checkpoint_results"][str(checkpoint)] = {
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if checkpoint not in self.state["completed_checkpoints"]:
            self.state["completed_checkpoints"].append(checkpoint)
        self.state["current_checkpoint"] = checkpoint


def load_test_data(data_file: Optional[str] = None) -> List[Dict[str, str]]:
    """Load the Q&A test dataset."""
    if data_file and os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "qa_pairs" in data:
                    return data["qa_pairs"]
        except Exception as e:
            logger.warning(f"Failed to load test data from {data_file}: {e}")
    
    # Use default test data
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


def load_documents(doc_dir: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load documents for ingestion."""
    documents = []
    
    if doc_dir and os.path.exists(doc_dir):
        doc_path = Path(doc_dir)
        # Support multiple file formats
        patterns = ["*.txt", "*.md", "*.json", "*.pdf"]
        
        for pattern in patterns:
            for file_path in doc_path.glob(pattern):
                if limit and len(documents) >= limit:
                    break
                
                try:
                    if file_path.suffix == ".json":
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                documents.extend(data[:limit - len(documents)] if limit else data)
                            elif isinstance(data, dict) and "content" in data:
                                documents.append({
                                    "id": str(file_path.stem),
                                    "content": data["content"]
                                })
                    else:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            documents.append({
                                "id": str(file_path.stem),
                                "content": content
                            })
                except Exception as e:
                    logger.warning(f"Failed to load document {file_path}: {e}")
    
    # If no documents found, use test contexts
    if not documents:
        test_data = load_test_data()
        documents = [
            {"id": f"doc_{i}", "content": item.get("context", "")}
            for i, item in enumerate(test_data)
            if "context" in item
        ]
    
    return documents


def initialize_rag(cache_dir: Optional[str] = None) -> LightRAG:
    """Initialize LightRAG with configuration from environment."""
    # Create cache directory
    if cache_dir is None:
        cache_dir = "./lightrag_cache"
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    # Get storage backend from environment
    storage_backend = os.getenv("STORAGE_BACKEND", "chroma")
    
    # Initialize LightRAG with environment parameters
    kwargs = {
        "working_dir": str(cache_path),
        "llm_model_func": openai_complete,
        "embedding_func": openai_embed,
        "chunk_token_size": int(os.getenv("CHUNK_SIZE", "1200")),
        "chunk_overlap_token_size": int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
        "llm_model_name": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "embedding_model_name": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "1536")),
    }
    
    # Add storage backend specific configuration
    if storage_backend == "neo4j":
        kwargs["graph_storage"] = "neo4j"
        kwargs["neo4j_uri"] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        kwargs["neo4j_user"] = os.getenv("NEO4J_USER", "neo4j")
        kwargs["neo4j_password"] = os.getenv("NEO4J_PASSWORD", "password")
    
    rag = LightRAG(**kwargs)
    
    return rag


def ingest_documents_incremental(
    rag: LightRAG,
    documents: List[Dict[str, str]],
    checkpoint_state: CheckpointState,
    target_count: int
) -> int:
    """Ingest documents incrementally up to target count."""
    current_count = checkpoint_state.get_ingested_count()
    
    if current_count >= target_count:
        logger.info(f"Already ingested {current_count} documents, target {target_count} reached")
        return current_count
    
    logger.info(f"Ingesting from {current_count} to {target_count} documents")
    
    for i in range(current_count, min(target_count, len(documents))):
        doc = documents[i]
        try:
            rag.insert(doc["content"])
            checkpoint_state.add_ingested_document(doc["id"], doc["content"])
            
            if (i + 1) % 10 == 0:
                logger.info(f"Ingested {i + 1} documents")
                checkpoint_state.save_state()
        except Exception as e:
            logger.error(f"Failed to ingest document {doc['id']}: {e}")
    
    final_count = checkpoint_state.get_ingested_count()
    checkpoint_state.save_state()
    
    return final_count


def evaluate_rag(rag: LightRAG, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluate the RAG pipeline on test data."""
    results = {
        "correct": 0,
        "total": len(test_data),
        "latencies": [],
        "responses": []
    }
    
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


def check_early_stopping(
    current_metrics: Dict[str, float],
    baseline_metrics: Optional[Dict[str, float]],
    threshold: float = 0.95
) -> bool:
    """Check if early stopping criteria are met."""
    if not baseline_metrics:
        return False
    
    # Check if current accuracy meets threshold of baseline
    current_acc = current_metrics.get("accuracy", 0.0)
    baseline_acc = baseline_metrics.get("accuracy", 1.0)
    
    if baseline_acc > 0 and (current_acc / baseline_acc) >= threshold:
        logger.info(f"Early stopping criteria met: {current_acc:.2%} >= {threshold * baseline_acc:.2%}")
        return True
    
    return False


def main():
    """Main evaluation function with staged execution support."""
    parser = argparse.ArgumentParser(description="Staged RAG evaluation with checkpoints")
    parser.add_argument("--checkpoint-docs", type=int, help="Run evaluation at specific checkpoint")
    parser.add_argument("--resume-from-state", type=str, help="Resume from saved state file")
    parser.add_argument("--documents-dir", type=str, help="Directory containing documents to ingest")
    parser.add_argument("--test-data", type=str, help="JSON file with test Q&A pairs")
    parser.add_argument("--baseline-metrics", type=str, help="JSON file with baseline metrics for early stopping")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.95, help="Threshold for early stopping")
    parser.add_argument("--checkpoints", type=str, help="Comma-separated checkpoint sizes")
    parser.add_argument("--no-staged", action="store_true", help="Run non-staged evaluation")
    
    args = parser.parse_args()
    
    # Load environment variables
    dotenv.load_dotenv("experiment.env")
    
    print("Starting RAG evaluation...")
    
    try:
        # Initialize checkpoint state
        state_file = args.resume_from_state or "checkpoint_state.json"
        checkpoint_state = CheckpointState(state_file)
        
        # Load documents and test data
        documents = load_documents(args.documents_dir)
        test_data = load_test_data(args.test_data)
        print(f"Loaded {len(documents)} documents and {len(test_data)} test examples")
        
        # Initialize RAG
        print("Initializing LightRAG...")
        rag = initialize_rag()
        
        # Load baseline metrics if provided
        baseline_metrics = None
        if args.baseline_metrics and os.path.exists(args.baseline_metrics):
            with open(args.baseline_metrics, 'r') as f:
                baseline_data = json.load(f)
                baseline_metrics = baseline_data.get("metrics", {})
        
        # Determine checkpoints
        if args.checkpoints:
            checkpoints = [int(x) for x in args.checkpoints.split(",")]
        else:
            checkpoints = DEFAULT_CHECKPOINTS
        
        # Filter checkpoints to those within document range
        checkpoints = [cp for cp in checkpoints if cp <= len(documents)]
        
        # Non-staged evaluation
        if args.no_staged:
            # Ingest all documents
            for i, doc in enumerate(documents):
                rag.insert(doc["content"])
                if (i + 1) % 10 == 0:
                    print(f"Ingested {i + 1} documents")
            
            # Run evaluation
            print("Running evaluation...")
            eval_results = evaluate_rag(rag, test_data)
            metrics = calculate_metrics(eval_results)
            
            # Write results
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
            
            with open("results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Evaluation complete. Accuracy: {metrics['accuracy']:.2%}")
            return
        
        # Staged evaluation
        if args.checkpoint_docs:
            # Run evaluation at specific checkpoint
            target_checkpoint = args.checkpoint_docs
            
            # Ingest documents up to checkpoint
            actual_count = ingest_documents_incremental(
                rag, documents, checkpoint_state, target_checkpoint
            )
            
            print(f"Running evaluation at checkpoint {actual_count}")
            eval_results = evaluate_rag(rag, test_data)
            metrics = calculate_metrics(eval_results)
            
            # Record checkpoint results
            checkpoint_state.record_checkpoint_result(actual_count, metrics)
            checkpoint_state.save_state()
            
            # Write checkpoint-specific results
            results = {
                "checkpoint": actual_count,
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
            
            output_file = f"checkpoint_{actual_count}_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Checkpoint {actual_count} evaluation complete. Accuracy: {metrics['accuracy']:.2%}")
            
            # Check early stopping
            if check_early_stopping(metrics, baseline_metrics, args.early_stopping_threshold):
                print("Early stopping criteria met!")
                # Write early stopping indicator
                with open("early_stopping.json", "w") as f:
                    json.dump({
                        "stopped_at_checkpoint": actual_count,
                        "metrics": metrics,
                        "baseline_metrics": baseline_metrics,
                        "threshold": args.early_stopping_threshold
                    }, f, indent=2)
        
        else:
            # Run through all checkpoints
            for checkpoint in checkpoints:
                if checkpoint_state.is_checkpoint_completed(checkpoint):
                    print(f"Checkpoint {checkpoint} already completed, skipping")
                    continue
                
                # Ingest documents up to checkpoint
                actual_count = ingest_documents_incremental(
                    rag, documents, checkpoint_state, checkpoint
                )
                
                print(f"Running evaluation at checkpoint {actual_count}")
                eval_results = evaluate_rag(rag, test_data)
                metrics = calculate_metrics(eval_results)
                
                # Record checkpoint results
                checkpoint_state.record_checkpoint_result(actual_count, metrics)
                checkpoint_state.save_state()
                
                # Write checkpoint-specific results
                results = {
                    "checkpoint": actual_count,
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
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                output_file = f"checkpoint_{actual_count}_results.json"
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                print(f"Checkpoint {actual_count} evaluation complete. Accuracy: {metrics['accuracy']:.2%}")
                
                # Check early stopping
                if check_early_stopping(metrics, baseline_metrics, args.early_stopping_threshold):
                    print("Early stopping criteria met!")
                    with open("early_stopping.json", "w") as f:
                        json.dump({
                            "stopped_at_checkpoint": actual_count,
                            "metrics": metrics,
                            "baseline_metrics": baseline_metrics,
                            "threshold": args.early_stopping_threshold
                        }, f, indent=2)
                    break
            
            # Write final consolidated results
            with open("results.json", "w") as f:
                json.dump({
                    "final_checkpoint": checkpoint_state.state["current_checkpoint"],
                    "all_checkpoints": checkpoint_state.state["checkpoint_results"],
                    "configuration": {
                        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                        "chunk_size": int(os.getenv("CHUNK_SIZE", "1200")),
                        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
                        "retrieval_mode": os.getenv("RETRIEVAL_MODE", "hybrid"),
                        "top_k": int(os.getenv("TOP_K", "60")),
                    },
                    "timestamp": datetime.datetime.now().isoformat()
                }, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
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