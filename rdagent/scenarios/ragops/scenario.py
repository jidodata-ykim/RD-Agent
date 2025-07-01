"""RAGOPS Scenario for optimizing LightRAG pipeline configurations."""

from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario


class RAGOPSScenario(Scenario):
    """Scenario for optimizing a LightRAG pipeline for a specific Q&A dataset."""

    def __init__(self) -> None:
        super().__init__()
        self._background = """You are optimizing a LightRAG-based RAG (Retrieval-Augmented Generation) pipeline.
The goal is to systematically experiment with different configuration parameters to find the optimal
settings that balance accuracy, latency, and cost for a specific Q&A dataset.

The LightRAG pipeline consists of:
1. Document ingestion and chunking
2. Embedding generation
3. Vector/graph storage
4. Query processing with retrieval strategies
5. LLM-based response generation

Your experiments will focus on tuning parameters like:
- LLM model selection (e.g., gpt-4o-mini, claude-3-haiku)
- Embedding model and dimensions
- Chunk size and overlap
- Retrieval strategies (local, global, hybrid)
- Top-K results and similarity thresholds
- Storage backend configurations"""
        
        self._output_format = """The experiment should output a JSON file named 'results.json' with the following structure:
{
    "metrics": {
        "accuracy": float,  # 0-1 score from evaluation
        "latency_ms": float,  # Average query latency in milliseconds
        "cost_per_query": float,  # Estimated cost per query in USD
        "comprehensiveness": float,  # 0-1 score
        "diversity": float,  # 0-1 score
        "empowerment": float  # 0-1 score
    },
    "configuration": {
        "llm_model": str,
        "embedding_model": str,
        "chunk_size": int,
        "chunk_overlap": int,
        "retrieval_mode": str,
        "top_k": int,
        # ... other parameters used
    },
    "errors": [],  # List of any errors encountered
    "timestamp": str  # ISO format timestamp
}"""

        self._interface = """Each experiment will be configured via environment variables in a .env file.
The main evaluation script reads this .env file to configure LightRAG and run the pipeline.

Key configurable parameters include:
- LLM_MODEL: The language model to use (e.g., "gpt-4o-mini", "claude-3-haiku")
- EMBEDDING_MODEL: The embedding model (e.g., "text-embedding-3-small")
- CHUNK_SIZE: Token size for document chunks (e.g., 1200)
- CHUNK_OVERLAP_SIZE: Overlap between chunks (e.g., 100)
- RETRIEVAL_MODE: Query mode (local|global|hybrid|naive)
- TOP_K: Number of results to retrieve (e.g., 60)

The evaluation script should be run as:
python evaluate_rag.py --env-file experiment.env --output results.json"""

        self._rich_style_description = "RAGOPS: LightRAG Pipeline Optimization"

    @property
    def background(self) -> str:
        return self._background

    @property
    def source_data(self) -> str:
        return """The evaluation uses a Q&A dataset with the following characteristics:
- Format: JSON lines file with question-answer pairs
- Fields: 'question' (str), 'answer' (str), 'context' (optional str)
- Size: Configurable subset for faster experimentation
- Domain: General knowledge Q&A or domain-specific based on configuration

The dataset is loaded by the evaluation script and used to:
1. Test the RAG pipeline's ability to answer questions
2. Measure retrieval quality and response accuracy
3. Calculate performance metrics"""

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        return f"""Background of the scenario:
{self.background}

Source data description:
{self.source_data}

The interface you should follow to configure experiments:
{self.interface}

The output format your evaluation script should produce:
{self.output_format}
"""