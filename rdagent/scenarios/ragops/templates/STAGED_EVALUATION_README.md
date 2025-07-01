# Staged RAG Evaluation System

This directory contains templates for running staged evaluation of RAG systems with checkpoints and early stopping support.

## Files

- `evaluate_rag_staged.py` - Main staged evaluation script with checkpoint support
- `mlflow_staged_runner.py` - MLflow integration for logging checkpoint metrics
- `sample_test_data.json` - Example Q&A test dataset
- `sample_baseline_metrics.json` - Example baseline metrics for early stopping
- `sample_documents.json` - Example documents for incremental ingestion

## Features

### Staged Evaluation Script (`evaluate_rag_staged.py`)

The staged evaluation script supports:

1. **Incremental Document Ingestion**: Ingest documents in stages with checkpoints at [5, 20, 100, 300, 500] documents
2. **State Persistence**: Save and resume from checkpoint state between runs
3. **Early Stopping**: Stop evaluation early if performance meets baseline threshold
4. **Multiple Storage Backends**: Support for ChromaDB (default) and Neo4j
5. **Backward Compatibility**: Can run non-staged evaluation with `--no-staged` flag

### Command Line Arguments

```bash
python evaluate_rag_staged.py [options]

Options:
  --checkpoint-docs N          Run evaluation at specific checkpoint
  --resume-from-state FILE     Resume from saved state file
  --documents-dir DIR          Directory containing documents to ingest
  --test-data FILE             JSON file with test Q&A pairs
  --baseline-metrics FILE      JSON file with baseline metrics
  --early-stopping-threshold F Threshold for early stopping (default: 0.95)
  --checkpoints SIZES          Comma-separated checkpoint sizes
  --no-staged                  Run non-staged evaluation
```

## Usage Examples

### 1. Run Full Staged Evaluation

```bash
# Run through all default checkpoints
python evaluate_rag_staged.py \
    --documents-dir ./documents \
    --test-data sample_test_data.json \
    --baseline-metrics sample_baseline_metrics.json
```

### 2. Run Specific Checkpoint

```bash
# Evaluate at 20 documents checkpoint
python evaluate_rag_staged.py \
    --checkpoint-docs 20 \
    --documents-dir ./documents \
    --test-data sample_test_data.json
```

### 3. Resume from Previous State

```bash
# Continue from where you left off
python evaluate_rag_staged.py \
    --checkpoint-docs 100 \
    --resume-from-state checkpoint_state.json \
    --documents-dir ./documents
```

### 4. Custom Checkpoints

```bash
# Use custom checkpoint sizes
python evaluate_rag_staged.py \
    --checkpoints "10,50,100,200" \
    --documents-dir ./documents
```

### 5. MLflow Integration

```bash
# Run with MLflow tracking
python mlflow_staged_runner.py \
    --experiment-name "rag-optimization" \
    --checkpoints "5,20,100,300,500" \
    --documents-dir ./documents \
    --test-data sample_test_data.json \
    --baseline-metrics sample_baseline_metrics.json
```

## Output Files

The evaluation produces several output files:

1. **Checkpoint Results**: `checkpoint_N_results.json` for each checkpoint
2. **State File**: `checkpoint_state.json` with ingestion progress
3. **Early Stopping**: `early_stopping.json` if early stopping is triggered
4. **Final Results**: `results.json` with consolidated metrics

### Checkpoint Results Format

```json
{
  "checkpoint": 20,
  "metrics": {
    "accuracy": 0.6,
    "latency_ms": 120.5,
    "cost_per_query": 0.00075,
    "comprehensiveness": 0.48,
    "diversity": 0.7,
    "empowerment": 0.54
  },
  "configuration": {
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 1200,
    "chunk_overlap": 100,
    "retrieval_mode": "hybrid",
    "top_k": 60
  },
  "timestamp": "2024-01-15T10:30:00",
  "details": {
    "correct": 3,
    "total": 5,
    "responses": [...]
  }
}
```

## MLflow Integration

The `mlflow_staged_runner.py` script logs:

1. **Metrics per Checkpoint**: All metrics logged with checkpoint as step parameter
2. **Configuration Parameters**: LLM model, embedding model, chunk size, etc.
3. **Artifacts**: Checkpoint result files, state files, and visualization plots
4. **Early Stopping Info**: If triggered, logs stopping checkpoint and reason

### Viewing in MLflow UI

```bash
mlflow ui
```

Then navigate to http://localhost:5000 to view:
- Metrics plotted over checkpoint steps
- Parameter comparisons across runs
- Artifact downloads

## Environment Variables

The scripts read configuration from `experiment.env`:

```bash
# LLM Configuration
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# Chunking Configuration
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100

# Retrieval Configuration
RETRIEVAL_MODE=hybrid
TOP_K=60

# Storage Backend (optional)
STORAGE_BACKEND=chroma  # or neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Early Stopping Logic

Early stopping is triggered when:
```
current_accuracy / baseline_accuracy >= threshold
```

Default threshold is 0.95 (95% of baseline performance).

## Integration with RD-Agent

To use this in the RD-Agent runner:

1. Replace the evaluation script template in `runner.py`
2. Add checkpoint logging in the experiment loop
3. Use MLflow's step parameter for checkpoint metrics

Example modification to runner.py:
```python
# Log checkpoint metrics with step
for checkpoint, metrics in checkpoint_results.items():
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value, step=int(checkpoint))
```