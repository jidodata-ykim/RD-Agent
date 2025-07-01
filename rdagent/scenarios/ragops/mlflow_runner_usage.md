# MLflowRAGOPSRunner Usage Guide

## Overview

The `MLflowRAGOPSRunner` extends the base `RAGOPSRunner` to add MLflow experiment tracking capabilities. It provides automatic tracking of parameters, metrics, and artifacts while maintaining backward compatibility.

## Features

- **Automatic Parameter Tracking**: Logs all parameters from the `.env` configuration file
- **Metrics Logging**: Captures and logs metrics from `results.json`
- **Artifact Storage**: Saves configuration files, results, and logs as MLflow artifacts
- **Error Handling**: Gracefully handles MLflow connection errors
- **Backward Compatibility**: Falls back to base runner behavior when MLflow is disabled

## Installation

```bash
# Install MLflow (optional - runner works without it)
pip install mlflow
```

## Usage

### Basic Usage

```python
from rdagent.scenarios.ragops.mlflow_runner import MLflowRAGOPSRunner
from rdagent.scenarios.ragops.scenario import RAGOPSScenario

# Enable MLflow tracking
import os
os.environ["ENABLE_MLFLOW"] = "true"

# Create scenario and runner
scenario = RAGOPSScenario()
runner = MLflowRAGOPSRunner(scenario)

# Run experiment (MLflow tracking happens automatically)
experiment = runner.develop(exp)
```

### Configuration

The runner can be configured through environment variables:

- `ENABLE_MLFLOW`: Set to "true" to enable MLflow tracking (default: "false")
- `MLFLOW_TRACKING_URI`: MLflow server URL (optional)

### MLflow Experiment Structure

The runner uses the following experiment naming convention:
- Experiment name: `ragops/lightrag/cuad`

### Tracked Information

1. **Parameters** (from `.env` file):
   - `LLM_MODEL`
   - `EMBEDDING_MODEL`
   - `CHUNK_SIZE`
   - `CHUNK_OVERLAP_SIZE`
   - `RETRIEVAL_MODE`
   - `TOP_K`
   - `EMBEDDING_DIM`
   - Additional metadata (timestamp, runner class, etc.)

2. **Metrics** (from `results.json`):
   - `accuracy`
   - `latency_ms`
   - `cost_per_query`
   - `comprehensiveness`
   - `diversity`
   - `empowerment`

3. **Artifacts**:
   - `experiment.env`: Configuration file
   - `results.json`: Experiment results
   - `evaluation_log.txt`: Execution logs
   - `evaluate_rag.py`: Evaluation script
   - Full workspace directory (if < 100MB)

### Error Handling

The runner handles various error scenarios:

- **MLflow not installed**: Falls back to base runner behavior
- **MLflow server unavailable**: Logs warning and continues without tracking
- **Invalid parameters/metrics**: Skips problematic values and continues

### Example with Full Workflow

```python
import os
from rdagent.core.experiment import Experiment
from rdagent.scenarios.ragops.mlflow_runner import MLflowRAGOPSRunner
from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.workspace import RAGOPSWorkspace

# Enable MLflow
os.environ["ENABLE_MLFLOW"] = "true"

# Optional: Set MLflow tracking URI
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Create components
scenario = RAGOPSScenario()
runner = MLflowRAGOPSRunner(scenario)

# Create experiment
exp = Experiment()
exp.experiment_workspace = RAGOPSWorkspace(scenario)

# Configure experiment (normally done by developer)
config = """
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
RETRIEVAL_MODE=hybrid
TOP_K=60
"""
(exp.experiment_workspace.workspace_path / "experiment.env").write_text(config)

# Run with automatic MLflow tracking
result = runner.develop(exp)
```

## Viewing Results

After running experiments with MLflow tracking enabled:

1. **Local MLflow UI**:
   ```bash
   mlflow ui
   # Navigate to http://localhost:5000
   ```

2. **Remote MLflow Server**:
   - Set `MLFLOW_TRACKING_URI` to your server URL
   - View experiments in your MLflow dashboard

## Design Principles

1. **Runner-Centric Control**: The evaluation script (`evaluate_rag.py`) remains MLflow-agnostic
2. **Graceful Degradation**: Works without MLflow installed or enabled
3. **Comprehensive Tracking**: Captures all relevant experiment information
4. **Error Resilience**: Continues execution even if tracking fails

## Troubleshooting

- **MLflow not tracking**: Check `ENABLE_MLFLOW` environment variable
- **Connection errors**: Verify `MLFLOW_TRACKING_URI` if using remote server
- **Missing metrics**: Ensure `results.json` is properly formatted
- **Large artifacts**: Workspace > 100MB won't be logged as artifact directory