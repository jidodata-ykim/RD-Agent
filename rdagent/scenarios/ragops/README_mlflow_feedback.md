# MLflow-Integrated RAGOPS Feedback

This module provides an enhanced version of the RAGOPS feedback system that integrates with MLflow to query historical experiment data.

## Features

- **MLflow Integration**: Queries MLflow for historical experiment runs instead of only using local files
- **Automatic Fallback**: Falls back to local `results.json` files if MLflow is unavailable
- **Caching**: Implements a 5-minute cache for SOTA metrics to reduce API calls
- **Graceful Error Handling**: Handles MLflow unavailability without breaking the feedback loop

## Usage

### Basic Usage

```python
from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.mlflow_feedback import MLflowRAGOPSExperiment2Feedback

# Initialize scenario
scen = RAGOPSScenario()

# Create feedback analyzer with MLflow integration
feedback_analyzer = MLflowRAGOPSExperiment2Feedback(scen)

# Generate feedback for an experiment
feedback = feedback_analyzer.generate_feedback(experiment, trace)
```

### With Custom MLflow Tracking URI

```python
# Specify a custom MLflow tracking URI
feedback_analyzer = MLflowRAGOPSExperiment2Feedback(
    scen=scen,
    mlflow_tracking_uri="http://localhost:5000"
)
```

### Cache Management

```python
# Manually invalidate the SOTA cache when needed
feedback_analyzer.invalidate_cache()

# The cache automatically expires after 5 minutes
# You can modify CACHE_TTL_SECONDS in the class to change this
```

### Retrieving All Runs

```python
# Get all runs from MLflow for analysis
all_runs = feedback_analyzer.get_all_runs(limit=100)

for run in all_runs:
    print(f"Run {run['run_id']}: Accuracy = {run['metrics'].get('accuracy', 0.0)}")
```

## Configuration

The class looks for experiments in MLflow under the name `ragops/lightrag/cuad`. This is configured in the `EXPERIMENT_NAME` class variable.

### Environment Variables

- `MLFLOW_TRACKING_URI`: Set this to specify the MLflow tracking server URI

### MLflow Run Requirements

For the integration to work properly, MLflow runs should log:

1. **Metrics**:
   - `accuracy`: Primary performance metric
   - `latency_ms`: Response time in milliseconds
   - `cost_per_query`: Estimated cost per query
   - `comprehensiveness`, `diversity`, `empowerment`: Optional quality metrics

2. **Parameters** (prefixed with `config.`):
   - Configuration parameters should be logged with a `config.` prefix
   - Example: `config.chunk_size`, `config.embedding_model`

3. **Tags**:
   - Optionally, the full configuration can be stored as a JSON string in the `configuration` tag

## Error Handling

The class handles various error scenarios:

1. **MLflow Unavailable**: Falls back to local file-based SOTA tracking
2. **Missing Experiment**: Logs a warning and falls back to local tracking
3. **Query Errors**: Catches and logs errors, continues with fallback
4. **Invalid Cache**: Automatically refreshes when cache expires

## Performance Considerations

- SOTA metrics are cached for 5 minutes to reduce MLflow API calls
- Only queries for the single best run (ordered by accuracy)
- Falls back quickly to local files if MLflow is unavailable

## Testing

Run the provided test script to verify the integration:

```bash
python test_mlflow_feedback.py
```

This will test:
- SOTA metrics retrieval
- Cache functionality
- Run retrieval
- Fallback behavior