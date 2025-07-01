#!/usr/bin/env python3
"""Unit tests for MLflowRAGOPSRunner integration."""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from rdagent.core.experiment import Experiment
from rdagent.scenarios.ragops.mlflow_runner import MLflowRAGOPSRunner
from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.workspace import RAGOPSWorkspace


class TestMLflowRAGOPSRunner(unittest.TestCase):
    """Test cases for MLflowRAGOPSRunner."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_enable_mlflow = os.environ.get("ENABLE_MLFLOW")
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        if self.original_enable_mlflow is not None:
            os.environ["ENABLE_MLFLOW"] = self.original_enable_mlflow
        elif "ENABLE_MLFLOW" in os.environ:
            del os.environ["ENABLE_MLFLOW"]
    
    def test_mlflow_disabled_by_default(self):
        """Test that MLflow is disabled by default."""
        if "ENABLE_MLFLOW" in os.environ:
            del os.environ["ENABLE_MLFLOW"]
            
        scenario = RAGOPSScenario()
        runner = MLflowRAGOPSRunner(scenario)
        self.assertFalse(runner.mlflow_enabled)
    
    def test_mlflow_enabled_by_env_var(self):
        """Test that MLflow can be enabled via environment variable."""
        os.environ["ENABLE_MLFLOW"] = "true"
        
        with patch('rdagent.scenarios.ragops.mlflow_runner.MLFLOW_AVAILABLE', True):
            with patch('rdagent.scenarios.ragops.mlflow_runner.mlflow') as mock_mlflow:
                scenario = RAGOPSScenario()
                runner = MLflowRAGOPSRunner(scenario)
                self.assertTrue(runner.mlflow_enabled)
                mock_mlflow.set_experiment.assert_called_once_with("ragops/lightrag/cuad")
    
    def test_parse_env_file(self):
        """Test parsing of .env file."""
        scenario = RAGOPSScenario()
        runner = MLflowRAGOPSRunner(scenario)
        
        # Create test .env file
        env_path = Path(self.temp_dir) / "test.env"
        env_content = """
# Comment line
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=1200
ENABLE_FEATURE=true
TEMPERATURE=0.7
EMPTY_VALUE=
"""
        env_path.write_text(env_content)
        
        params = runner._parse_env_file(env_path)
        
        self.assertEqual(params["LLM_MODEL"], "gpt-4o-mini")
        self.assertEqual(params["CHUNK_SIZE"], 1200)
        self.assertEqual(params["ENABLE_FEATURE"], True)
        self.assertEqual(params["TEMPERATURE"], 0.7)
        self.assertEqual(params["EMPTY_VALUE"], "")
    
    def test_flatten_dict(self):
        """Test dictionary flattening for metrics."""
        scenario = RAGOPSScenario()
        runner = MLflowRAGOPSRunner(scenario)
        
        nested_dict = {
            "accuracy": 0.95,
            "performance": {
                "latency": 100,
                "throughput": 50
            },
            "costs": {
                "api": {
                    "openai": 0.5,
                    "azure": 0.3
                }
            }
        }
        
        flat = runner._flatten_dict(nested_dict)
        
        self.assertEqual(flat["accuracy"], 0.95)
        self.assertEqual(flat["performance.latency"], 100)
        self.assertEqual(flat["performance.throughput"], 50)
        self.assertEqual(flat["costs.api.openai"], 0.5)
        self.assertEqual(flat["costs.api.azure"], 0.3)
    
    @patch('rdagent.scenarios.ragops.mlflow_runner.MLFLOW_AVAILABLE', True)
    @patch('rdagent.scenarios.ragops.mlflow_runner.mlflow')
    def test_develop_with_mlflow_tracking(self, mock_mlflow):
        """Test develop method with MLflow tracking enabled."""
        os.environ["ENABLE_MLFLOW"] = "true"
        
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run
        
        scenario = RAGOPSScenario()
        runner = MLflowRAGOPSRunner(scenario)
        
        # Create mock experiment
        exp = MagicMock(spec=Experiment)
        exp.experiment_workspace = MagicMock(spec=RAGOPSWorkspace)
        exp.experiment_workspace.workspace_path = Path(self.temp_dir)
        
        # Create test .env file
        env_path = exp.experiment_workspace.workspace_path / "experiment.env"
        env_path.write_text("LLM_MODEL=gpt-4o-mini\nCHUNK_SIZE=1200")
        
        # Create test results.json
        results_path = exp.experiment_workspace.workspace_path / "results.json"
        results_path.write_text('{"metrics": {"accuracy": 0.95}, "configuration": {"model": "test"}}')
        
        # Mock parent develop method
        with patch.object(RAGOPSRunner, 'develop', return_value=exp):
            result = runner.develop(exp)
        
        # Verify MLflow calls
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_param.assert_any_call("LLM_MODEL", "gpt-4o-mini")
        mock_mlflow.log_param.assert_any_call("CHUNK_SIZE", 1200)
        mock_mlflow.log_metric.assert_any_call("metrics.accuracy", 0.95)
        mock_mlflow.end_run.assert_called_once()
    
    def test_develop_without_mlflow(self):
        """Test that develop works without MLflow when disabled."""
        os.environ["ENABLE_MLFLOW"] = "false"
        
        scenario = RAGOPSScenario()
        runner = MLflowRAGOPSRunner(scenario)
        
        # Create mock experiment
        exp = MagicMock(spec=Experiment)
        
        # Mock parent develop method
        with patch.object(RAGOPSRunner, 'develop', return_value=exp) as mock_develop:
            result = runner.develop(exp)
            
        # Verify parent method was called
        mock_develop.assert_called_once_with(exp)
        self.assertEqual(result, exp)


if __name__ == "__main__":
    unittest.main()