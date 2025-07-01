"""RAGOPS Workspace for experiment execution."""

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import DockerEnv, EnvConf


class RAGOPSDockerConf(EnvConf):
    """Docker configuration for RAGOPS experiments."""
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        # Use a Python image with necessary dependencies
        self.default_image = "python:3.11-slim"
        self.default_entry = "cd /app && python evaluate_rag.py"
        
        # Mount both the experiment workspace and LightRAG code
        self.extra_volumes = {
            # LightRAG will be mounted from the host
            str(Path.home() / "ragops/github/LightRAG"): "/app/LightRAG"
        }


class RAGOPSDockerEnv(DockerEnv):
    """Docker environment for running RAGOPS experiments."""
    
    def __init__(self):
        conf = RAGOPSDockerConf()
        super().__init__(conf)
        
    def prepare(self):
        """Prepare the Docker environment."""
        # Pull the base image if needed
        client = docker.from_env()
        try:
            client.images.get(self.conf.default_image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image: {self.conf.default_image}")
            client.images.pull(self.conf.default_image)
            
        # Install required dependencies in the container
        # This could be done via a custom Dockerfile in production
        pass


class RAGOPSWorkspace(FBWorkspace):
    """Workspace for RAGOPS experiments."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def execute(self, run_env: Dict[str, str] = None) -> Any:
        """Execute the RAGOPS experiment in Docker environment.
        
        Returns:
            The result of the experiment execution
        """
        logger.info(f"Running RAGOPS experiment in {self.workspace_path}")
        
        # Prepare Docker environment
        docker_env = RAGOPSDockerEnv()
        docker_env.prepare()
        
        # Set up environment variables
        if run_env is None:
            run_env = {}
            
        # Ensure API keys are passed through
        import os
        if "OPENAI_API_KEY" in os.environ:
            run_env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        
        # Execute in Docker
        try:
            execute_log = docker_env.run(
                local_path=str(self.workspace_path),
                env=run_env,
            )
            
            # Check if results.json was created
            results_path = self.workspace_path / "results.json"
            if results_path.exists():
                import json
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    
                # Return accuracy as the main metric
                return results.get("metrics", {}).get("accuracy", 0.0)
            else:
                logger.error("results.json not found after execution")
                return None
                
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise


# Import docker here to avoid issues if docker is not installed
try:
    import docker
except ImportError:
    logger.warning("Docker Python library not installed. RAGOPSDockerEnv will not work.")
    docker = None