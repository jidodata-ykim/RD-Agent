"""RAGOPS Developer that configures experiments via .env files."""

import json
from pathlib import Path
from typing import Dict, Any

from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.scenarios.ragops.scenario import RAGOPSScenario


class RAGOPSDeveloper(Developer[Experiment]):
    """Developer that writes configuration files instead of code.
    
    This developer takes a Task with parameter specifications and creates
    a .env file to configure the LightRAG pipeline for the experiment.
    """

    def __init__(self, scen: RAGOPSScenario) -> None:
        super().__init__(scen)
        # Default base configuration
        self.base_config = {
            # LLM Configuration
            "LLM_BINDING": "openai",
            "LLM_MODEL": "gpt-4o-mini",
            "LLM_BINDING_HOST": "https://api.openai.com/v1",
            "TEMPERATURE": "0",
            "MAX_TOKENS": "32768",
            "MAX_ASYNC": "4",
            
            # Embedding Configuration
            "EMBEDDING_BINDING": "openai",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "1536",
            "EMBEDDING_BINDING_HOST": "https://api.openai.com/v1",
            "EMBEDDING_BATCH_NUM": "32",
            
            # Storage Configuration
            "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
            "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
            
            # RAG Query Settings
            "RETRIEVAL_MODE": "hybrid",
            "TOP_K": "60",
            "COSINE_THRESHOLD": "0.2",
            "CHUNK_SIZE": "1200",
            "CHUNK_OVERLAP_SIZE": "100",
        }

    def develop(self, exp: Experiment) -> Experiment:
        """Generate a .env configuration file based on the experiment task.
        
        The task description should contain a JSON object with the parameters
        to modify from the base configuration.
        """
        # Parse the task description to get parameter modifications
        task_params = self._parse_task_parameters(exp.tasks[0])
        
        # Merge base config with task-specific parameters
        config = self.base_config.copy()
        config.update(task_params)
        
        # Write the .env file to the experiment workspace
        env_content = self._generate_env_content(config)
        exp.experiment_workspace.inject_files(**{"experiment.env": env_content})
        
        # Also write a config summary for reference
        config_summary = {
            "base_config": self.base_config,
            "modifications": task_params,
            "final_config": config
        }
        exp.experiment_workspace.inject_files(**{
            "config_summary.json": json.dumps(config_summary, indent=2)
        })
        
        return exp

    def _parse_task_parameters(self, task) -> Dict[str, str]:
        """Extract parameter modifications from the task description.
        
        Expected format in task.description:
        {"env": {"LLM_MODEL": "claude-3-haiku", "CHUNK_SIZE": "800"}}
        """
        try:
            # Try to parse JSON from task description
            task_data = json.loads(task.description)
            if isinstance(task_data, dict) and "env" in task_data:
                # Convert all values to strings for .env format
                return {k: str(v) for k, v in task_data["env"].items()}
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, try to extract key-value pairs
            # from natural language description
            params = {}
            desc = str(task.description).lower()
            
            # Simple heuristic extraction
            if "gpt-4o" in desc and "mini" not in desc:
                params["LLM_MODEL"] = "gpt-4o"
            elif "claude" in desc:
                if "haiku" in desc:
                    params["LLM_MODEL"] = "claude-3-haiku"
                elif "sonnet" in desc:
                    params["LLM_MODEL"] = "claude-3-sonnet"
                elif "opus" in desc:
                    params["LLM_MODEL"] = "claude-3-opus"
            
            if "chunk" in desc:
                # Try to extract chunk size
                import re
                chunk_match = re.search(r'chunk[_ ]?size[:\s]+(\d+)', desc, re.IGNORECASE)
                if chunk_match:
                    params["CHUNK_SIZE"] = chunk_match.group(1)
            
            if "local" in desc and "mode" in desc:
                params["RETRIEVAL_MODE"] = "local"
            elif "global" in desc and "mode" in desc:
                params["RETRIEVAL_MODE"] = "global"
            elif "naive" in desc:
                params["RETRIEVAL_MODE"] = "naive"
                
            if "top" in desc and "k" in desc:
                topk_match = re.search(r'top[_ ]?k[:\s]+(\d+)', desc, re.IGNORECASE)
                if topk_match:
                    params["TOP_K"] = topk_match.group(1)
                    
        return params

    def _generate_env_content(self, config: Dict[str, str]) -> str:
        """Generate the content of the .env file from the configuration dict."""
        lines = []
        
        # Group related configurations
        groups = {
            "LLM Configuration": ["LLM_", "TEMPERATURE", "MAX_TOKENS", "MAX_ASYNC"],
            "Embedding Configuration": ["EMBEDDING_"],
            "Storage Configuration": ["LIGHTRAG_"],
            "RAG Query Settings": ["RETRIEVAL_MODE", "TOP_K", "COSINE_THRESHOLD", "CHUNK_"],
        }
        
        for group_name, prefixes in groups.items():
            lines.append(f"# {group_name}")
            for key, value in sorted(config.items()):
                if any(key.startswith(prefix) for prefix in prefixes) or key in prefixes:
                    lines.append(f"{key}={value}")
            lines.append("")  # Empty line between groups
        
        # Add any remaining keys not in groups
        remaining_keys = []
        for key in config:
            if not any(
                any(key.startswith(prefix) for prefix in prefixes) or key in prefixes
                for prefixes in groups.values()
            ):
                remaining_keys.append(key)
        
        if remaining_keys:
            lines.append("# Other Configuration")
            for key in sorted(remaining_keys):
                lines.append(f"{key}={config[key]}")
            lines.append("")
        
        return "\n".join(lines).strip() + "\n"