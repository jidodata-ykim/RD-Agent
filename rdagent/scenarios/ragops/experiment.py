"""RAGOPS Experiment classes."""

from pathlib import Path
from typing import List

from rdagent.core.experiment import Experiment, Task
from rdagent.scenarios.ragops.workspace import RAGOPSWorkspace


class RAGOPSExperiment(Experiment):
    """Experiment class for RAGOPS scenario."""
    
    def __init__(self, tasks: List[Task], based_experiments: List[Experiment] = None):
        super().__init__(tasks, based_experiments)
        
        # Create RAGOPS-specific workspace
        workspace_path = Path(self.experiment_workspace.workspace_path)
        self.experiment_workspace = RAGOPSWorkspace(workspace_path=workspace_path)
        
        # Copy files from based experiments if any
        if based_experiments:
            for base_exp in based_experiments:
                if hasattr(base_exp, 'experiment_workspace'):
                    # Copy evaluation script and other necessary files
                    for file in ['evaluate_rag.py', 'requirements.txt']:
                        src_file = base_exp.experiment_workspace.workspace_path / file
                        if src_file.exists():
                            self.experiment_workspace.inject_file(
                                file, 
                                src_file.read_text()
                            )