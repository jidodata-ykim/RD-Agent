"""RAGOPS RD Loop configuration and entry point."""

from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.scenarios.ragops.scenario import RAGOPSScenario
from rdagent.scenarios.ragops.proposal import RAGOPSHypothesisGen, RAGOPSHypothesis2Experiment
from rdagent.scenarios.ragops.developer import RAGOPSDeveloper
from rdagent.scenarios.ragops.runner import RAGOPSRunner
from rdagent.scenarios.ragops.feedback import RAGOPSExperiment2Feedback


class RAGOPSPropSetting(BasePropSetting):
    """Configuration for RAGOPS RD Loop components."""
    
    hypothesis_gen: str = "rdagent.scenarios.ragops.proposal:RAGOPSHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.ragops.proposal:RAGOPSHypothesis2Experiment"
    coder: str = "rdagent.scenarios.ragops.developer:RAGOPSDeveloper"
    runner: str = "rdagent.scenarios.ragops.runner:RAGOPSRunner"
    summarizer: str = "rdagent.scenarios.ragops.feedback:RAGOPSExperiment2Feedback"


class RAGOPSRDLoop(RDLoop):
    """RD Loop specifically configured for RAGOPS optimization."""
    
    def __init__(self, PROP_SETTING: RAGOPSPropSetting):
        # Initialize scenario
        scen = RAGOPSScenario()
        
        # Initialize with RAGOPS-specific settings
        super().__init__(PROP_SETTING, scen)


def main():
    """Main entry point for running RAGOPS RD Loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGOPS RD Loop")
    parser.add_argument("--loop_n", type=int, default=5, help="Number of loop iterations")
    parser.add_argument("--log_dir", type=str, default="log", help="Directory for logs")
    args = parser.parse_args()
    
    # Configure and run the loop
    prop_setting = RAGOPSPropSetting()
    loop = RAGOPSRDLoop(prop_setting)
    
    # Run the specified number of iterations
    for i in range(args.loop_n):
        logger.info(f"Running RAGOPS RD Loop iteration {i+1}/{args.loop_n}")
        loop.step()
    
    logger.info("RAGOPS RD Loop completed")


if __name__ == "__main__":
    from rdagent.log import rdagent_logger as logger
    main()