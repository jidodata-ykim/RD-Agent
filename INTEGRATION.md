<integration_plan>
### **High-Level Plan for Integrating RD-Agent into a LightRAG-based RAGOPS Project**

This plan outlines the steps to integrate Microsoft's RD-Agent to automate the experimentation and optimization of a RAG pipeline built with LightRAG. The primary goal is to leverage RD-Agent's research and development loop to systematically test different LightRAG configurations and identify optimal parameters for the RAGOPS evaluation pipeline.

---

#### **Step 1: Foundational Analysis and Project Setup**

**Objective:** Understand the core mechanics of both RD-Agent and the LightRAG project to establish a solid foundation for integration.

**Breakdown:**
1.  **Analyze RD-Agent:**
    *   Review the core components: `Scenario`, `Experiment`, `Hypothesis`, `Developer`, `Runner`, and `Experiment2Feedback`.
    *   Focus on the `RDLoop` class (`rdagent/components/workflow/rd_loop.py`) to understand the sequence of `propose -> code -> run -> feedback`.
    *   Study the configuration management using `pydantic-settings` and `.env` files.
    *   Examine the `DockerEnv` utility (`rdagent/utils/env.py`) as it will be crucial for running isolated experiments.
2.  **Analyze the LightRAG RAGOPS Project:**
    *   Identify the main evaluation script that runs the end-to-end RAG pipeline (e.g., `ingest -> query -> evaluate`).
    *   Pinpoint the configuration entry point. LightRAG is configured via its constructor and environment variables (`.env`, `config.ini`). These parameters (e.g., `chunk_token_size`, `llm_model`, `embedding_model`, storage types, `QueryParam` options) are the "knobs" RD-Agent will turn.
    *   Determine the format of the final evaluation output (e.g., a JSON file with metrics like accuracy, latency, and cost from Opik).

**TDD & Best Practices:**
*   **Testing:** Before any integration, create a simple, fast-running version of the RAGOPS evaluation script. This will serve as the baseline for integration tests.
*   **Documentation:** Start a `DECISIONS.md` file. The first entry should be: "Decided to use RD-Agent to manage LightRAG configurations via `.env` files rather than direct code generation, simplifying the 'Development' step to a 'Configuration' step."

---

#### **Step 2: Define the RAGOPS Scenario for RD-Agent**

**Objective:** Create a new, dedicated scenario within RD-Agent that defines the context and goals of optimizing the LightRAG pipeline.

**Breakdown:**
1.  **Create Scenario Structure:**
    *   Create a new directory: `rdagent/scenarios/ragops/`.
    *   Inside, create the necessary Python modules for the scenario's `experiment`, `proposal`, `developer`, and `runner`.
2.  **Implement `RAGOPSScenario`:**
    *   Create a `RAGOPSScenario` class inheriting from `rdagent.core.scenario.Scenario`.
    *   Define the `background` to describe the goal: "Optimize a LightRAG pipeline for a specific Q&A dataset by experimenting with different parameters."
    *   Define `get_source_data_desc` to describe the evaluation dataset.

**TDD & Best Practices:**
*   **Testing:** Write a unit test that instantiates `RAGOPSScenario` and asserts that its properties (e.g., `background`) return the expected strings.
*   **Documentation:** Update the project's main `README.md` to include "RAGOPS" as a new supported scenario.

---

#### **Step 3: Design the Experiment and Hypothesis Mechanism**

**Objective:** Adapt RD-Agent's abstract concepts of `Hypothesis` and `Experiment` to the task of RAG pipeline configuration.

**Breakdown:**
1.  **Hypothesis Generation (`RAGOPSHypothesisGen`):**
    *   The `Hypothesis` will be a proposed change to one or more LightRAG parameters. For example: "Changing the `llm_model` from `gpt-4o-mini` to `claude-3-haiku` will reduce latency while maintaining accuracy."
    *   The `HypothesisGen` will analyze past results (e.g., "the last run had high latency") to propose new parameter sets.
2.  **Task Generation (`RAGOPSHypothesis2Experiment`):**
    *   This component will translate the natural language `Hypothesis` into a concrete `Task`.
    *   The `Task`'s description will be a structured representation of the parameters to change, e.g., `{"env": {"LLM_MODEL": "claude-3-haiku"}}`.
3.  **The "Developer" as a Config-Writer:**
    *   Implement a `RAGOPSDeveloper` class. Its `develop` method will receive the `Task`.
    *   Instead of writing Python code, it will read a template `.env` file, modify it with the parameters from the `Task`, and place it inside the experiment's `Workspace`. This is the core simplification of the integration.

**TDD & Best Practices:**
*   **Testing:**
    *   Write a test for `RAGOPSDeveloper` that provides a sample `Task` and asserts that the output `.env` file in the workspace contains the correct, modified parameters.
    *   Unit test the `Hypothesis` and `Task` data structures.
*   **Code Style:** Adhere to RD-Agent's use of Pydantic models for structured data and clear separation of concerns between components.
*   **Documentation:** In `DECISIONS.md`, log the decision to use a config-writing `Developer` and explain the benefits (simplicity, safety, clear separation from the RAGOPS codebase).

---

#### **Step 4: Implement the Experiment Runner and Feedback Loop**

**Objective:** Execute the configured RAG pipeline and feed the results back into the RD-Agent loop.

**Breakdown:**
1.  **Implement `RAGOPSRunner`:**
    *   This class will use RD-Agent's `DockerEnv` to run the RAGOPS evaluation script.
    *   The `Workspace` (containing the evaluation script and the *run-specific `.env` file*) will be mounted into the Docker container.
    *   The script inside the container will read the `.env` file to initialize LightRAG with the experimental parameters.
    *   The script must write its evaluation metrics to a standardized file (e.g., `results.json`) in the workspace.
2.  **Implement `RAGOPSExperiment2Feedback`:**
    *   This component will be triggered after the `Runner` completes.
    *   It will parse `results.json` from the workspace.
    *   It will compare the new metrics (accuracy, latency, cost) with the SOTA results from the `Trace`.
    *   It will generate a `HypothesisFeedback` object, stating whether the new configuration is an improvement (`decision: True/False`) and providing a `reason`. This feedback will fuel the next iteration of the `HypothesisGen`.

**TDD & Best Practices:**
*   **Testing:**
    *   Create a mock `results.json` and write a unit test for the `Summarizer` to ensure it parses the file correctly and generates the right feedback.
    *   Write an integration test for the `Runner` that executes a "hello world" script inside the Docker container and checks its output.
*   **CI:** Add a new job to the CI pipeline (`.github/workflows/ci.yml`) that runs the integration tests for the `ragops` scenario.
*   **Documentation:** Create `docs/scenarios/ragops.md` explaining the new scenario, its purpose, and how to interpret its results.

</integration_plan>

<install_rd_agent_task>
### **Task: Create `install_rd_agent.py` Invoke Task**

**Objective:** Create a Python `invoke` task to automate the setup of the RD-Agent dependency for the RAGOPS project.

**Instructions:**
1.  **Create the Task File:**
    *   Create a new file named `install_rd_agent.py` in the `tasks/` directory of the RAGOPS project.
2.  **Implement the Invoke Task:**
    *   Use the `@task` decorator from the `invoke` library.
    *   The task should perform the following actions:
        *   Clone the official Microsoft RD-Agent repository from `https://github.com/microsoft/RD-Agent.git` into a subdirectory, e.g., `vendor/rd-agent`.
        *   Install RD-Agent's dependencies by running `pip install -r vendor/rd-agent/requirements.txt`.
        *   Add the `vendor/rd-agent` directory to the `PYTHONPATH` or handle it via project configuration to make it importable.
3.  **Create the README:**
    *   Create a `README_install_rd_agent.md` file.
    *   **Content:**
        *   **Purpose:** "This task automates the installation of the Microsoft RD-Agent framework as a project dependency."
        *   **Prerequisites:** List necessary tools like `git` and `pip`.
        *   **How to Run:** Provide the exact command: `invoke install-rd-agent`.
        *   **What it Does:** Detail the steps the script performs (cloning, installing dependencies).
        *   **Verification:** Explain how to verify a successful installation (e.g., "Check for the existence of the `vendor/rd-agent` directory.").

</install_rd_agent_task>

<run_sample_experiment_task>
### **Task: Create `run_sample_experiment_rd_agent.py` Invoke Task**

**Objective:** Create an `invoke` task to run a sample end-to-end experiment using the integrated RD-Agent to optimize the LightRAG pipeline.

**Instructions:**
1.  **Create the Task File:**
    *   Create a new file named `run_sample_experiment_rd_agent.py` in the `tasks/` directory.
2.  **Implement the Invoke Task:**
    *   Use the `@task` decorator from the `invoke` library.
    *   The task should execute the main entry point for the new `ragops` scenario. This will likely be a Python script that instantiates and runs the `RAGOPS_RDLoop`.
    *   The script should be called using `dotenv run -- python ...` to ensure the `.env` fi[Ile with LLM API keys is loaded.
    *   Example command to run: `dotenv run -- python -m rdagent.scenarios.ragops.loop --loop_n 5`
3.  **Create the README:**
    *   Create a `README_run_sample_experiment.md` file.
    *   **Content:**
        *   **Purpose:** "This task runs a sample R&D loop using RD-Agent to find optimal parameters for our LightRAG pipeline."
        *   **Prerequisites:**
            *   "Successful completion of the `invoke install-rd-agent` task."
            *   "A valid `.env` file in the project root with `OPENAI_API_KEY` (or other LLM provider keys) configured for RD-Agent's LLM calls."
        *   **How to Run:** Provide the exact command: `invoke run-sample-experiment`.
        *   **What to Expect:**
            *   Describe the expected console output (RD-Agent logs showing loops, steps, and decisions).
            *   Explain that results, logs, and experiment workspaces will be saved in the `log/` and `git_ignore_folder/` directories created by RD-Agent.
            *   Mention that the process can be monitored using the RD-Agent UI: `rdagent ui --log_dir log/`.

</run_sample_experiment_task>
<final_recommendations>
### **Final Recommendations and Guiding Philosophy**

This plan provides the "what" and "where". This final section outlines the "how" and "why" to ensure the integration is not just functional, but truly effective. The goal is not merely to connect two systems, but to build an intelligent, automated experimentation framework.

---

#### **1. The "Developer" is a "Configurator", Not a Coder**

This is the most critical principle for this integration. RD-Agent is capable of writing code, but that is **not** its primary role here. The RAGOPS project already has a well-defined pipeline using LightRAG.

*   **Core Task:** The `RAGOPSDeveloper`'s `develop` step should be implemented as a **configuration writer**. It reads a template `.env` or `config.ini` file, modifies it with the parameters defined in the `Task` (e.g., `LLM_MODEL=claude-3-haiku`, `CHUNK_SIZE=512`), and saves it to the experiment's workspace.
*   **Why this is important:**
    *   **Simplicity & Safety:** It avoids the complexity and risk of having an LLM modify the core RAG pipeline code.
    *   **Separation of Concerns:** The RAGOPS codebase remains the stable system under test. The RD-Agent's role is to manipulate the *inputs* (configurations) to that system.
    *   **Focus:** This allows the integration effort to focus on the most valuable part: the decision-making logic.

---

#### **2. The Power is in the Loop, Not the Individual Components**

The true value of this integration will come from the iterative `propose -> configure -> run -> feedback` loop. The implementer's primary focus should be on making this loop intelligent.

*   **Hypothesis Generation:** The `RAGOPSHypothesisGen` should become increasingly sophisticated. Initially, it might randomly select parameters. Over time, it should learn from the `Trace` to make informed decisions. For example:
    *   "The last run showed high latency. The next hypothesis is to switch to a smaller, faster embedding model."
    *   "Accuracy has plateaued with model changes. The next hypothesis is to experiment with smaller chunk sizes."
*   **Feedback Analysis:** The `RAGOPSExperiment2Feedback` component is the "brain" of the operation. It must translate raw metrics (latency, cost, accuracy) into a structured `HypothesisFeedback` that explains *why* an experiment succeeded or failed, and what the next logical step should be.

---

#### **3. Embrace Flexibility and Document Everything**

This is an exploratory project. The initial set of parameters chosen for experimentation may not be the most impactful. The framework must be flexible enough to adapt.

*   **Modular Configuration:** Design the `RAGOPSScenario` and associated tasks so that adding a new tunable parameter (e.g., a new `QueryParam` option in LightRAG) is straightforward and does not require rewriting the entire loop.
*   **DECISIONS.md is Non-Negotiable:** Every significant choice must be logged. Why was `top_k` chosen as the first parameter to tune? Why was the feedback summarizer designed to prioritize accuracy over cost? This log will be invaluable for understanding the agent's "thought process" and for future improvements.
*   **Iterate on the Framework Itself:** The initial integration is just the starting point. Be prepared to revisit and refactor the `Scenario`, `Developer`, and `Runner` as you discover more effective ways to structure the experiments.

---

The objective is to create a system that doesn't just run experiments, but *learns* from them. By treating the RAG pipeline as a black box and using RD-Agent as the intelligent experimenter, you will build a powerful tool for automated RAG optimization.

Let's begin.
</final_recommendations>
