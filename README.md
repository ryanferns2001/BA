# Semantic Typing – Grouping Extension (Bachelor Thesis)

This repository contains the implementation developed in the Bachelor thesis

**"Evaluation of Contextual Attribute Grouping to Support Semantic Typing in a Blackboard-Based Agentic-AI System"**

The code builds on the existing SAST blackboard-based semantic typing prototype and extends it with a two-stage Grouping Agent. The goal is to evaluate whether explicit contextual grouping of attributes improves semantic typing performance in a controlled experimental setting.

---

## Repository Structure

The repository currently contains three relevant branches:

- `main`  
  Original blackboard-based semantic typing pipeline (baseline).

- `thesis-grouping-agent`  
  Extended version including the two-stage Grouping Agent.

- `eval-candidate-generator`  
  Diagnostic setup isolating the Candidate Generator for ablation experiments.

The thesis evaluation compares the baseline (`main`) with the grouping-enhanced system (`thesis-grouping-agent`). The ablation branch is used only to analyze candidate ranking behavior independently from downstream pruning.

---

## Model Configuration

All LLM-based agents use **GPT-5** via the OpenAI API.

- No fine-tuning was performed.
- Model configuration is identical across experimental conditions.
- Differences in results are therefore attributable to architectural changes only.

---

## Dataset

The system is evaluated using the **VC-SLAM corpus** (Burgdorf et al., 2022).

The corpus provides:
- 101 urban open-data datasets
- Structured raw data (CSV / GeoJSON)
- Documentation
- Expert reference annotations
- A predefined ontology

A slightly revised version of the ontology is used. The extension only introduces minor structural refinements and does not change conceptual coverage. The same ontology is used for all experiments.

The dataset itself is not included here.

---

## Environment Setup

Create a folder named `env` in the project root.  
Inside this folder, create a file called `.env` containing:

OPENAI_API_KEY="your_api_key_here"

This environment variable is required for all LLM-based components. 

> **Note:** 
> Additional configuration parameters (e.g., Sample IDs, historical references) are defined in main.py and can be modified there if needed. 

---
## Running the Code 
1. Make sure you have Python ≥ 3.10 installed. 
2. Install the dependencies (requirements.txt): pip install -r requirements.txt
3. Ensure that env/.env exists and contains your API key. 
4. Start the program with the main.py in the root folder: python main.py

## Additional Information
The main.py file contains the main entry points for both architectures (SimpleLLM and Blackboard). Both architectures receive the same Sample IDs (SIDs), historical reference IDs (HIDs), and other configuration parameters.

If a SID appears within the selected HIDs, that specific HID is not reused for that sample to avoid leakage.

Each architecture creates a timestamped export folder in the configured output path.
Within this folder, subfolders are generated per SID containing the corresponding results.

If the parameter evaluation_run is set to True, additional subfolders are created containing aggregated evaluation results across all SIDs.

Each architecture folder contains a codebase subdirectory with the relevant implementation, while the main runner is located in the corresponding core subfolder.


