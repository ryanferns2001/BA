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

Create in Root folder if this project a new folder named "env", in which you must create a file named **`.env`** containing:
OPENAIKEY="your api key here"


This environment variable is required for all LLM-based components.

> **Note:**  
> Other parameters which will be used for both architectures (Sample IDs, historical references etc.) are located in `main.py` and can be modified easily.

---

## Running the Code

1. Make sure you have Python ≥ 3.10 installed.
2. Install the dependencies (requirements.txt):
3. Ensure that `env/.env` exists and contains your API key.
4. Start the program with the main.py in the root folder:

## Other Information
Within the main.py in the root folder are the main calls for the architectures SimpleLLM and the Blackboard.
Both will be given the same Sample IDs (SIDs) and historical references ID (HIDs) and other parameters.
Regarding the HIDs: If an architecture tries to process a SID which is in identical HIDs, this specific hid will not be utilized for this Sample, 

Each architecture generates by default in the given export path a new subfolder with the current timestamp.
Within this timestamp folder will be subfolder created for each SID which was processed, with the results for this specific SID.
If the parameter "evaluation run" like currently is set to true when invoking an architecture, other subfolders within the timestamp folder beside the SIDS will be created,
which hold global evaluations over all SIDs, and other specialized evaluations.

For each architecture folder exist a subfolder codebase with contains all relevant code for running the specific code,
while the main runner (a single python here) for the architecture can be found in the core subfolder under codebase.

Within the main runner may be other config parameters, like for the Blackboard Architecture which Chat-GPT should be used.
