# 🧬 PhenoAgent (`pheno_agent`)

This package implements a modular, **multi-agent clinical decision support system** designed to automate clinical phenotyping and diagnostic tasks from Electronic Health Records (EHR) and structured laboratory test values. 

The framework integrates clinical expertise, structured rules (such as date-aware lab cutoffs and Marsh grading mappings), and a **corrective reflection loop** between signal extraction and validation agents.

---

## 📐 System Architecture & Data Flow

The system orchestrates a step-by-step pipeline for each patient, mapping raw records to high-confidence diagnoses.

```text
  ┌────────────────┐       ┌────────────────┐         ┌─────────────────┐
  │  EHR Markdown  │       │  TTG-IgA Labs  │         │ Clinician Rules │
  └───────┬────────┘       └───────┬────────┘         └────────┬────────┘
          │ (ehr_reader.py)        │ (lab_lookup.py)           │
          ▼                        ▼                           │
  ┌─────────────────────────────────────────┐                  │
  │          DataGatherer Agent             │                  │
  │     (Runs keyword_scanner.py first)     │                  │
  └───────────────────┬─────────────────────┘                  │
                      │                                        │
                      ▼                                        │
             { Is TTG > Cutoff? } ───(Yes)──► [ POSITIVE OVERRIDE ]
                      │                                        │
                    (No)                                       │
                      │                                        │
                      ▼                                        │
      ┌──────►┌────────────────┐                               │
      │       │SignalExtractor │ ◄─────────────────────────────┤
(Critic       └───────┬────────┘                               │
Feedback)       (Raw Signals)                                  │
      │               ▼                                        │
      └───────┌────────────────┐                               │
              │  Critic Agent  │ ◄─────────────────────────────┤
              └───────┬────────┘                               │
              (Verified Signals)                               │
                      │                                        │
                      ▼                                        │
              ┌────────────────┐                               │
              │  Adjudicator   │ ◄─────────────────────────────┘
              └───────┬────────┘
                      │
                      ▼
           [ Final Results JSON/CSV ]
```

---

## 📂 Directory Structure

```text
PhenoAgent/
├── .gitignore
├── pyproject.toml          # Project configuration & dependencies
├── README.md               # This file
└── src/
    └── pheno_agent/
        ├── __init__.py
        ├── config.py              # Central system configuration, models, & paths
        ├── llm.py                 # Robust Ollama LLM wrapper with JSON recovery & retry logic
        ├── orchestrator.py        # Central host coordinating patient diagnosis flows
        ├── pipeline.py            # CLI entry point, aggregated results, & evaluator
        │
        ├── agents/                # AI persona scripts
        │   ├── __init__.py
        │   ├── adjudicator.py     # Deterministic clinician rules mapper + reasoning writer
        │   ├── critic.py          # Fuzzy quote verifier and negation check reflection loop
        │   ├── data_gatherer.py   # Compiles notes, pre-screens keywords, selects relevant logs
        │   └── signal_extractor.py# LLM-based pathology signal extractor (Marsh/IEL/Villous)
        │
        └── tools/                 # Supporting data utilities
            ├── __init__.py
            ├── chroma_retriever.py# Vector chunk retrieval with BioClinicalBERT embeddings
            ├── ehr_reader.py      # Parser for EHR Markdown records (Notes & Labs)
            ├── keyword_scanner.py # High-speed celiac vocabulary pre-screener
            └── lab_lookup.py      # Date-aware TTG lab value phenotype evaluator
```

---

## 🤖 Agent Personas

### 1. DataGatherer
* **Type**: Deterministic Heuristics
* **Role**: Resolves patient files, looks up TTG lab histories, and scans notes using regex keywords. Focuses downstream LLM calls by extracting only notes that present celiac diagnostic signals.

### 2. SignalExtractor
* **Type**: LLM-Based (`qwen3.6:35b-a3b` by default)
* **Role**: Reads selected clinical notes in batches and extracts pathological findings into standard JSON objects mapping Marsh scores, villous architecture anomalies, IEL status, and external biopsy history.

### 3. Critic
* **Type**: Hybrid (Deterministic + LLM `qwen3.6:27b` by default)
* **Role**: Evaluates the extracted signals for correctness.
  * **Quote Verification**: Uses sliding-window fuzzy string matching to ensure that quoted diagnostic phrases actually exist in the raw EHR file.
  * **Consistency Check**: Resolves contradictions (e.g., negative sentences parsed as positive) and flags corrections back to the `SignalExtractor` for a feedback-driven re-extraction.

### 4. Adjudicator
* **Type**: Hybrid (Deterministic Rule Table + LLM `qwen3.6:27b` by default)
* **Role**: Maps verified signals to the final diagnosis (`Positive`, `Negative`, or `Indeterminate`) using the clinician's priority rules. Once selected, invokes the LLM to write a concise, citation-backed clinical rationale detailing why the patient was classified.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your system:
* **Python**: Version `3.10` or higher (verified up to `3.12`)
* **uv**: A fast, modern Python package manager. You can install it via:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Ollama**: For hosting and running local LLMs offline. Download it from [ollama.com](https://ollama.com/).

### 2. Environment Setup
Clone or navigate to the project directory, then run `uv sync` to automatically set up the virtual environment and install all package requirements:
```bash
cd PhenoAgent
uv sync
```
This command initializes a `.venv` directory and installs all dependencies specified in [pyproject.toml](file:///home/biand/Projects/PhenoAgent/pyproject.toml).

### 3. Model Provisioning (Ollama)
Start the local Ollama service:
```bash
ollama serve
```
Then pull the default models configured for the extraction and validation agents:
* **Structured Signal Extractor and Data Gatherer model** (`qwen3.6:35b-a3b`):
  ```bash
  ollama pull qwen3.6:35b-a3b
  ```
* **Reasoning, Critic, and Adjudicator model** (`qwen3.6:27b`):
  ```bash
  ollama pull qwen3.6:27b
  ```

> [!NOTE]
> If you wish to use different models, you can pull them via Ollama and override the default configurations using CLI flags (e.g. `--reasoning-model` / `--extraction-model`) or by modifying [src/pheno_agent/config.py](file:///home/biand/Projects/PhenoAgent/src/pheno_agent/config.py).

### 4. Path and Data Configuration
The agent requires local clinical datasets and clinician rule configurations to run:
1. Open [src/pheno_agent/config.py](file:///home/biand/Projects/PhenoAgent/src/pheno_agent/config.py).
2. Configure `PROJECT_ROOT` to point to the directory containing your celiac medical data:
   ```python
   PROJECT_ROOT = Path("/path/to/your/data/directory")
   ```
3. Make sure the dataset directory structure matches the expected configuration parameters in [src/pheno_agent/config.py](file:///home/biand/Projects/PhenoAgent/src/pheno_agent/config.py).
4. **Vector Retrieval (Optional)**: The system can perform semantic search over EHR records using ChromaDB (initialized via the parent/sister project's ingestion scripts). If the databases are not built, the agent will gracefully fall back to direct file reading and keyword scanning.

---

## 🛠 Command Line Interface & Usage

Use the pipeline via `uv run` in the project root.

### 1. Diagnose a Single Patient
```bash
uv run python src/pheno_agent/pipeline.py --grids R201643869 --use-local-ollama
```

### 2. Batch Diagnose Patients from a Manual Review File
```bash
uv run python src/pheno_agent/pipeline.py \
  --grids-from-file "/home/biand/Projects/Celiac_BioVU/data/Celiac Diagnosis by Manual Review.xlsx" \
  --sample 10 \
  --use-local-ollama
```

### 3. Run Pipeline for All Available Patients in the EHR Dataset
```bash
uv run python src/pheno_agent/pipeline.py --all --use-local-ollama
```

### 4. Evaluate Pipeline Against Ground Truth
```bash
uv run python src/pheno_agent/pipeline.py \
  --grids-from-file "/home/biand/Projects/Celiac_BioVU/data/Celiac Diagnosis by Manual Review.xlsx" \
  --evaluate \
  --use-local-ollama
```
*Outputs classification reports and saves a confusion matrix visualization to `results/figures/agent_confusion_matrix.png`.*

### 5. Override Default Models
```bash
uv run python src/pheno_agent/pipeline.py \
  --grids R201643869 \
  --reasoning-model qwen3.5:122b \
  --extraction-model gemma4:31b-it-q4_K_M \
  --use-local-ollama
```

---

## 🧬 Domain-Specific Heuristics
* **Date-Aware TTG Cutoffs**: The system evaluates labs by date to align with change points in laboratory assays:
  * *Pre-June 12, 2023*: Case Cutoff = **100**, Positive Cutoff = **20**
  * *June 12, 2023 – March 1, 2024*: Case Cutoff = **40**, Positive Cutoff = **4**
  * *Post-March 1, 2024*: Case Cutoff = **100**, Positive Cutoff = **15**
* **Strict Negation Check**: Strips out negated keywords before executing signal pre-screening (e.g., "no increased intraepithelial lymphocytes").
* **ChromaDB BioClinicalBERT Integration**: Leverages mean-pooled embeddings from `Bio_ClinicalBERT` across multi-query search strategies to retrieve highly relevant context segments.
