# VITAL-Tree RAG for Longitudinal Critical-Care Reasoning

This repository provides the experiment code for evaluating VITAL-Tree RAG on a public MIMIC-IV cohort and a protected institutional ICU cohort. It includes timestamp-constrained data extraction, longitudinal summarization, retrieval and ablation baselines, LLM-based evaluation, structured machine-learning comparators, and Clinical Decision Trajectory Consistency (CDTC) analysis.

Only source code, configuration files, tests, and simulated CDTC examples are included. Patient records, human-rating workbooks, model outputs, service credentials, and protected institutional data are not distributed.

> Research use only. The software does not provide medical advice and must not be used to make or execute clinical decisions.

## Repository layout

| Path | Description |
| --- | --- |
| `experiments/public_mimic/data_extraction/` | Timestamp-constrained AKI, ARDS, and sepsis cohort extraction from MIMIC-IV |
| `experiments/public_mimic/vital_tree_rag/` | Longitudinal summary, hierarchical retrieval, and complete-framework inference |
| `experiments/public_mimic/no_tree_rag/` | Retrieval and routing ablations |
| `experiments/public_mimic/llm_only/` | LLM-only longitudinal summary and classification workflows |
| `experiments/private_icu/` | Protected-cohort preprocessing, de-identification, generation, retry, and reporting utilities |
| `prompts/locked_prompts.py` | Versioned de-identified P1-P7 prompt templates |
| `scripts/` | Baselines, LLM-as-a-judge evaluation, CDTC runs, statistical analysis, and result assembly |
| `cdtc_pipeline/` | Standalone CDTC implementation, tests, notebook, and simulated examples |

Run commands from the repository root so that default relative paths resolve consistently.

## Installation

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy the required values from `.env.example` into your shell environment. The code does not automatically load `.env` files.

PowerShell example:

```powershell
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:OPENAI_API_KEY = "your-api-key"
$env:RAG_API_BASE = "http://localhost:9380"
$env:RAG_API_KEY = "your-rag-key"
$env:RAG_DATASET_ID = "your-dataset-id"
```

All LLM endpoints are expected to implement an OpenAI-compatible chat-completions API. RAG experiments expect a RAGFlow-compatible retrieval endpoint.

## Prompt set

The de-identified prompt set is defined in `prompts/locked_prompts.py` and includes:

1. Task-independent longitudinal summary
2. Hierarchical retrieval-term extraction
3. Retrieved-evidence synthesis
4. Public final classification
5. Persistent patient-state memory update
6. Universal private-cohort clinical response
7. CDTC trajectory extraction

The file also records the prompt-set version and analysis-lock identifier. Compared configurations use matched final prompts and differ only in their prespecified context, memory, retrieval, or routing components.

## Standalone CDTC example

This workflow uses simulated data and requires neither a clinical database nor an external API.

```bash
python cdtc_pipeline/generate_simulated_data.py \
  --output_dir cdtc_pipeline/data/example_generated \
  --n_cases 60 \
  --seed 2026

python cdtc_pipeline/run_pipeline.py \
  --input_dir cdtc_pipeline/data/example_generated \
  --output_dir cdtc_pipeline/data/output \
  --figure_dir cdtc_pipeline/figures \
  --config cdtc_pipeline/config.yaml
```

The bundled example can also be analyzed directly by setting `--input_dir` to `cdtc_pipeline/data/example`.

## Public MIMIC-IV experiments

MIMIC-IV is not redistributed. Obtain credentialed access through PhysioNet and load the approved MIMIC-IV schemas into PostgreSQL before running the extraction scripts.

The public workflow is:

1. Extract task-specific positive and negative cohorts with the scripts in `experiments/public_mimic/data_extraction/`.
2. Generate timestamp-constrained longitudinal summaries.
3. Run the complete framework or the matched ablation in `experiments/public_mimic/`.
4. Run strong LLM baselines or structured machine-learning comparators from `scripts/`.
5. Analyze patient-level outputs and calibration metrics.

Example strict-context smoke test:

```powershell
python .\scripts\run_strong_baselines.py `
  --task AKI `
  --baseline full_context_llm `
  --model your-model-name `
  --api-key $env:OPENAI_API_KEY `
  --context-policy strict_no_summary `
  --workers 2 `
  --limit 10
```

Public model inputs are assembled from timestamped structured records in the MIMIC-IV ICU and Hosp modules. No free-text content from MIMIC-IV-Note is used by the supplied public extraction pipeline.

## Protected institutional cohort

Protected institutional records are not included. The private-cohort scripts expect an approved local MySQL environment configured through the `PRIVATE_TEST_DB_*` variables in `.env.example`.

Small generation smoke test:

```powershell
python .\experiments\private_icu\private_dataset_generate.py `
  --stage admission `
  --num-patients 5 `
  --workers 1 `
  --model-workers 1 `
  --output private_dataset_model_outputs.xlsx
```

Evaluation examples:

```powershell
python .\scripts\run_private_full_200_llm_judge_doctor_mimic_v4.py `
  --api-key $env:OPENAI_API_KEY `
  --workers 1 `
  --limit-patients 5

python .\scripts\run_private_full_200_cdtc.py `
  --api-key $env:OPENAI_API_KEY `
  --workers 1 `
  --limit-patients 5
```

## Validation

```bash
python -m pytest cdtc_pipeline/tests -q
python -m compileall -q cdtc_pipeline experiments prompts scripts
```

Database- and API-dependent workflows are not exercised by the unit tests. Use the available `--limit` or `--limit-patients` options for local smoke tests.

## Privacy and data governance

- Do not commit patient records, patient identifiers, absolute clinical timestamps, human-rating workbooks, or raw model outputs.
- Process protected records only inside an approved secure environment.
- MIMIC-IV remains subject to its PhysioNet data-use agreement and must not be redistributed through this repository.
- Rotate any credential that has entered Git history, even if it was later removed from the working tree.

## Reproducibility

- CDTC simulation and analysis configurations record deterministic random seeds.
- The prompt-set version and analysis lock are declared in code.
- Formal runs should record model identifiers, serving-software versions, decoding parameters, context limits, retry policy, inference dates, knowledge-base freeze date, and input-data snapshot version.
- Generated output can vary across model snapshots and serving implementations even when the source prompts are unchanged.

