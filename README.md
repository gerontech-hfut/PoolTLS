# PoolTLS (Shared-Pool Timeline Summarization)

`PoolTLS` implements the complete constrained timeline generation workflow for CREST and WCEP-CTG: prepare Stage-I supervision, train a Llama QLoRA adapter, generate events from articles, cluster candidates, train the Stage-II cross-encoder, and evaluate test timelines. Both entry points, `run_all.sh` and `scripts/run_pipeline.py`, run this workflow.

## Repository Structure

```text
configs/                 Configurations for CREST and WCEP-CTG
dataset/                 The two datasets
pooltls/                 Core Python package
scripts/                 Command-line entry points for each stage
requirements.txt         Python dependencies
run_all.sh               Complete workflow for one dataset
```

Local pretrained model weights belong in `models/` or another configured local directory. Experiment outputs go to `runs/` by default. Weights, run outputs, trained checkpoints.

## Datasets and Experimental Protocol

```text
dataset/
├── crest_split/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── constraint_dict.json
└── WCEP-CTG/
    ├── train/
    ├── validation/
    └── test/
```

`pooltls.data.DatasetReader` exposes the splits as `train`, `development`, and `test`; `development` corresponds to the on-disk `validation/` split.

`WCEP-CTG/` contains 40 topics, 23,538 articles, 200 constraint-specific timelines, and 4,638 reference event–constraint entries. An event appearing under multiple constraints is counted once for each corresponding timeline. The constraints and event assignments have been reviewed and refined using each topic's original reference pool. Constraints are independent queries: an event may belong to every constraint it directly satisfies.

Stage-I supervision uses training articles and their gold timelines. The trained adapter then generates Mention records independently for each split using article text and the requested constraints. These generated mentions supply the candidate text for clustering and timeline construction. Stage-II supervision uses training candidates and training gold timelines. Every Stage-II trial trains on all positive pairs and all reliable negative pairs produced by cross-constraint screening, with class-balanced binary cross-entropy. Final metrics compare test predictions with test gold timelines.

## Environment

Use Python 3.10 and a CUDA environment that supports BF16 and bitsandbytes NF4 training. Install the pinned dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run the Complete Workflow

Run commands from the repository root. Start each new experiment in a new run directory. 

CREST, using Bash:

```bash
PYTHON_BIN=python GPU_INDEX=0 bash run_all.sh crest runs/crest_full
```

WCEP-CTG, using Bash:

```bash
PYTHON_BIN=python GPU_INDEX=0 bash run_all.sh wcep_ctg runs/wcep_ctg_full
```

`GPU_INDEX` selects the GPU exposed to the Bash launcher.

Direct Python entry point, including Windows PowerShell:

```powershell
python scripts/run_pipeline.py `
  --config configs/crest.yaml `
  --run-dir runs/crest_full `
  --device cuda:0
```

For WCEP-CTG, use `configs/wcep_ctg.yaml` and a separate run directory. 

## Stages

`scripts/run_pipeline.py` executes these stages in order:

| Stage | Work performed |
| --- | --- |
| `prepare_stage1` | Align training gold events to training articles and write full-document SFT records. |
| `train_stage1` | Train the Llama QLoRA adapter and save `models/stage1/final_adapter/`. |
| `generate_train` | Generate event mentions from training articles with that adapter. |
| `generate_development` | Generate event mentions from development articles with the same adapter. |
| `generate_test` | Generate event mentions from test articles with the same adapter. |
| `cluster_all` | Cluster same-day mentions with complete linkage, independently for all splits. |
| `prepare_stage2` | Build training positives and reliable negatives screened across all constraints. |
| `train_stage2` | Train MiniLM trials and select checkpoints and fusion settings using development metrics. |
| `select_development` | Copy the selected Stage-II configuration into `selection/selected_config.json`. |
| `score_test` | Score test candidates, fuse cross-encoder and GTE-large scores, and decode with the selected settings. |
| `build_test_timelines` | Export the decoded predictions as JSONL and in the CREST timeline layout. |
| `evaluate_test` | Write TILSE ROUGE-1/ROUGE-2 and date precision, recall, and F1 metrics. |

## Outputs

Each run retains its configuration copy, workflow manifest, and stage logs. Useful artifacts under the run directory include:

| Path | Contents |
| --- | --- |
| `stage1_data/` | Training records, article alignments, and supervision summary |
| `models/stage1/` | QLoRA checkpoints, final adapter, and training summary |
| `mentions/<split>/` | Article-generated event mentions and parsing metadata in `_meta/` |
| `candidates/<split>/` | Clustered candidates and clustering summary |
| `stage2_data/train.jsonl` | Stage-II supervised training pairs |
| `models/cross_encoder/` | Trial checkpoints and development selection results |
| `selection/selected_config.json` | Checkpoint and fusion settings used for test scoring |
| `scores/test/` | Cross-encoder, direct, and fused scores; decoded predictions |
| `timelines/test_predictions.jsonl` | Final test predictions |
| `timelines/crest/` | One-file-per-timeline export |
| `evaluation/test_metrics.json` | Final test evaluation metrics |
| `logs/` | Per-stage command output |
