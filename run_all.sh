#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
gpu_index="${GPU_INDEX:-0}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {crest|wcep_ctg} [run_dir]" >&2
  exit 2
fi

dataset="$1"
case "$dataset" in
  crest|wcep_ctg) ;;
  *) echo "Dataset must be crest or wcep_ctg" >&2; exit 2 ;;
esac
run_dir="${2:-$project_root/runs/${dataset}_full}"

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$python_bin" "$project_root/scripts/run_pipeline.py" \
  --config "$project_root/configs/$dataset.yaml" \
  --run-dir "$run_dir" \
  --device cuda:0
