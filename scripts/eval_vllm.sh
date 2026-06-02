#!/usr/bin/env bash
# Example: run an open-source vLLM model on the full OVO-S-Bench parquet
# across multiple GPUs. Auto-merges shards when done.
set -euo pipefail

MODEL="${1:-qwen3-vl-32b}"
ANNOTATION="${2:-data/ovo_s_bench.parquet}"
GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"

python launch.py \
    --model "${MODEL}" \
    --annotation "${ANNOTATION}" \
    --gpus "${GPUS}"

python score.py --result "results/${MODEL}/$(basename "${ANNOTATION%.*}").json"
