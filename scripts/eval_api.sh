#!/usr/bin/env bash
# Example: run an API model (GPT-4o) on the full OVO-S-Bench parquet.
# Auto-resumes from results/gpt-4o/ovo_s_bench.json if present.
set -euo pipefail

MODEL="${1:-gpt-4o}"
ANNOTATION="${2:-data/ovo_s_bench.parquet}"
WORKERS="${WORKERS:-4}"

python inference.py \
    --model "${MODEL}" \
    --annotation "${ANNOTATION}" \
    --workers "${WORKERS}"

python score.py --result "results/${MODEL}/$(basename "${ANNOTATION%.*}").json"
