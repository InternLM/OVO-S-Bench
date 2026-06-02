#!/usr/bin/env python3
"""Multi-GPU launcher for OVO-S-Bench evaluation.

Calculates the number of inference processes from `--gpus` and the model's
`tensor_parallel_size`, launches each with the right CUDA_VISIBLE_DEVICES /
--rank / --world-size, and merges per-rank result shards when done.

Usage:
    # 8 GPUs, qwen3-vl-32b (tp=2) → 4 processes
    python launch.py --model qwen3-vl-32b --annotation data/ovo_s_bench.parquet --gpus 8

    # Dry run (print commands only)
    python launch.py --model qwen3-vl-32b --annotation data/ovo_s_bench.parquet --gpus 8 --dry-run

    # With prompt style and sampling strategy
    python launch.py --model qwen3.5-4b --annotation data/ovo_s_bench.parquet --gpus 8 \\
        --prompt-style cot --sampling-strategy fixed --nframes 64
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.config_utils import resolve_all_models


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU launcher for OVO-S-Bench evaluation")

    parser.add_argument("--model", type=str, required=True,
                        help="Model name (from config.yaml)")
    parser.add_argument("--annotation", type=str, nargs="+", required=True,
                        help="Path(s) to annotation file(s). When multiple are "
                             "given, inference.py loads the model once and "
                             "processes each annotation sequentially with its "
                             "own output file.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--gpus", type=int, required=True,
                        help="Total number of GPUs to use")
    parser.add_argument("--procs-per-gpu", type=int, default=1,
                        help="Independent worker replicas per TP GPU group")
    parser.add_argument("--prompt-style", type=str, default=None,
                        help="Prompt template name (see prompts.py)")
    parser.add_argument("--sampling-strategy", type=str, default=None, nargs="+",
                        choices=["auto", "fps", "fixed",
                                 "single_at_query", "recent_window",
                                 "evidence_only", "log_decay"],
                        help="Frame sampling strategy. Multiple values run all "
                             "policies in the same model load.")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size override (0 = use config)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers per process (for API models)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh instead of resuming")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Filter by task subcategories (e.g. 1.1.1 1.1.2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of queries (0 = all)")
    parser.add_argument("--nframes", type=int, default=None, nargs="+",
                        help="Override nframes for fixed sampling. Multiple "
                             "values align with --sampling-strategy.")
    parser.add_argument("--tp-size", type=int, default=0,
                        help="Override tensor_parallel_size from config (0 = use config value)")
    parser.add_argument("--results-dir-suffix", type=str, default=None, nargs="+",
                        help="Appended to the model dir under results/ so "
                             "policies sharing the same nframes write to "
                             "sibling dirs. Multiple values align with "
                             "--sampling-strategy.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip automatic merge after local launch")

    return parser.parse_args()


def _load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_inference_cmd(args, rank, world_size):
    """Build the inference.py command for a given rank."""
    cmd = [
        "python", "inference.py",
        "--model", args.model,
        "--annotation", *list(args.annotation),
        "--config", args.config,
        "--rank", str(rank),
        "--world-size", str(world_size),
    ]
    if args.prompt_style:
        cmd += ["--prompt-style", args.prompt_style]
    if args.sampling_strategy:
        cmd += ["--sampling-strategy", *args.sampling_strategy]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.workers > 1:
        cmd += ["--workers", str(args.workers)]
    if args.no_resume:
        cmd += ["--no-resume"]
    if args.tasks:
        cmd += ["--tasks"] + args.tasks
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.nframes:
        cmd += ["--nframes", *map(str, args.nframes)]
    if args.tp_size:
        cmd += ["--tp-size", str(args.tp_size)]
    if args.results_dir_suffix:
        cmd += ["--results-dir-suffix", *args.results_dir_suffix]
    return cmd


def _launch_local(args, model_config, config):
    """Launch multiple processes locally with CUDA_VISIBLE_DEVICES."""
    tp_size = int(args.tp_size) if args.tp_size > 0 else model_config.get("tensor_parallel_size", 1)
    procs_per_gpu = max(1, int(args.procs_per_gpu))
    gpu_groups = args.gpus // tp_size
    num_procs = gpu_groups * procs_per_gpu
    if gpu_groups < 1:
        print(f"Error: {args.gpus} GPUs < tensor_parallel_size {tp_size}")
        sys.exit(1)

    print(
        f"Launching {num_procs} processes "
        f"(tp={tp_size}, gpus={args.gpus}, procs_per_gpu={procs_per_gpu})"
    )

    processes = []
    for rank in range(num_procs):
        gpu_group = rank // procs_per_gpu
        gpu_start = gpu_group * tp_size
        gpu_ids = ",".join(str(gpu_start + i) for i in range(tp_size))
        cmd = _build_inference_cmd(args, rank, num_procs)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        if args.dry_run:
            print(f"  [rank {rank}] CUDA_VISIBLE_DEVICES={gpu_ids} {' '.join(cmd)}")
        else:
            print(f"  [rank {rank}] CUDA_VISIBLE_DEVICES={gpu_ids}")
            proc = subprocess.Popen(cmd, env=env)
            processes.append(proc)

    if args.dry_run:
        return

    print(f"\nWaiting for {len(processes)} processes to complete...")
    exit_codes = []
    for proc in processes:
        proc.wait()
        exit_codes.append(proc.returncode)

    failed = sum(1 for c in exit_codes if c != 0)
    if failed:
        print(f"\nWarning: {failed}/{len(processes)} processes failed")

    # Auto-merge per-annotation shards (one merge per annotation file).
    if not args.no_merge and num_procs > 1:
        for annotation_path in args.annotation:
            annotation_stem = Path(annotation_path).stem
            if args.nframes:
                annotation_stem = f"{annotation_stem}_n{args.nframes[0]}"
            results_dir = Path(config["PATHS"]["results_dir"]) / (
                args.model + ((args.results_dir_suffix[0] if args.results_dir_suffix else "") or "")
            )
            shards = sorted(results_dir.glob(f"{annotation_stem}_rank*.json"))
            if shards:
                output = results_dir / f"{annotation_stem}.json"
                merge_cmd = [
                    "python", "merge_results.py",
                    *[str(s) for s in shards],
                    "-o", str(output),
                    "--cleanup",
                ]
                print(f"\nMerging {len(shards)} shards → {output}")
                subprocess.run(merge_cmd, check=True)


def main():
    args = parse_args()

    config = _load_config(args.config)
    all_models = resolve_all_models(config)

    if args.model not in all_models:
        print(f"Error: Model '{args.model}' not found in config.yaml")
        print(f"Available: {sorted(all_models.keys())}")
        sys.exit(1)

    model_config = all_models[args.model]
    tp_size = int(args.tp_size) if args.tp_size > 0 else model_config.get("tensor_parallel_size", 1)

    if args.gpus % tp_size != 0:
        print(f"Error: --gpus ({args.gpus}) must be divisible by "
              f"tensor_parallel_size ({tp_size})")
        sys.exit(1)

    _launch_local(args, model_config, config)


if __name__ == "__main__":
    main()
