#!/usr/bin/env python3
"""
Merge sharded result files from multi-GPU inference.

Usage:
    # Merge specific files
    python merge_results.py results/qwen3.5-4b/level_1_rank*.json -o results/qwen3.5-4b/level_1.json

    # Auto-discover shards in a directory
    python merge_results.py --dir results/qwen3.5-4b/ --pattern "level_1_rank*"

    # Merge and clean up rank files
    python merge_results.py results/qwen3.5-4b/level_1_rank*.json -o results/qwen3.5-4b/level_1.json --cleanup
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded OVO-S result files")

    parser.add_argument(
        "files",
        type=str,
        nargs="*",
        help="Shard result files to merge"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory to search for shard files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_rank*.json",
        help="Glob pattern for shard files (used with --dir)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-detect from shard names)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete shard files after successful merge"
    )

    return parser.parse_args()


def merge_results(shard_paths: list) -> dict:
    """Merge multiple shard result files into one.

    Deduplicates by query_id, keeping the first occurrence.
    """
    seen_ids = set()
    merged_results = []
    metadata = {}

    for path in sorted(shard_paths):
        with open(path, "r") as f:
            data = json.load(f)

        # Preserve metadata from first shard
        if not metadata:
            metadata = {
                "model": data.get("model", "unknown"),
                "model_config": data.get("model_config", {}),
                "prompt_style": data.get("prompt_style", "default"),
            }

        for result in data.get("results", []):
            qid = result.get("query_id")
            if qid and qid not in seen_ids:
                seen_ids.add(qid)
                merged_results.append(result)

    merged = {
        **metadata,
        "merged_from": [str(p) for p in shard_paths],
        "merged_at": datetime.now().isoformat(),
        "total_queries": len(merged_results),
        "results": merged_results,
    }
    return merged


def main():
    args = parse_args()

    # Collect shard files
    shard_paths = []
    if args.files:
        shard_paths = [Path(f) for f in args.files]
    elif args.dir:
        shard_paths = sorted(Path(args.dir).glob(args.pattern))
    else:
        print("Error: Provide shard files as arguments or use --dir/--pattern")
        sys.exit(1)

    # Filter to existing files
    shard_paths = [p for p in shard_paths if p.exists()]
    if not shard_paths:
        print("Error: No shard files found")
        sys.exit(1)

    print(f"Merging {len(shard_paths)} shard files:")
    for p in shard_paths:
        print(f"  {p}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-detect: remove _rank{N} suffix
        import re
        base = str(shard_paths[0])
        output_path = Path(re.sub(r'_rank\d+', '', base))

    # Merge
    merged = merge_results(shard_paths)
    print(f"\nMerged {merged['total_queries']} unique queries")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")

    # Cleanup
    if args.cleanup:
        for p in shard_paths:
            if p != output_path:
                p.unlink()
                print(f"  Removed: {p}")
        # Clean up queue/lock files for THIS annotation only
        stem = output_path.stem
        for suffix in [".json", ".lock"]:
            qf = output_path.parent / f"._queue_{stem}{suffix}"
            if qf.exists():
                qf.unlink()
                print(f"  Removed: {qf}")


if __name__ == "__main__":
    main()
