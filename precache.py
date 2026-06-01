#!/usr/bin/env python3
"""
Pre-cache video frames for OVO-S evaluation.

CPU-only — does not load any model. Uses ProcessPoolExecutor to extract
frames in parallel and write them to .frame_cache/.

Usage:
    # Use model config to determine sampling parameters
    python precache.py --annotation data/ovo_s_bench_l1_l4.parquet --model qwen3-vl-32b --workers 16

    # Manual parameters
    python precache.py --annotation data/ovo_s_bench_l1_l4.parquet --strategy fps --max-frames 128 --fps 2 --workers 16

    # Fixed-count strategy
    python precache.py --annotation data/ovo_s_bench_l1_l4.parquet --strategy fixed --nframes 128 --workers 8
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from inference import load_annotations, expand_annotations_to_queries, filter_queries, get_video_path, load_config
from utils.frame_utils import (
    extract_frames_for_query,
    extract_frames_fixed_count,
    extract_single_frame_at_query,
    extract_recent_window,
    extract_evidence_only,
    extract_log_decay,
    _compute_target_frames,
    _get_video_fps,
    _cache_key,
    _get_cache_dir,
)


def _is_cached(video_path: str, target_frames: list, frame_size: int) -> bool:
    """Check if frames are already cached."""
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return False
    fpath = cache_dir / _cache_key(video_path, target_frames, frame_size)
    return fpath.exists()


def _extract_one(args_tuple):
    """Worker function for a single query. Must be top-level for pickling.

    args_tuple shape:
        (video_path, query_time, max_frames, fps, frame_size, strategy, nframes,
         evidence_times, window_fps)
    """
    (video_path, query_time, max_frames, fps, frame_size, strategy, nframes,
     evidence_times, window_fps) = args_tuple
    try:
        if strategy == "fixed":
            extract_frames_fixed_count(video_path, query_time,
                                       nframes=nframes, frame_size=frame_size)
        elif strategy == "single_at_query":
            extract_single_frame_at_query(video_path, query_time,
                                          frame_size=frame_size)
        elif strategy == "recent_window":
            extract_recent_window(video_path, query_time, nframes=nframes,
                                  window_fps=window_fps, frame_size=frame_size)
        elif strategy == "evidence_only":
            extract_evidence_only(video_path, query_time,
                                  evidence_times=evidence_times,
                                  nframes=nframes, frame_size=frame_size)
        elif strategy == "log_decay":
            extract_log_decay(video_path, query_time, nframes=nframes,
                              frame_size=frame_size)
        else:  # fps
            extract_frames_for_query(video_path, query_time,
                                     max_frames=max_frames, fps=fps,
                                     frame_size=frame_size)
        return True, None
    except Exception as e:
        return False, str(e)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-cache frames for OVO-S evaluation")

    parser.add_argument("--annotation", type=str, required=True,
                        help="Path to annotation JSON file")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (reads sampling params from config)")
    parser.add_argument("--strategy", type=str, default="fps",
                        choices=["fps", "fixed",
                                 "single_at_query", "recent_window",
                                 "evidence_only", "log_decay"],
                        help="Sampling strategy (default: fps). New strategies "
                             "for §4.3.2 frame-sampling sensitivity: "
                             "single_at_query, recent_window, evidence_only, "
                             "log_decay.")
    parser.add_argument("--max-frames", type=int, default=128,
                        help="Max frames (for fps strategy)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Sampling fps (for fps strategy)")
    parser.add_argument("--nframes", type=int, default=128,
                        help="Fixed frame count (for fixed / recent_window / "
                             "evidence_only / log_decay strategies)")
    parser.add_argument("--window-fps", type=float, default=4.0,
                        help="Sampling fps inside the recent_window strategy "
                             "(default 4 fps → 16-frame default ≈ 4 s window).")
    parser.add_argument("--frame-size", type=int, default=512,
                        help="Max frame dimension")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Filter by task subcategories")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of queries (0 = all)")

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    # Resolve model config if specified
    if args.model:
        from utils.config_utils import resolve_all_models
        all_models = resolve_all_models(config)
        if args.model not in all_models:
            print(f"Error: Model '{args.model}' not found in config.yaml")
            sys.exit(1)
        mc = all_models[args.model]
        max_frames = mc.get("max_frames", args.max_frames)
        fps = mc.get("fps", args.fps)
        frame_size = mc.get("frame_size", args.frame_size)
        nframes = mc.get("nframes", max_frames)
        strategy = args.strategy
    else:
        max_frames = args.max_frames
        fps = args.fps
        frame_size = args.frame_size
        nframes = args.nframes
        strategy = args.strategy

    # Load and expand annotations
    annotations = load_annotations(args.annotation)
    queries = expand_annotations_to_queries(annotations)
    queries = filter_queries(queries, args.tasks, args.limit)
    print(f"Total queries: {len(queries)}")

    # Build work items and check cache
    work_items = []
    skipped = 0
    for q in queries:
        video_path = str(get_video_path(q, config))
        if not Path(video_path).exists():
            continue
        evidence_times = q.get("evidence_times") if strategy == "evidence_only" else None
        work_items.append((
            video_path, q["query_time"], max_frames, fps,
            frame_size, strategy, nframes,
            evidence_times, args.window_fps,
        ))

    # Deduplicate by (video_path, query_time, strategy + relevant params).
    # For evidence_only the evidence interval is part of the cache key.
    seen = set()
    unique_items = []
    for item in work_items:
        (vp, qt, mf, fp_, fs, strat, nf, ev_t, win_fps) = item
        if strat == "fps":
            key = (vp, qt, strat, mf, fp_, fs)
        elif strat == "recent_window":
            key = (vp, qt, strat, nf, win_fps, fs)
        elif strat == "single_at_query":
            key = (vp, qt, strat, fs)
        elif strat == "evidence_only":
            ev_key = tuple(tuple(s) for s in (ev_t or []))
            key = (vp, qt, strat, nf, ev_key, fs)
        else:  # fixed or log_decay
            key = (vp, qt, strat, nf, fs)
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    print(f"Unique extractions: {len(unique_items)} (deduplicated from {len(work_items)})")

    # Check which are already cached. For each strategy compute the same set
    # of target frame indices that the corresponding sampler would extract.
    import numpy as np

    def _targets_for(item):
        (vp, qt, mf, fp_, fs, strat, nf, ev_t, win_fps) = item
        video_fps = _get_video_fps(vp)
        if video_fps <= 0:
            return None
        end_frame = max(1, int(qt * video_fps))
        if strat == "fixed":
            return sorted(set(np.linspace(0, end_frame, nf, dtype=int).tolist()))
        if strat == "single_at_query":
            return [max(0, int(qt * video_fps))]
        if strat == "recent_window":
            step = max(1, int(round(video_fps / float(win_fps))))
            start = max(0, end_frame - (nf - 1) * step)
            targets = list(range(start, end_frame + 1, step))
            return targets[-nf:] if len(targets) > nf else targets
        if strat == "log_decay":
            near_s = max(0, end_frame - int(30 * video_fps))
            mid_s = max(0, end_frame - int(300 * video_fps))
            bands = [(near_s, end_frame, 0.60),
                     (mid_s, near_s, 0.30),
                     (0, mid_s, 0.10)]
            targets, leftover = [], nf
            for i, (s_f, e_f, frac) in enumerate(bands):
                if e_f <= s_f:
                    continue
                n_band = nf - len(targets) if i == len(bands) - 1 else int(round(nf * frac))
                n_band = min(n_band, leftover)
                if n_band <= 0:
                    continue
                targets += np.linspace(s_f, e_f - 1, n_band, dtype=int).tolist()
                leftover -= n_band
                if leftover <= 0:
                    break
            return sorted(set(targets)) or [0]
        if strat == "evidence_only":
            spans = []
            for sp in (ev_t or []):
                try:
                    s_t, e_t = float(sp[0]), float(sp[1])
                except Exception:
                    continue
                s_f = max(0, int(s_t * video_fps))
                e_f = max(s_f, min(end_frame, int(e_t * video_fps)))
                if e_f > s_f:
                    spans.append((s_f, e_f))
            if not spans:
                return sorted(set(np.linspace(0, end_frame, nf, dtype=int).tolist()))
            lengths = [e - s for s, e in spans]
            total = sum(lengths) or 1
            raw_alloc = [max(1, round(nf * L / total)) for L in lengths]
            diff = sum(raw_alloc) - nf
            while diff > 0:
                i = max(range(len(raw_alloc)), key=lambda k: raw_alloc[k])
                if raw_alloc[i] > 1:
                    raw_alloc[i] -= 1
                    diff -= 1
                else:
                    break
            targets = []
            for (s_f, e_f), n in zip(spans, raw_alloc):
                n = max(1, n)
                targets += np.linspace(s_f, e_f, n, dtype=int).tolist()
            return sorted(set(targets))
        # fps fallback
        return _compute_target_frames(qt, video_fps, mf, fp_)

    to_extract = []
    for item in unique_items:
        targets = _targets_for(item)
        if targets is None:
            continue
        fs = item[4]
        if not _is_cached(item[0], targets, fs):
            to_extract.append(item)

    print(f"Already cached: {len(unique_items) - len(to_extract)}")
    print(f"To extract: {len(to_extract)}")

    if not to_extract:
        print("All frames already cached.")
        return

    # Extract in parallel
    success = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract_one, item): item for item in to_extract}
        with tqdm(total=len(to_extract), desc="Extracting frames") as pbar:
            for future in as_completed(futures):
                ok, err = future.result()
                if ok:
                    success += 1
                else:
                    errors += 1
                    if errors <= 5:
                        item = futures[future]
                        print(f"  Error: {item[0]} @ {item[1]}s: {err}")
                pbar.update(1)

    print(f"\nDone: {success} extracted, {errors} errors")


if __name__ == "__main__":
    main()
