#!/usr/bin/env python3
"""
Main inference entry point for OVO-S evaluation.

Each annotation may have multiple query_times, and each query_time corresponds
to a separate question with its own ground truth answer. The total number of
questions equals the sum of all query_times across all annotations.

Usage:
    python inference.py --model gpt-4o --annotation ../annotation/test_result/level_1.json
    python inference.py --model gemini-2.0-flash --annotation ../annotation/result/level_1.json --limit 10
"""

import os
import sys
import json
import argparse


def _maybe_early_cuda_init() -> None:
    if os.getenv("OVOS_EARLY_CUDA_INIT", "").lower() not in {"1", "true", "yes"}:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            print(f"[EARLY_CUDA_INIT] device={device} name={name}", flush=True)
    except Exception as exc:
        print(f"[EARLY_CUDA_INIT_FAILED] {type(exc).__name__}: {exc}", flush=True)
        raise


_maybe_early_cuda_init()

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.api_models import create_model
from annotation_utils import load_annotations as _load_annotations
from utils.frame_utils import (
    extract_frames_for_query,
    extract_frames_fixed_count,
    extract_single_frame_at_query,
    extract_recent_window,
    extract_evidence_only,
    extract_log_decay,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OVO-S Benchmark Evaluation")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (must be defined in config.yaml)"
    )
    parser.add_argument(
        "--annotation",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to annotation JSON/parquet file(s). When multiple are "
             "given, the model is loaded once and each annotation is processed "
             "sequentially with its own output file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Only honored when a single --annotation is "
             "given; with multiple annotations, outputs are auto-named per "
             "annotation under results/{model}/."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of queries to process (0 = all)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Filter by task subcategories (e.g., 1.1.1 1.1.2)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoint"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for API calls (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size for offline models (0 = auto from config, 1 = sequential)"
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default=None,
        help="Prompt template name (default: from model config or 'default')"
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default=None,
        nargs="+",
        choices=["auto", "fps", "fixed",
                 "single_at_query", "recent_window",
                 "evidence_only", "log_decay"],
        help="Frame sampling strategy. Multiple values allowed; in that case "
             "each policy is run sequentially within the same model load "
             "(used by §4.3.2 frame-sampling sensitivity)."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Process rank for multi-GPU sharding (0-indexed)"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of processes for multi-GPU sharding"
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=None,
        nargs="+",
        help="Override nframes for fixed sampling (implies --sampling-strategy fixed). "
             "Multiple values allowed; must align with --sampling-strategy list."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=0,
        help="Override tensor_parallel_size from config (0 = use config value)"
    )
    parser.add_argument(
        "--results-dir-suffix",
        type=str,
        default=None,
        nargs="+",
        help="Append this suffix to the model dir under results/ (e.g. "
             "'__evidence_only'). Used by §4.3.2 frame-sampling ablations to "
             "differentiate policies that share the same nframes (e.g. "
             "evidence_only n=128 vs uniform-128) by writing to a sibling dir. "
             "Multiple values align with --sampling-strategy.",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_annotations(annotation_path: str) -> list:
    """Load annotations from a JSON or Parquet file."""
    return _load_annotations(annotation_path)


def expand_annotations_to_queries(annotations: list) -> list:
    """
    Expand annotations to individual queries.
    Each query_time becomes a separate query with its corresponding answer.

    Args:
        annotations: List of annotation dictionaries

    Returns:
        List of query dictionaries, each with a single query_time and answer
    """
    queries = []

    for ann in annotations:
        query_times = ann.get("query_times", [])

        # Get question, options, answers - check both direct and re_constructed formats
        reconstructed = ann.get("re_constructed", {})
        question = reconstructed.get("question") or ann.get("question", "")
        options = reconstructed.get("options") or ann.get("options", {})
        answers = reconstructed.get("answers") or ann.get("answers", [])

        # Ensure answers list matches query_times length
        if len(answers) < len(query_times):
            last_answer = answers[-1] if answers else ""
            answers = answers + [last_answer] * (len(query_times) - len(answers))

        for idx, query_time in enumerate(query_times):
            query_id = f"{ann['id']}_q{idx}"
            answer = answers[idx] if idx < len(answers) else answers[0] if answers else ""

            query = {
                "query_id": query_id,
                "annotation_id": ann["id"],
                "query_index": idx,
                "source_dataset": ann.get("source_dataset", ""),
                "video_id": ann.get("video_id", ""),
                "video_path": ann.get("video_path", ""),
                "task_subcategory": ann.get("task_subcategory", ""),
                "task_type_name": ann.get("task_type_name", ""),
                "query_time": query_time,
                "question": question,
                "options": options,
                "ground_truth": answer
            }
            queries.append(query)

    return queries


def filter_queries(
    queries: list,
    tasks: list = None,
    limit: int = 0
) -> list:
    """Filter queries by task and limit."""
    filtered = queries

    # Filter by task subcategories
    if tasks:
        filtered = [q for q in filtered if q.get("task_subcategory") in tasks]

    # Apply limit
    if limit > 0:
        filtered = filtered[:limit]

    return filtered


def get_video_path(query: dict, config: dict) -> Path:
    """Get full video path for a query."""
    video_base = Path(config["PATHS"]["video_base_dir"])
    video_path = query.get("video_path", "")
    if video_path:
        path = Path(video_path)
        path = path if path.is_absolute() else video_base / path
        if path.exists():
            return path

    source = query.get("source_dataset", "")
    video_id = query.get("video_id", "")

    source_mapping = config.get("VIDEO_SOURCES", {})
    if source in source_mapping:
        source_dir = source_mapping[source]
        return video_base / source_dir / f"{video_id}.mp4"

    # Fallback to video_path field.
    return video_base / video_path


NO_FRAME_SAMPLING_STRATEGIES = {
    "none",
    "no_frames",
    "no-frames",
    "text_only",
    "text-only",
}


def _is_no_frame_sampling(sampling_strategy) -> bool:
    """Return True when an eval item intentionally removes video frames."""
    return str(sampling_strategy or "").lower() in NO_FRAME_SAMPLING_STRATEGIES


def _extract_frames_for_query(model, query, config, sampling_strategy="auto"):
    """Extract frames for a single query (CPU-bound work).

    Args:
        sampling_strategy: 'auto' (default, use model override if available),
                          'fps' (always fps-based sampling),
                          'fixed' (always fixed frame count),
                          'text_only'/'none' (skip sampled video frames).
    """
    if _is_no_frame_sampling(sampling_strategy):
        return []

    video_path = get_video_path(query, config)
    if sampling_strategy == "auto":
        if hasattr(model, 'extract_frames') and callable(model.extract_frames):
            return model.extract_frames(str(video_path), query["query_time"])
        return extract_frames_for_query(
            video_path=str(video_path),
            query_time=query["query_time"],
            max_frames=model.max_frames,
            fps=model.fps,
            frame_size=model.frame_size
        )
    elif sampling_strategy == "fixed":
        nframes = getattr(model, 'nframes', model.max_frames)
        return extract_frames_fixed_count(
            video_path=str(video_path),
            query_time=query["query_time"],
            nframes=nframes,
            frame_size=model.frame_size
        )
    elif sampling_strategy == "single_at_query":
        return extract_single_frame_at_query(
            video_path=str(video_path),
            query_time=query["query_time"],
            frame_size=model.frame_size,
        )
    elif sampling_strategy == "recent_window":
        nframes = getattr(model, 'nframes', None) or model.max_frames or 16
        # Default to 4 fps window unless model config carries an override.
        window_fps = float(getattr(model, 'window_fps', None)
                           or model.config.get('window_fps', 4.0))
        return extract_recent_window(
            video_path=str(video_path),
            query_time=query["query_time"],
            nframes=int(nframes),
            window_fps=window_fps,
            frame_size=model.frame_size,
        )
    elif sampling_strategy == "evidence_only":
        nframes = getattr(model, 'nframes', model.max_frames) or 128
        return extract_evidence_only(
            video_path=str(video_path),
            query_time=query["query_time"],
            evidence_times=query.get("evidence_times"),
            nframes=int(nframes),
            frame_size=model.frame_size,
        )
    elif sampling_strategy == "log_decay":
        nframes = getattr(model, 'nframes', model.max_frames) or 128
        return extract_log_decay(
            video_path=str(video_path),
            query_time=query["query_time"],
            nframes=int(nframes),
            frame_size=model.frame_size,
        )
    else:  # "fps"
        return extract_frames_for_query(
            video_path=str(video_path),
            query_time=query["query_time"],
            max_frames=model.max_frames,
            fps=model.fps,
            frame_size=model.frame_size
        )


def _model_accepts_kwarg(method, kwarg: str) -> bool:
    """Check whether *method* accepts *kwarg* explicitly or via **kwargs."""
    import inspect

    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.name == kwarg:
            return True
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _call_inference(model, frames, prompt, option_images=None):
    """Call model.inference, forwarding option_images when supported."""
    if option_images and _model_accepts_kwarg(model.inference, "option_images"):
        return model.inference(frames, prompt, option_images=option_images)
    return model.inference(frames, prompt)


def _call_batch_inference(model, batch_frames, batch_prompts, batch_option_images=None):
    """Call model.batch_inference, forwarding option images when supported."""
    has_any_opt = batch_option_images and any(batch_option_images)
    if has_any_opt and _model_accepts_kwarg(model.batch_inference, "batch_option_images"):
        return model.batch_inference(
            batch_frames, batch_prompts, batch_option_images=batch_option_images
        )
    return model.batch_inference(batch_frames, batch_prompts)


def _make_result(query, response, num_frames):
    """Build a result dict for a query."""
    return {
        "query_id": query["query_id"],
        "annotation_id": query["annotation_id"],
        "query_index": query["query_index"],
        "task_subcategory": query["task_subcategory"],
        "task_type_name": query["task_type_name"],
        "query_time": query["query_time"],
        "question": query["question"],
        "options": query["options"],
        "response": response,
        "ground_truth": query["ground_truth"],
        "num_frames": num_frames,
    }


def _make_error_result(query, error):
    return {
        "query_id": query["query_id"],
        "annotation_id": query["annotation_id"],
        "error": str(error),
        "response": None,
        "ground_truth": query["ground_truth"],
    }


def run_inference(
    model,
    queries: list,
    config: dict,
    output_path: Path,
    resume: bool = True,
    num_workers: int = 1,
    batch_size: int = 0,
    prompt_style: str = None,
    sampling_strategy: str = "auto",
    rank: int = 0,
    world_size: int = 1,
    reset_queue: bool = None,
    extra_result_fields: dict = None,
):
    """Run inference with batching and pipelining for offline models.

    For offline models with batch_inference():
      - Extracts frames for the next batch in a thread pool while the
        current batch is running on GPU (pipelining).
      - Calls batch_inference() to let vLLM's continuous batching work.

    For API models or models without batch_inference():
      - Falls back to the original sequential / threaded approach.

    When world_size > 1, uses a shared task queue so multiple processes
    can dynamically claim work (no static partitioning).
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # Load existing results if resuming
    results = {
        "model": model.model_name,
        "model_config": model.config,
        "prompt_style": prompt_style or "default",
        "started_at": datetime.now().isoformat(),
        "results": []
    }
    if extra_result_fields:
        results.update(extra_result_fields)
    processed_ids = set()

    if resume and output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        results["results"] = existing.get("results", [])
        processed_ids = {r["query_id"] for r in results["results"]}
        print(f"[rank {rank}] Resuming from {len(processed_ids)} processed queries")

    # ── Multi-process sharding (world_size > 1) ────────────────────────
    if world_size > 1:
        return _run_sharded(
            model, queries, config, results, output_path,
            processed_ids, prompt_style, sampling_strategy,
            batch_size, num_workers, rank, world_size,
            reset_queue=(not resume if reset_queue is None else reset_queue),
        )

    # ── Single-process path (original) ────────────────────────────────
    to_process = [q for q in queries if q["query_id"] not in processed_ids]

    save_interval = config["EVAL"].get("save_interval", 10)
    use_streaming = (
        getattr(model, "supports_video_streaming", False)
        and not _is_no_frame_sampling(sampling_strategy)
    )
    use_batch = (
        hasattr(model, "batch_inference")
        and batch_size != 1
    )

    if use_streaming:
        _run_streaming(model, to_process, config, results,
                       output_path, save_interval, prompt_style)
    elif use_batch:
        _run_batched(model, to_process, config, results,
                     output_path, save_interval, batch_size, prompt_style,
                     sampling_strategy)
    elif num_workers > 1:
        _run_threaded(model, to_process, config, results,
                      output_path, save_interval, num_workers, prompt_style,
                      sampling_strategy)
    else:
        _run_sequential(model, to_process, config, results,
                        output_path, save_interval, prompt_style,
                        sampling_strategy)

    results["completed_at"] = datetime.now().isoformat()
    results["total_queries"] = len(results["results"])
    save_results(results, output_path)
    return results


def _run_streaming(model, to_process, config, results, output_path, save_interval,
                    prompt_style=None):
    """Video-level streaming for models with supports_video_streaming=True.

    Groups queries by video, sorts each group by query_time, streams the video
    once and branches at each query point.
    """
    from tqdm import tqdm
    from collections import defaultdict

    # Group queries by (source_dataset, video_id)
    video_groups = defaultdict(list)
    for q in to_process:
        key = (q["source_dataset"], q["video_id"])
        video_groups[key].append(q)

    # Sort each group by query_time
    for key in video_groups:
        video_groups[key].sort(key=lambda q: q["query_time"])

    total_queries = len(to_process)
    completed = 0
    print(f"Processing {total_queries} queries across {len(video_groups)} videos (streaming)...")

    for (source, vid), queries in tqdm(video_groups.items(), desc="Videos"):
        video_path = get_video_path(queries[0], config)
        if not Path(video_path).exists():
            print(f"Warning: Video not found: {video_path}")
            for q in queries:
                results["results"].append(_make_error_result(q, f"Video not found: {video_path}"))
            completed += len(queries)
            continue

        # Build prompts and attach to queries
        for q in queries:
            q["prompt"], q["option_images"] = model.prepare_prompt(
                q["question"], q["options"], prompt_style=prompt_style
            )

        try:
            pairs = model.stream_video_inference(str(video_path), queries)
            for q, response in pairs:
                results["results"].append(_make_result(q, response, 0))
        except Exception as e:
            print(f"Error streaming video {vid}: {e}")
            for q in queries:
                results["results"].append(_make_error_result(q, e))

        completed += len(queries)
        if completed % save_interval < len(queries):
            save_results(results, output_path)


def _run_sequential(model, to_process, config, results, output_path, save_interval,
                     prompt_style=None, sampling_strategy="auto"):
    """Sequential model inference with CPU/cache prefetching.

    The model is still called from the main thread only.  A small background
    pool prepares upcoming frames/prompts so GPU generation is less likely to
    idle on mmap/PIL/processor-adjacent CPU work.
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    n = len(to_process)
    if n == 0:
        print("No queries to process.")
        return

    prefetch_workers = max(0, int(os.getenv("OVOS_PREFETCH_WORKERS", "1")))
    prefetch_window = max(1, int(os.getenv("OVOS_PREFETCH_WINDOW", "2")))

    def prepare(query):
        try:
            frames = _extract_frames_for_query(model, query, config, sampling_strategy)
            prompt, option_images = model.prepare_prompt(
                query["question"], query["options"], prompt_style=prompt_style
            )
            return query, frames, prompt, option_images, None
        except Exception as e:
            return query, None, None, None, e

    if prefetch_workers <= 0:
        print(f"Processing {n} queries sequentially (prefetch disabled)...")
        iterator = ((prepare(q)) for q in to_process)
        for i, (query, frames, prompt, option_images, prep_error) in enumerate(tqdm(iterator, total=n, desc="Evaluating")):
            _run_prepared_query(model, results, query, frames, prompt, prep_error,
                                option_images=option_images)
            if (i + 1) % save_interval == 0:
                save_results(results, output_path)
        return

    print(
        f"Processing {n} queries sequentially "
        f"(prefetch_workers={prefetch_workers}, window={prefetch_window})..."
    )
    with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        pending = deque()
        next_index = 0
        initial = min(n, prefetch_window)
        for _ in range(initial):
            pending.append(pool.submit(prepare, to_process[next_index]))
            next_index += 1

        with tqdm(total=n, desc="Evaluating") as pbar:
            for i in range(n):
                future = pending.popleft()
                if next_index < n:
                    pending.append(pool.submit(prepare, to_process[next_index]))
                    next_index += 1

                query, frames, prompt, option_images, prep_error = future.result()
                _run_prepared_query(model, results, query, frames, prompt, prep_error,
                                    option_images=option_images)
                if (i + 1) % save_interval == 0:
                    save_results(results, output_path)
                pbar.update(1)


def _run_prepared_query(model, results, query, frames, prompt, prep_error,
                        option_images=None):
    """Run one already-prepared query and append a result."""
    if prep_error is not None:
        print(f"Error preparing {query['query_id']}: {prep_error}")
        if os.getenv("OVOS_PRINT_ERROR_TRACEBACK"):
            import traceback

            traceback.print_exception(type(prep_error), prep_error, prep_error.__traceback__)
        results["results"].append(_make_error_result(query, prep_error))
        return

    try:
        if not frames:
            print(f"Warning: No frames for {query['query_id']}")
        response = _call_inference(model, frames, prompt, option_images=option_images)
        results["results"].append(_make_result(query, response, len(frames)))
    except Exception as e:
        print(f"Error processing {query['query_id']}: {e}")
        if os.getenv("OVOS_PRINT_ERROR_TRACEBACK"):
            import traceback

            traceback.print_exc()
        results["results"].append(_make_error_result(query, e))


def _run_threaded(model, to_process, config, results, output_path,
                  save_interval, num_workers, prompt_style=None,
                  sampling_strategy="auto"):
    """Threaded processing for API models."""
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results_lock = threading.Lock()
    print(f"Processing {len(to_process)} queries with {num_workers} workers...")

    def process_one(query):
        try:
            frames = _extract_frames_for_query(model, query, config, sampling_strategy)
            if not frames:
                print(f"Warning: No frames for {query['query_id']}")
            prompt, option_images = model.prepare_prompt(
                query["question"], query["options"], prompt_style=prompt_style
            )
            response = _call_inference(model, frames, prompt, option_images=option_images)
            return _make_result(query, response, len(frames))
        except Exception as e:
            print(f"Error processing {query['query_id']}: {e}")
            return _make_error_result(query, e)

    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, q): q for q in to_process}
        with tqdm(total=len(to_process), desc="Evaluating") as pbar:
            for future in as_completed(futures):
                result = future.result()
                with results_lock:
                    results["results"].append(result)
                    completed += 1
                    if completed % save_interval == 0:
                        save_results(results, output_path)
                pbar.update(1)


def _run_batched(model, to_process, config, results, output_path,
                 save_interval, batch_size, prompt_style=None,
                 sampling_strategy="auto"):
    """Batched + pipelined processing for offline models.

    Pipeline:
      1. Pre-extract frames for batch 0 (blocking)
      2. For each batch:
         a. Submit frame extraction for batch N+1 to thread pool
         b. Run batch_inference on batch N (GPU)
         c. Collect batch N+1 frames (join threads)
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    if batch_size <= 0:
        batch_size = model.config.get("batch_size", 1)

    n = len(to_process)
    if n == 0:
        print("No queries to process.")
        return

    print(f"Processing {n} queries in batches of {batch_size} (pipelined)...")

    # Split into batches
    batches = [to_process[i:i + batch_size] for i in range(0, n, batch_size)]
    if not batches:
        return

    # Frame extraction thread pool (CPU-bound, use multiple threads for I/O)
    frame_pool = ThreadPoolExecutor(max_workers=min(batch_size, 8))

    def extract_batch_frames(batch):
        """Extract frames for all queries in a batch using thread pool."""
        futures = [
            frame_pool.submit(_extract_frames_for_query, model, q, config,
                              sampling_strategy)
            for q in batch
        ]
        return [f.result() for f in futures]

    # Pre-extract first batch
    if not batches:
        print("No batches to process.")
        frame_pool.shutdown(wait=False)
        return

    current_frames = extract_batch_frames(batches[0])
    completed = 0

    with tqdm(total=n, desc="Evaluating (batched)") as pbar:
        for bi, batch in enumerate(batches):
            # Start extracting next batch in background (pipelining)
            next_frames_future = None
            if bi + 1 < len(batches):
                next_frames_future = frame_pool.submit(
                    extract_batch_frames, batches[bi + 1]
                )

            # Build prompts and decode any per-option base64 images.
            prompts: list = []
            batch_option_images: list = []
            for q in batch:
                p, opt_imgs = model.prepare_prompt(
                    q["question"], q["options"], prompt_style=prompt_style
                )
                prompts.append(p)
                batch_option_images.append(opt_imgs or None)

            # Run batch inference on GPU
            try:
                responses = _call_batch_inference(
                    model, current_frames, prompts,
                    batch_option_images=batch_option_images,
                )
            except Exception as e:
                print(f"Batch inference error: {e}, falling back to one-by-one batch")
                # After a cache crash the engine may be broken for single
                # inference() calls too, so retry via batch_inference with
                # size-1 lists which re-enters the same code path but with
                # a single item (less cache pressure).
                responses = []
                for frames, prompt, opt_imgs in zip(current_frames, prompts, batch_option_images):
                    try:
                        resp = _call_batch_inference(
                            model, [frames], [prompt],
                            batch_option_images=[opt_imgs],
                        )
                        responses.append(resp[0])
                    except Exception as e2:
                        print(f"  Single-item fallback also failed: {e2}")
                        responses.append(None)

            # Collect results
            for query, frames, response in zip(batch, current_frames, responses):
                if response is not None:
                    results["results"].append(
                        _make_result(query, response, len(frames))
                    )
                else:
                    results["results"].append(
                        _make_error_result(query, "inference returned None")
                    )
                completed += 1

            pbar.update(len(batch))

            if completed % save_interval < batch_size:
                save_results(results, output_path)

            # Wait for next batch frames (should already be done if GPU was busy)
            if next_frames_future is not None:
                current_frames = next_frames_future.result()

    frame_pool.shutdown(wait=False)


def save_results(results: dict, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Dynamic task sharding helpers (shared queue + file lock)
# ---------------------------------------------------------------------------

# Native wrappers process one query at a time, so small queue claims distribute
# work across same-GPU replicas instead of letting one fast rank drain a level.
CLAIM_BATCH_SIZE = int(os.getenv("OVOS_CLAIM_BATCH_SIZE", "1"))
QUEUE_WAIT_SECONDS = int(os.getenv("OVOS_QUEUE_WAIT_SECONDS", "1800"))
QUEUE_WAIT_INTERVAL_SECONDS = 0.5


def _queue_run_id() -> str:
    return os.getenv("OVOS_RUN_ID", "")


def _init_queue(queue_path: Path, lock_path: Path, all_query_ids: list):
    """Rank 0 initializes the queue file (only when it does not exist)."""
    import fcntl
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not queue_path.exists():
                queue = {"run_id": _queue_run_id(), "pending": all_query_ids, "completed": []}
                queue_path.write_text(json.dumps(queue))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _claim_batch(queue_path: Path, lock_path: Path, batch_size: int) -> list:
    """Atomically claim the next batch of query IDs from the shared queue."""
    import fcntl
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            queue = json.loads(queue_path.read_text())
            claimed = queue["pending"][:batch_size]
            queue["pending"] = queue["pending"][batch_size:]
            queue_path.write_text(json.dumps(queue))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return claimed


def _claim_video_batch(queue_path: Path, lock_path: Path) -> list:
    """Atomically claim the next video key from a video-level queue.

    Returns a list of query_ids belonging to that video, or [] when done.
    """
    import fcntl
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            queue = json.loads(queue_path.read_text())
            pending_videos = queue.get("pending_videos", [])
            if not pending_videos:
                return []
            video_key = pending_videos[0]
            queue["pending_videos"] = pending_videos[1:]
            query_ids = queue["video_to_queries"].get(video_key, [])
            queue_path.write_text(json.dumps(queue))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return query_ids


def _mark_completed(queue_path: Path, lock_path: Path, query_ids: list):
    """Mark query IDs as completed in the shared queue."""
    import fcntl
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            queue = json.loads(queue_path.read_text())
            queue["completed"].extend(query_ids)
            queue_path.write_text(json.dumps(queue))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _init_video_queue(queue_path: Path, lock_path: Path, queries: list):
    """Initialize a video-level queue for streaming models."""
    import fcntl
    from collections import defaultdict

    video_groups = defaultdict(list)
    for q in queries:
        key = f"{q['source_dataset']}|{q['video_id']}"
        video_groups[key].append(q["query_id"])

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not queue_path.exists():
                queue = {
                    "run_id": _queue_run_id(),
                    "pending_videos": list(video_groups.keys()),
                    "video_to_queries": dict(video_groups),
                    "completed": [],
                }
                queue_path.write_text(json.dumps(queue))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _run_sharded(
    model, all_queries, config, results, output_path,
    processed_ids, prompt_style, sampling_strategy,
    batch_size, num_workers, rank, world_size,
    reset_queue=False,
):
    """Dynamic sharding: all ranks claim work from a shared queue file.

    Each rank writes to its own output file (e.g. level_1_rank0.json).
    Use merge_results.py to combine afterwards.
    """
    from tqdm import tqdm
    from collections import defaultdict
    import time
    start_time = time.time()

    save_interval = config["EVAL"].get("save_interval", 10)
    use_streaming = (
        getattr(model, "supports_video_streaming", False)
        and not _is_no_frame_sampling(sampling_strategy)
    )
    use_batch = hasattr(model, "batch_inference") and batch_size != 1

    # Queue file paths — shared across all ranks
    annotation_stem = output_path.stem.split("_rank")[0]  # strip rank suffix
    queue_dir = output_path.parent
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_path = queue_dir / f"._queue_{annotation_stem}.json"
    lock_path = queue_dir / f"._queue_{annotation_stem}.lock"

    # Exclude already-processed queries
    remaining = [q for q in all_queries if q["query_id"] not in processed_ids]
    remaining_ids = [q["query_id"] for q in remaining]
    query_lookup = {q["query_id"]: q for q in remaining}

    # Rank 0 initializes the queue; other ranks wait briefly
    if rank == 0:
        if reset_queue:
            for stale_path in (queue_path, lock_path):
                try:
                    stale_path.unlink()
                except FileNotFoundError:
                    pass
        if use_streaming:
            _init_video_queue(queue_path, lock_path, remaining)
        else:
            _init_queue(queue_path, lock_path, remaining_ids)
    else:
        # With --no-resume, avoid racing on a stale queue from a stopped run.
        queue_ready = False
        wait_iters = max(1, int(QUEUE_WAIT_SECONDS / QUEUE_WAIT_INTERVAL_SECONDS))
        for _ in range(wait_iters):
            try:
                queue_mtime = queue_path.stat().st_mtime
                queue = json.loads(queue_path.read_text())
            except FileNotFoundError:
                queue_mtime = None
                queue = None
            except Exception:
                queue_mtime = None
                queue = None
            if queue_mtime is not None:
                queue_run_id = queue.get("run_id") if isinstance(queue, dict) else None
                if (
                    not reset_queue
                    or queue_run_id == _queue_run_id()
                    or (queue_run_id is None and queue_mtime >= start_time - 1)
                ):
                    queue_ready = True
                    break
            time.sleep(QUEUE_WAIT_INTERVAL_SECONDS)
        if not queue_ready:
            raise RuntimeError(
                f"Timed out after {QUEUE_WAIT_SECONDS}s waiting for queue: {queue_path}"
            )

    completed_count = 0
    print(f"[rank {rank}] Starting sharded inference (world_size={world_size})...")

    if use_streaming:
        # Streaming: claim one video at a time
        while True:
            claimed_ids = _claim_video_batch(queue_path, lock_path)
            if not claimed_ids:
                break
            batch_queries = [query_lookup[qid] for qid in claimed_ids if qid in query_lookup]
            if not batch_queries:
                continue

            # Process this video's queries with _run_streaming
            _run_streaming(model, batch_queries, config, results,
                           output_path, save_interval, prompt_style)

            _mark_completed(queue_path, lock_path, claimed_ids)
            completed_count += len(batch_queries)
            save_results(results, output_path)
    else:
        # Non-streaming: claim enough work to keep large GPU batches full.
        effective_batch_size = batch_size if batch_size > 0 else model.config.get("batch_size", 1)
        claim_batch_size = (
            max(CLAIM_BATCH_SIZE, effective_batch_size * 2)
            if use_batch else CLAIM_BATCH_SIZE
        )
        while True:
            claimed_ids = _claim_batch(queue_path, lock_path, claim_batch_size)
            if not claimed_ids:
                break
            batch_queries = [query_lookup[qid] for qid in claimed_ids if qid in query_lookup]
            if not batch_queries:
                continue

            # Process this batch using the appropriate method
            if use_batch:
                _run_batched(model, batch_queries, config, results,
                             output_path, save_interval, batch_size,
                             prompt_style, sampling_strategy)
            elif num_workers > 1:
                _run_threaded(model, batch_queries, config, results,
                              output_path, save_interval, num_workers,
                              prompt_style, sampling_strategy)
            else:
                _run_sequential(model, batch_queries, config, results,
                                output_path, save_interval, prompt_style,
                                sampling_strategy)

            _mark_completed(queue_path, lock_path, claimed_ids)
            completed_count += len(batch_queries)
            save_results(results, output_path)

    print(f"[rank {rank}] Finished. Processed {completed_count} queries this session.")
    results["completed_at"] = datetime.now().isoformat()
    results["total_queries"] = len(results["results"])
    results["rank"] = rank
    results["world_size"] = world_size
    save_results(results, output_path)
    return results


def main():
    args = parse_args()

    # Load environment variables
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load config
    config = load_config(args.config)

    # Resolve nested model config into flat dict
    from utils.config_utils import resolve_all_models
    all_models = resolve_all_models(config)

    # Check if model is defined
    if args.model not in all_models:
        print(f"Error: Model '{args.model}' not found in config.yaml")
        print(f"Available models: {sorted(all_models.keys())}")
        sys.exit(1)

    model_config = all_models[args.model]

    if args.tp_size > 0:
        print(f"Overriding tensor_parallel_size: {model_config.get('tensor_parallel_size')} -> {args.tp_size}")
        model_config = {**model_config, "tensor_parallel_size": args.tp_size}

    # Create model instance
    print(f"Initializing model: {args.model}")
    model = create_model(args.model, model_config)

    # ── Multi-policy support (§4.3.2 frame-sampling sensitivity) ──
    # Normalize CLI args (now nargs='+') to aligned lists, broadcasting
    # length-1 entries.  This lets one model-load run all 7 policies.
    strategies = list(args.sampling_strategy) if args.sampling_strategy else [None]
    nframes_list = list(args.nframes) if args.nframes else [0]
    suffix_list = list(args.results_dir_suffix) if args.results_dir_suffix else [""]
    max_pol = max(len(strategies), len(nframes_list), len(suffix_list))
    if len(strategies) == 1: strategies = strategies * max_pol
    if len(nframes_list) == 1: nframes_list = nframes_list * max_pol
    if len(suffix_list) == 1: suffix_list = suffix_list * max_pol
    if not (len(strategies) == len(nframes_list) == len(suffix_list) == max_pol):
        raise SystemExit(
            f"--sampling-strategy ({len(strategies)}), --nframes "
            f"({len(nframes_list)}), and --results-dir-suffix ({len(suffix_list)}) "
            f"must all be length 1 or the same length."
        )
    print(f"Running {max_pol} policy/policies in this model load:")
    for i, (s, n, sf) in enumerate(zip(strategies, nframes_list, suffix_list)):
        print(f"  policy {i+1}/{max_pol}: strategy={s!r} nframes={n} suffix={sf!r}")

    # Resolve prompt style once.
    prompt_style = args.prompt_style or model_config.get("prompt_style")

    annotation_paths = list(args.annotation)
    if args.output and (len(annotation_paths) > 1 or max_pol > 1):
        print("Warning: --output is ignored when multiple annotations or policies are given; "
              "outputs will be auto-named per (policy, annotation).")

    overall_summary: list[tuple[str, str, int]] = []
    for policy_idx, (cli_strategy, cli_nframes, cli_suffix) in enumerate(
        zip(strategies, nframes_list, suffix_list)
    ):
        print(f"\n{'#' * 70}")
        print(f"[POLICY {policy_idx + 1}/{max_pol}] "
              f"strategy={cli_strategy} nframes={cli_nframes} suffix={cli_suffix}")
        print(f"{'#' * 70}")

        # Apply per-policy nframes override to the loaded model.
        if cli_nframes:
            model.nframes = cli_nframes
            model.max_frames = cli_nframes

        # Resolve effective sampling strategy.
        if cli_nframes and not cli_strategy:
            sampling_strategy = "fixed"
        else:
            sampling_strategy = cli_strategy or model_config.get("sampling_strategy", "auto")

        for ann_idx, annotation_path in enumerate(annotation_paths):
            print(f"\n{'=' * 70}")
            print(f"[{ann_idx + 1}/{len(annotation_paths)}] Annotation: {annotation_path}")
            print(f"{'=' * 70}")

            try:
                # Load annotations
                print(f"Loading annotations from: {annotation_path}")
                annotations = load_annotations(annotation_path)
                print(f"Loaded {len(annotations)} annotations")

                # Expand annotations to queries (one per query_time)
                queries = expand_annotations_to_queries(annotations)
                print(f"Expanded to {len(queries)} queries")

                # Filter queries
                queries = filter_queries(queries, args.tasks, args.limit)
                print(f"After filtering: {len(queries)} queries")

                # Determine output path
                if args.output and len(annotation_paths) == 1 and max_pol == 1:
                    output_path = Path(args.output)
                else:
                    annotation_name = Path(annotation_path).stem
                    if cli_nframes:
                        annotation_name = f"{annotation_name}_n{cli_nframes}"
                    if args.world_size > 1:
                        fname = f"{annotation_name}_rank{args.rank}.json"
                    else:
                        fname = f"{annotation_name}.json"
                    output_path = Path(config["PATHS"]["results_dir"]) / (args.model + (cli_suffix or "")) / fname

                print(f"Output will be saved to: {output_path}")

                # Run inference
                results = run_inference(
                    model=model,
                    queries=queries,
                    config=config,
                    output_path=output_path,
                    resume=not args.no_resume,
                    num_workers=args.workers,
                    batch_size=args.batch_size,
                    prompt_style=prompt_style,
                    sampling_strategy=sampling_strategy,
                    rank=args.rank,
                    world_size=args.world_size,
                )

                n_done = len(results["results"])
                print(f"\nCompleted! Processed {n_done} queries")
                print(f"Results saved to: {output_path}")
                overall_summary.append((f"{cli_suffix or 'default'}::{annotation_path}", "ok", n_done))
            except KeyboardInterrupt:
                print(f"\nInterrupted on annotation {annotation_path}; "
                      "checkpoint is preserved.")
                overall_summary.append((annotation_path, "interrupted", 0))
                raise
            except Exception as exc:
                import traceback
                print(f"\n[ANNOTATION_FAILED] {annotation_path}: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                overall_summary.append((annotation_path, f"failed:{type(exc).__name__}", 0))
                print("Continuing with next annotation.")

    if len(annotation_paths) * max_pol > 1:
        print(f"\n{'=' * 70}")
        print("Run summary (policy::annotation)")
        print(f"{'=' * 70}")
        for path, status, n_done in overall_summary:
            print(f"  [{status:>20s}] n={n_done:<6d} {path}")
        failed = [p for p, s, _ in overall_summary if not s == "ok"]
        if failed:
            print(f"\n{len(failed)}/{len(overall_summary)} (policy, annotation) cells did not complete cleanly.")
            sys.exit(1)


if __name__ == "__main__":
    main()
