"""
Frame extraction utilities for OVO-S evaluation.

Strategy:
  1. On first access, extract target frames and cache as JPEG files.
  2. On subsequent accesses, load directly from cache (~0.1s vs ~35s).
  3. Uses decord for extraction (falls back to OpenCV).

Cache layout:
  {cache_dir}/{video_id}/{frame_idx}.jpg
"""

import os
import cv2
import hashlib
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
from pathlib import Path

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

# Default cache directory (can be overridden via set_cache_dir)
_CACHE_DIR: Optional[Path] = None


def set_cache_dir(path: str):
    """Set the frame cache directory."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_dir() -> Optional[Path]:
    """Get cache dir, auto-initialize if not set."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        # Auto-detect: use .frame_cache next to the eval directory
        default = Path(__file__).parent.parent / ".frame_cache"
        default.mkdir(parents=True, exist_ok=True)
        _CACHE_DIR = default
    return _CACHE_DIR


def _compute_target_frames(
    query_time: float, video_fps: float, max_frames: int, fps: float
) -> List[int]:
    """Compute sorted list of target frame indices to extract."""
    end_frame = int(query_time * video_fps)
    if end_frame <= 0:
        end_frame = 1

    fps_sample_count = int(query_time * fps) + 1

    if fps_sample_count < max_frames:
        sample_interval = video_fps / fps
        target_frames = []
        frame_pos = 0.0
        while frame_pos <= end_frame:
            target_frames.append(int(frame_pos))
            frame_pos += sample_interval
        if target_frames[-1] != end_frame:
            target_frames.append(end_frame)
    else:
        if max_frames == 1:
            target_frames = [end_frame]
        else:
            target_frames = np.linspace(
                0, end_frame, max_frames, dtype=int
            ).tolist()

    return sorted(set(target_frames))


def _resize_frame(frame_rgb: np.ndarray, frame_size: int) -> np.ndarray:
    """Resize frame so max dimension <= frame_size."""
    h, w = frame_rgb.shape[:2]
    if max(h, w) > frame_size:
        scale = frame_size / max(h, w)
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
    return frame_rgb


def _cache_key(video_path: str, target_frames: List[int], frame_size: int) -> str:
    """Build a unique cache filename for this exact extraction request."""
    vkey = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
    fkey = hashlib.md5(
        f"{sorted(target_frames)}_{frame_size}".encode()
    ).hexdigest()[:8]
    return f"{vkey}_{fkey}.npy"


def _try_load_from_cache(
    video_path: str, target_frames: List[int], frame_size: int
) -> Optional[List[Image.Image]]:
    """Load all frames from a single .npy cache file (uncompressed, fast)."""
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return None
    fpath = cache_dir / _cache_key(video_path, target_frames, frame_size)
    if not fpath.exists():
        return None
    try:
        # Load fully into memory (not mmap): mmap regions accumulate in RSS
        # because PIL Images keep references to the underlying buffer, and the
        # cgroup OOM-kills us at ~6 GB after a few dozen queries.
        # np.array(...) forces a copy so PIL Images don't hold the np.load buffer.
        arr = np.load(fpath)
        return [Image.fromarray(np.ascontiguousarray(arr[i])) for i in range(arr.shape[0])]
    except Exception:
        return None
    return frames


def _resize_cache_fallback_enabled() -> bool:
    value = os.getenv("OVOS_FRAME_CACHE_RESIZE_FALLBACK", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disable", "disabled"}


def _candidate_source_sizes(frame_size: int) -> List[int]:
    raw = os.getenv("OVOS_FRAME_CACHE_SOURCE_SIZES", "512,768,1024")
    sizes = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            size = int(item)
        except ValueError:
            continue
        if size > frame_size and size not in sizes:
            sizes.append(size)
    # Prefer the nearest larger cached size to minimize unnecessary resizing.
    return sorted(sizes)


def _try_derive_from_larger_cache(
    video_path: str, target_frames: List[int], frame_size: int
) -> Optional[List[Image.Image]]:
    """Create this cache entry by resizing an existing larger-frame cache."""
    if not _resize_cache_fallback_enabled():
        return None

    for source_size in _candidate_source_sizes(frame_size):
        source_frames = _try_load_from_cache(video_path, target_frames, source_size)
        if source_frames is None:
            continue

        resized_frames = []
        for img in source_frames:
            frame_rgb = np.array(img)
            frame_rgb = _resize_frame(frame_rgb, frame_size)
            resized_frames.append(Image.fromarray(np.ascontiguousarray(frame_rgb)))
        _save_to_cache(video_path, target_frames, resized_frames, frame_size)
        if os.getenv("OVOS_FRAME_CACHE_VERBOSE"):
            print(
                "Derived frame cache "
                f"frame_size={frame_size} from frame_size={source_size}: {video_path}"
            )
        return resized_frames

    return None


def _save_to_cache(
    video_path: str, target_frames: List[int],
    frames: List[Image.Image], frame_size: int
):
    """Save all frames as a single .npy array (uncompressed, fast mmap read)."""
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return
    fpath = cache_dir / _cache_key(video_path, target_frames, frame_size)
    if fpath.exists():
        return
    try:
        # Stack into (N, H, W, 3) — pad to uniform size if needed
        arrays = [np.array(img) for img in frames]
        # All frames should be same size after resize, but be safe
        max_h = max(a.shape[0] for a in arrays)
        max_w = max(a.shape[1] for a in arrays)
        padded = np.zeros((len(arrays), max_h, max_w, 3), dtype=np.uint8)
        for i, a in enumerate(arrays):
            padded[i, :a.shape[0], :a.shape[1]] = a
        np.save(fpath, padded)
    except Exception:
        pass


def _extract_frames_raw(
    video_path: str, target_frames: List[int], frame_size: int
) -> List[Image.Image]:
    """Extract frames from video (decord preferred, OpenCV fallback)."""
    extract_backend = os.getenv("OVOS_FRAME_EXTRACT_BACKEND", "").lower()
    use_decord = HAS_DECORD and extract_backend not in {"opencv", "cv2"}
    if use_decord:
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=4)
            total = len(vr)
            if total <= 0:
                return []
            indices = sorted({min(max(int(f), 0), total - 1) for f in target_frames})
            batch = vr.get_batch(indices).asnumpy()
            frames = []
            for frame_rgb in batch:
                frame_rgb = _resize_frame(frame_rgb, frame_size)
                frames.append(Image.fromarray(frame_rgb))
            return frames
        except Exception as e:
            print(f"Warning: decord failed ({e}), falling back to OpenCV")

    # OpenCV fallback
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            target_frames = sorted(
                {min(max(int(fidx), 0), total - 1) for fidx in target_frames}
            )
        frames = []
        for fidx in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_frame(frame_rgb, frame_size)
            frames.append(Image.fromarray(frame_rgb))
        return frames
    finally:
        cap.release()


def _get_video_fps(video_path: str) -> float:
    """Get video FPS using decord or OpenCV."""
    fps_backend = os.getenv("OVOS_FRAME_FPS_BACKEND", "").lower()
    use_decord = HAS_DECORD and fps_backend not in {"opencv", "cv2"}
    if use_decord:
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            fps = vr.get_avg_fps()
            del vr
            return fps
        except Exception:
            pass
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def extract_frames_for_query(
    video_path: str,
    query_time: float,
    max_frames: int = 64,
    fps: float = 1.0,
    frame_size: int = 512
) -> List[Image.Image]:
    """
    Extract frames for a single query point.

    First checks disk cache; on miss, extracts via decord/OpenCV and
    caches the result. Subsequent calls for the same video+frames
    return in ~0.1s instead of ~35s.
    """
    if not Path(video_path).exists():
        print(f"Warning: Video not found: {video_path}")
        return []

    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        print(f"Warning: Invalid FPS for video: {video_path}")
        return []

    targets = _compute_target_frames(query_time, video_fps, max_frames, fps)

    # Try cache first
    cached = _try_load_from_cache(video_path, targets, frame_size)
    if cached is not None:
        return cached

    derived = _try_derive_from_larger_cache(video_path, targets, frame_size)
    if derived is not None:
        return derived

    # Cache miss — extract and save whatever frames are readable.  Some
    # videos cannot seek all requested indices; caching the partial result
    # avoids repeatedly live-decoding the same problematic segment.
    frames = _extract_frames_raw(video_path, targets, frame_size)
    if frames:
        _save_to_cache(video_path, targets, frames, frame_size)

    return frames


def extract_frames_fixed_count(
    video_path: str,
    query_time: float,
    nframes: int = 128,
    frame_size: int = 512,
) -> List[Image.Image]:
    """
    Extract a fixed number of frames uniformly from [0, query_time].

    Uses np.linspace to sample exactly `nframes` frames.  Designed for
    video-type model input where a fixed frame count is required.

    Args:
        video_path: Path to the video file.
        query_time: End time in seconds (start is always 0).
        nframes:    Exact number of frames to extract.
        frame_size: Max dimension for frame resize.

    Returns:
        List of PIL Image frames (length == nframes, or fewer on error).
    """
    if not Path(video_path).exists():
        print(f"Warning: Video not found: {video_path}")
        return []

    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        print(f"Warning: Invalid FPS for video: {video_path}")
        return []

    end_frame = int(query_time * video_fps)
    if end_frame <= 0:
        end_frame = 1

    target_frames = sorted(set(np.linspace(0, end_frame, nframes, dtype=int).tolist()))

    # Try cache first
    cached = _try_load_from_cache(video_path, target_frames, frame_size)
    if cached is not None:
        return cached

    derived = _try_derive_from_larger_cache(video_path, target_frames, frame_size)
    if derived is not None:
        return derived

    # Cache miss — extract and save whatever frames are readable.  Some
    # videos cannot seek all requested indices; caching the partial result
    # avoids repeatedly live-decoding the same problematic segment.
    frames = _extract_frames_raw(video_path, target_frames, frame_size)
    if frames:
        _save_to_cache(video_path, target_frames, frames, frame_size)

    return frames


def _extract_at_indices(
    video_path: str,
    target_frames: List[int],
    frame_size: int,
) -> List[Image.Image]:
    """Shared cache→derive→raw→save path used by all fixed-set samplers."""
    target_frames = sorted(set(int(t) for t in target_frames))
    cached = _try_load_from_cache(video_path, target_frames, frame_size)
    if cached is not None:
        return cached
    derived = _try_derive_from_larger_cache(video_path, target_frames, frame_size)
    if derived is not None:
        return derived
    frames = _extract_frames_raw(video_path, target_frames, frame_size)
    if frames:
        _save_to_cache(video_path, target_frames, frames, frame_size)
    return frames


def extract_single_frame_at_query(
    video_path: str,
    query_time: float,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `single@query`: a single frame closest to query_time."""
    if not Path(video_path).exists():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    idx = max(0, int(query_time * video_fps))
    return _extract_at_indices(video_path, [idx], frame_size)


def extract_recent_window(
    video_path: str,
    query_time: float,
    nframes: int = 16,
    window_fps: float = 4.0,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `nearest-Nf@Kfps`: N frames at K fps ending at query_time.

    Default 16 frames at 4 fps → covers the last ~4 seconds before the query.
    Always anchors the last sample at query_time; pads to the start if the
    requested window extends earlier than the video itself.
    """
    if not Path(video_path).exists():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    end_frame = max(0, int(query_time * video_fps))
    step_frames = max(1, int(round(video_fps / window_fps)))
    start_frame = max(0, end_frame - (nframes - 1) * step_frames)
    target_frames = list(range(start_frame, end_frame + 1, step_frames))
    if len(target_frames) > nframes:
        target_frames = target_frames[-nframes:]
    return _extract_at_indices(video_path, target_frames, frame_size)


def extract_evidence_only(
    video_path: str,
    query_time: float,
    evidence_times: Optional[List[List[float]]] = None,
    nframes: int = 128,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `oracle-evidence`: nframes uniformly sampled inside the
    annotated evidence interval(s). Falls back to uniform-N over [0, query_time]
    if no evidence_times are provided (so the policy degrades gracefully on
    annotations missing evidence).
    """
    if not Path(video_path).exists():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    end_frame = max(1, int(query_time * video_fps))

    # Build a sorted union of evidence intervals (in frame indices). Treat any
    # malformed entry as empty.
    spans: List[Tuple[int, int]] = []
    for span in (evidence_times or []):
        try:
            s_t, e_t = float(span[0]), float(span[1])
        except Exception:
            continue
        s_f = max(0, int(s_t * video_fps))
        e_f = max(s_f, min(end_frame, int(e_t * video_fps)))
        if e_f > s_f:
            spans.append((s_f, e_f))
    if not spans:
        # No evidence available — degrade to uniform-N over full prefix.
        targets = sorted(set(np.linspace(0, end_frame, nframes, dtype=int).tolist()))
        return _extract_at_indices(video_path, targets, frame_size)

    # Allocate the nframes budget proportional to each span's length, but give
    # every span at least 1 frame so very short evidence windows still appear.
    lengths = [e - s for s, e in spans]
    total = sum(lengths) or 1
    raw_alloc = [max(1, round(nframes * L / total)) for L in lengths]
    # Trim back to exactly nframes, removing from the longest span first.
    diff = sum(raw_alloc) - nframes
    while diff > 0:
        i = max(range(len(raw_alloc)), key=lambda k: raw_alloc[k])
        if raw_alloc[i] > 1:
            raw_alloc[i] -= 1
            diff -= 1
        else:
            break

    targets: List[int] = []
    for (s_f, e_f), n in zip(spans, raw_alloc):
        n = max(1, n)
        targets += np.linspace(s_f, e_f, n, dtype=int).tolist()
    return _extract_at_indices(video_path, sorted(set(targets)), frame_size)


def extract_log_decay(
    video_path: str,
    query_time: float,
    nframes: int = 128,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `log-decay-N`: 60% of the budget within last 30 s before
    query_time, 30% in the 30 s – 5 min window, 10% earlier. Each band is
    sampled uniformly in its own range. Falls back gracefully to the longer
    band when the prefix is shorter than the band boundary.
    """
    if not Path(video_path).exists():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    end_frame = max(1, int(query_time * video_fps))
    near_start = max(0, end_frame - int(30 * video_fps))      # last 30 s
    mid_start = max(0, end_frame - int(300 * video_fps))      # last 5 min
    far_start = 0

    bands = [
        (near_start, end_frame, 0.60),
        (mid_start, near_start, 0.30),
        (far_start, mid_start, 0.10),
    ]
    targets: List[int] = []
    leftover = nframes
    for i, (s_f, e_f, frac) in enumerate(bands):
        if e_f <= s_f:
            continue
        n_band = nframes - len(targets) if i == len(bands) - 1 else int(round(nframes * frac))
        n_band = min(n_band, leftover)
        if n_band <= 0:
            continue
        targets += np.linspace(s_f, e_f - 1, n_band, dtype=int).tolist()
        leftover -= n_band
        if leftover <= 0:
            break
    targets = sorted(set(targets))
    if not targets:
        targets = [0]
    return _extract_at_indices(video_path, targets, frame_size)


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


# Legacy function for backward compatibility
def extract_frames_for_annotation(
    video_path: str,
    query_times: List[float],
    evidence_times: List[List[float]] = None,
    max_frames: int = 8,
    fps: float = 1.0,
    frame_size: int = 512,
    **kwargs
) -> List[Image.Image]:
    """Legacy wrapper — extracts frames for the first query time only."""
    if not query_times:
        return []
    return extract_frames_for_query(
        video_path, query_times[0], max_frames, fps, frame_size
    )
