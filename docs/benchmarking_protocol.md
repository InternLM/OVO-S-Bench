# Benchmarking protocol

OVO-S-Bench evaluates models in the **online streaming** protocol: each
question carries a `query_time` (seconds from the video start), and the model
is only allowed to use the video prefix `[0, query_time]`. This mirrors the
real-world setting where an agent answers questions about what it has
seen so far, not what it will see in the future.

## Frame sampling strategies

`inference.py --sampling-strategy {policy}` selects how frames are pulled
from each video prefix before they're passed to the model. Available policies:

| Strategy            | Behavior                                                                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `auto` (default)    | Delegates to the model's own `extract_frames()` if defined, else falls back to `fps`.                                            |
| `fps`               | Samples every `1/fps` seconds within `[0, query_time]` up to `max_frames` (default `--fps 1` → one frame per second).            |
| `fixed`             | Samples exactly `nframes` frames uniformly across `[0, query_time]`. Default for video-input vLLM models that expect a fixed N. |
| `single_at_query`   | One frame at `query_time` only — useful as a single-image baseline.                                                              |
| `recent_window`     | The last `nframes` frames at 4 fps from `[query_time - nframes/4, query_time]`. Tests short-horizon recency bias.                |
| `evidence_only`     | Only frames inside the annotated `evidence_times` intervals (oracle setting; bypass the online protocol).                        |
| `log_decay`         | `nframes` frames with logarithmic temporal density — denser toward `query_time`, sparser toward `t=0`.                            |

The §4.3.2 frame-sampling sensitivity ablation in our paper covers
`single_at_query / recent_window / evidence_only / log_decay` against the
default `fixed @ 128 frames` baseline.

## Frame caching

First-time extraction of 128 frames from a 5-minute video takes ~35 seconds on
CPU. To amortize, the framework caches the extracted frames as `.npy` arrays
under `.frame_cache/<video_hash>_<frame_hash>/`. Cached lookups are ~0.1 s.

Set `OVO_S_FRAME_CACHE` in `.env` to put the cache on fast local storage
(NVMe, tmpfs). For full benchmark runs the cache can reach ~50 GB.

## Decoder fallback

The frame extractor prefers `decord` (faster, better seek accuracy). If
decord can't open a file (e.g. AV1 codec), it falls back to OpenCV.

YouTube clips encoded in AV1 may need pre-transcoding to H.264 before
extraction works on the cv2 fallback — see `scripts/build_master_parquet.py`
for the list of affected videos.

## Online vs. offline split

A few models in `models/extras/` (HERMES, FluxMem, InfiniPot-V) implement
their own streaming pipelines that process the entire video prefix
incrementally rather than via uniform frame sampling. These set
`supports_video_streaming = True` and bypass the standard extraction path.
Their compression diagnostics (retained-time density, evidence recall) are
written to `results/<model>/<annotation>_retention.jsonl` when run via the
analysis tools (§4.3.4(b) of the paper).
