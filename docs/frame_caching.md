# Frame caching

`utils/frame_utils.py` extracts and caches video frames so that re-running
the same evaluation (resume, different model, different prompt) doesn't pay
the decode cost twice.

## Where the cache lives

- Default: `.frame_cache/` at the repo root (gitignored).
- Override via the `OVO_S_FRAME_CACHE` env var in `.env`.

The cache layout is:

```
.frame_cache/
└── <video_hash>_<extraction_params_hash>/
    ├── frame_000.npy
    ├── frame_001.npy
    └── ...
```

- `video_hash` is derived from the video path + file size + modification time.
- `extraction_params_hash` covers `query_time`, `nframes`, `fps`, `frame_size`,
  and the sampling strategy.

That means two different sampling strategies on the same video produce
distinct cache entries (no cross-contamination), but the same strategy
across resumes hits cache instantly.

## Cache size estimates

| max_frames | frame_size | Per-query (1 video) | Full benchmark (1695 q) |
| ---------- | ---------- | ------------------- | ----------------------- |
| 32         | 384        | ~6 MB               | ~10 GB                  |
| 64         | 448        | ~14 MB              | ~25 GB                  |
| 128        | 512        | ~35 MB              | ~60 GB                  |
| 256        | 512        | ~70 MB              | ~120 GB                 |

(Numbers assume RGB uint8 `.npy` arrays — no compression.)

## Pre-caching

For batch evaluations across many models, pre-cache once with the
CPU-only `precache.py` script:

```bash
python precache.py --annotation data/ovo_s_bench.parquet --model qwen3-vl-32b --workers 16
```

`precache.py` reads the model config to determine the sampling params, but
doesn't load the model weights — so it runs on any machine, no GPU needed.

## Cache invalidation

If you change `video_base_dir` in `config.yaml` (so the same video shows up
under a different absolute path), the cache key changes and you'll re-extract.
That's intentional: the framework can't trust that two paths point at the
same bytes without re-hashing the file.

To force a full re-extraction, delete `.frame_cache/`. To inspect which
videos are cached:

```bash
ls .frame_cache/ | wc -l
```
