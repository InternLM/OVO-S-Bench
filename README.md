# OVO-S-Bench

**OVO-S-Bench** is a hierarchical benchmark for streaming spatial intelligence
in multimodal LLMs — 1,695 questions across 4 levels of difficulty, sampled
from 7 public video datasets (Ego4D, RoomTour3D, CODa, OmniWorld, VSI-Bench,
ARKitScenes, Honda, plus Sekai / YouTube clips).

This repository hosts the evaluation framework. The benchmark data (videos +
annotations parquet) is distributed via **🤗 HuggingFace Datasets**:

```
🤗 InternLM/OVO-S-Bench  (TBD — link added once the dataset is public)
```

> Project page: <https://internlm.github.io/OVO-S-Bench>

---

## Quick start

```bash
# 1) Install dependencies (API-only setup; ~1 min)
pip install -r requirements.txt

# (Optional) for offline open-source models via vLLM:
pip install -r requirements-vllm.txt

# 2) Configure API keys (only what you'll actually use)
cp .env.example .env
$EDITOR .env

# 3) Download the benchmark from HuggingFace
#    Layout produced:
#      data/ovo_s_bench_l1_l4.parquet
#      data/videos/...
huggingface-cli download InternLM/OVO-S-Bench --repo-type dataset --local-dir ./data

# 4) Run inference (auto-resumes on rerun)
python inference.py --model gpt-4o --annotation data/ovo_s_bench_l1_l4.parquet

# 5) Score
python score.py --result results/gpt-4o/ovo_s_bench_l1_l4.json
```

For multi-GPU local inference of a vLLM model:

```bash
python launch.py --model qwen3-vl-32b --annotation data/ovo_s_bench_l1_l4.parquet --gpus 8
```

---

## Repository layout

```
OVO-S-Bench/
├── inference.py            # Main entry point
├── score.py                # Compute per-category accuracy
├── launch.py               # Multi-GPU dynamic sharding launcher
├── precache.py             # CPU-only frame pre-extraction
├── merge_results.py        # Merge per-rank result shards
├── prompts.py              # Pluggable prompt templates
├── annotation_utils.py     # Parquet/JSON annotation loader
├── option_utils.py         # Image-option helpers (§4.3 PNG options)
│
├── config.yaml             # Models + sampling defaults
├── models/
│   ├── base.py             # BaseModel interface
│   ├── api_models.py       # OpenAI / Gemini / Claude
│   ├── vllm_models.py      # Qwen3-VL, Qwen3.5
│   ├── internvl_models.py  # InternVL3.5
│   ├── llava_onevision_vllm_models.py
│   ├── minicpmv_models.py
│   └── extras/             # Models requiring upstream cloning
│       ├── README.md       # BYO install instructions
│       ├── hermes_models.py
│       ├── fluxmem_models.py
│       └── ...
├── utils/
│   ├── frame_utils.py      # Video → frame extraction (decord + cv2 fallback)
│   └── config_utils.py     # Nested → flat config resolution
│
├── scripts/
│   ├── eval_api.sh         # Example: run a single API model
│   └── eval_vllm.sh        # Example: run a single vLLM model
├── data/
│   └── README.md           # Points to the HF dataset
└── docs/
    ├── adding_models.md
    ├── frame_caching.md
    └── benchmarking_protocol.md
```

---

## Task taxonomy

| Level | Description                          | Subcategories |
| ----- | ------------------------------------ | ------------- |
| L1    | Spatial perception                   | 1.1.x metric, 1.2.x topological, 1.3.x dynamic |
| L2    | Scene understanding                  | 2.1.x recognition, 2.2.x memory, 2.3.x temporal |
| L3    | Spatial-temporal multi-hop reasoning | 3.1.x / 3.2.x / 3.3.x |
| L4    | Compositional reasoning              | 4.1 directional, 4.2 topological, 4.3 trajectory matching (image options) |

L4 questions carry **image options** (base64 PNGs embedded directly in the
parquet's `options` field). Models that don't natively support per-option
images fall back to `option_utils.append_option_images_to_frames`, which
appends each option image as an extra frame at the end of the visual input.

---

## How models are registered

Each model wrapper inherits from `models.base.BaseModel` and implements
`inference(frames: List[PIL.Image], prompt: str) -> str`. Wrappers are
registered in:

- `models/api_models.py` — `MODEL_REGISTRY` (closed-source API providers)
- `models/vllm_models.py` — `_get_offline_registry()` (offline vLLM models)

Add a new model:

1. Subclass `BaseModel` in a new file under `models/` (or `models/extras/` if
   it needs an upstream repo cloned alongside).
2. Register it in the appropriate registry.
3. Add its config entry to `config.yaml`.

See [docs/adding_models.md](docs/adding_models.md) for the full walkthrough.

---

## Sharded inference

`inference.py` supports two parallelism modes:

```bash
# Threaded (API models)
python inference.py --model gpt-4o --annotation data/ovo_s_bench_l1_l4.parquet --workers 4

# Dynamic multi-process sharding (vLLM models on multi-GPU)
python launch.py --model qwen3-vl-32b --annotation data/ovo_s_bench_l1_l4.parquet --gpus 8
```

The launcher spawns N processes (auto-calculated from `gpus / tensor_parallel_size`),
each claiming queries from a shared `_queue.json` file via `fcntl` locks.
When all ranks finish, `merge_results.py` is invoked automatically.

---

## Citation

```bibtex
@misc{ovosbench2026,
  title  = {OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs},
  author = {OVO-S-Bench Team},
  year   = {2026},
  note   = {arXiv:TBD}
}
```

## License

MIT (see [LICENSE](LICENSE)). The benchmark annotations are released under
CC-BY-4.0; original video assets retain their respective source licenses.
