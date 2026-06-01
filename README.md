<p align="center">
  <img src="https://raw.githubusercontent.com/InternLM/OVO-S-Bench/webpage/static/images/teaser.png" alt="OVO-S-Bench teaser" width="90%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Tasks-1695-blue" alt="Tasks"/>
  <img src="https://img.shields.io/badge/Levels-4-purple" alt="Levels"/>
  <img src="https://img.shields.io/badge/Sources-9-green" alt="Datasets"/>
  <img src="https://img.shields.io/badge/Models-38-orange" alt="Models"/>
  <a href="https://internlm.github.io/OVO-S-Bench/"><img src="https://img.shields.io/badge/%F0%9F%8F%86_Leaderboard-OVO--S--Bench-8c2416" alt="Leaderboard"/></a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://huggingface.co/datasets/InternLM/OVO-S-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Dataset-yellow" alt="HF Dataset"/></a>
  <a href="https://internlm.github.io/OVO-S-Bench/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Project-Page-blue" alt="Project Page"/></a>
</p>

# OVO-S-Bench

> "Hierarchical streaming spatial intelligence — from instantaneous perception to allocentric mapping."

## Introduction

Multimodal agents in robotics, AR, and autonomous driving must reason about places and layouts from continuous egocentric streams, often using evidence outside the current view. Existing benchmarks either evaluate offline over full videos or target events rather than spatial structure. We introduce **OVO-S-Bench**, a fully human-annotated benchmark for streaming spatial intelligence, comprising **1,695 questions over 339 source videos**. Annotation involves **12 trained annotators** (each also serving as a blind cross-reviewer) across roughly **804 person-hours** of multi-round quality assurance. Each question carries a query timestamp and an evidence interval, and at evaluation the model sees only the prefix preceding the query. Questions span **four levels of increasing abstraction**: instantaneous egocentric perception, spatiotemporal context tracking, spatial simulation and reasoning, and allocentric mapping. Across **38 proprietary and open-source MLLMs**, Gemini-3.1-Pro trails human experts by **27 points (59.2 vs. 86.6)**, with allocentric mapping as the dominant bottleneck. Notably, streaming and spatially fine-tuned MLLMs underperform their own backbones. We further find that chain-of-thought reasoning amplifies spatial errors when ungrounded in the stream. By exposing these limitations, OVO-S-Bench establishes a demanding testbed for next-generation streaming spatial MLLMs.

## Four-Level Streaming Spatial Taxonomy

OVO-S-Bench organizes its 1,695 questions into **four cumulative levels** by the spatial state a model must access at query time. The progression goes from evidence in the current view to global-map queries requiring cross-viewpoint integration — a gradient of persistence and abstraction.

| Level | Capability | # Questions | Task families |
| --- | --- | :-: | --- |
| **L1** | Instantaneous Egocentric Perception | **629** | egocentric metric (distance, scale, clearance, height) · local spatial relations (containment, occlusion, support, layout) · dynamic spatial perception (ego/object motion) |
| **L2** | Spatiotemporal Context Tracking | **513** | scene revisit recognition · spatial memory beyond the view · chronological spatial memory |
| **L3** | Spatial Simulation and Reasoning | **279** | spatial simulation (reorientation, removal consequences, physical feasibility) · spatiotemporal consistency · spatial route planning |
| **L4** | Allocentric Spatial Mapping | **274** | allocentric direction reasoning · topological structure reasoning · **trajectory-map alignment** (image options) |

Mean prefix at query time: **8.8 min**. Evidence-span medians by level: **L1 2.0 s · L2 36.8 s · L3 2.0 s · L4 278.7 s** — reflecting the spatial persistence each level demands.

> L4.3 trajectory-matching questions ship their option images **embedded inline** as base64 data URIs in the parquet's `options` column. No separate image-asset download is needed.

## Why OVO-S-Bench?

Most video benchmarks evaluate models with the *entire* video available — useful for offline understanding but unrealistic for embodied agents. OVO-S-Bench enforces an **online streaming protocol**:

| Capability | What we test | Why it's hard |
| --- | --- | --- |
| **🎯 Streaming protocol** | Each query carries a `query_time`; only frames in `[0, query_time]` are visible | Models can't peek at later evidence — many existing video MLLMs implicitly assume the whole clip |
| **🧱 Hierarchical** | 4 levels × 30 canonical task types | Each level needs a capability the previous lacks; ablations isolate where reasoning breaks down |
| **🗺️ Allocentric reasoning** | L4 integrates egocentric stream into a global map (cardinal directions, room topology, BEV trajectory matching) | Dominates the gap to human performance; 28 of 34 systems score lowest on L4 |
| **🌍 Source diversity** | 9 datasets across 5 regimes (indoor walkthroughs, egocentric activities, outdoor/world, driving, 3D annotated) | Spatial reasoning must hold across drastically different visual statistics |
| **👥 Human-written** | 12 annotators · 804 person-hours · cross-review + LLM-probe shortcut filtering | No template-generated questions; distractors are visually plausible but evidentially wrong |

## Key Findings

| Finding | Numbers |
| --- | --- |
| **Significant gap with human performance** — strongest system **Gemini-3.1-Pro** reaches **59.2** overall, far below human experts under the same streaming protocol (**86.6**; **92.2** offline). Best open-source: **Qwen3-VL-235B-A22B** at **53.6**, trailing human-streaming by 33 pts. | 27 pts gap |
| **Allocentric mapping is the dominant bottleneck** — L4 is the lowest-scoring level for **28 of 34** systems, with a mean gap of **9.3%** between L1–L3 and L4. | 28 / 34 |
| **Closed-source advantage is narrow and uneven** — only **+5.6 pts** overall vs. best open-source. On L3, the best open-source backbone *exceeds* Gemini-3.1-Pro by **+5.3 pts**. | +5.6 / +5.3 |
| **Specialization hurts the backbone** — no streaming-architecture or spatially fine-tuned variant outperforms its comparable general backbone; **13 of 15** lag behind their own base (median −2.0, range −18.4 to +0.5). | 13 / 15 |
| **Chain-of-thought is double-edged** — explicit reasoning consistently helps L2 (mean Δ = **+3.9**, 8/9 pairs positive) but slightly hurts L1 (mean Δ = **−1.0**). 60–80% of CoT failures are mis-grounded visual evidence. | +3.9 / −1.0 |
| **Retention is not the bottleneck** — for HERMES, StreamingTOM, FluxMem, per-query Pearson between evidence recall and answer correctness is near zero. Compression methods don't lose useful evidence — they lose nothing yet still don't help. | r ≈ 0 |

## News

- **2026-06** Initial release: evaluation framework (this repo) + HuggingFace dataset.
- **2026-06** Project page online: <https://internlm.github.io/OVO-S-Bench/>.

---

## Leaderboard

Full interactive leaderboard with per-level breakdowns: <https://internlm.github.io/OVO-S-Bench/>

**Top systems (overall, streaming protocol):**

| Rank | Model | Org | Overall | L1 | L2 | L3 | L4 |
| :-: | --- | --- | :-: | :-: | :-: | :-: | :-: |
| 🥇 | **Human Expert (offline)** | — | **92.2** | — | — | — | — |
| 🥈 | Human Expert (streaming) | — | **86.6** | — | — | — | — |
| 🥉 | **Gemini-3.1-Pro** | Google DeepMind | **59.2** | — | — | 55.9 | — |
| 4 | Qwen3-VL-235B-A22B | Alibaba Cloud | 53.6 | — | — | **61.2** | — |
| · | Random | — | 31.3 | — | — | — | — |
| · | Text-Only baseline | — | 37.1 | — | — | — | — |

> Full table with all 38 systems and per-level scores: see the project page.

---

## Quick Start

### Install

```bash
git clone https://github.com/InternLM/OVO-S-Bench.git
cd OVO-S-Bench

# Default install (API providers only; ~1 min)
pip install -r requirements.txt

# Optional: open-source MLLMs via vLLM
pip install -r requirements-vllm.txt
```

### Download the benchmark

The annotations parquet + 339 source videos (~219 GiB) are hosted on HuggingFace Datasets:

```bash
pip install -U "huggingface_hub[cli]"
hf download InternLM/OVO-S-Bench --repo-type dataset --local-dir ./data
```

Layout you should see after download:

```
data/
├── ovo_s_bench_l1_l4.parquet     # 1695 questions, 35 MB
└── videos/                       # 339 .mp4 files, 219 GiB
    ├── Ego4D/...
    ├── RoomTour3D/...
    ├── annotated_videos/...
    └── ...
```

### Configure API keys

```bash
cp .env.example .env
$EDITOR .env       # fill in the provider(s) you'll use
```

### Run

**API model** (single-process, threaded):

```bash
bash scripts/eval_api.sh gpt-4o data/ovo_s_bench_l1_l4.parquet
```

**Open-source MLLM via vLLM** (multi-GPU auto-sharded):

```bash
GPUS=8 bash scripts/eval_vllm.sh qwen3-vl-32b data/ovo_s_bench_l1_l4.parquet
```

Both scripts call `inference.py` then `score.py`. Inference auto-resumes on rerun — interrupted jobs pick up from the last checkpoint.

### Add a new model

See [docs/adding_models.md](docs/adding_models.md). The contract is `BaseModel.inference(frames: List[PIL.Image], prompt: str) -> str`; add a config entry to `config.yaml::MODELS` and register the wrapper in `models/api_models.py` or `models/vllm_models.py`.

### Custom API endpoint (no HuggingFace / vendor lock-in)

Set `*_BASE_URL` in `.env` to point at your proxy or local vLLM server:

```bash
OPENAI_API_KEY=your-proxy-key
OPENAI_BASE_URL=http://localhost:8000/v1
```

API providers (`provider: openai`, `gemini`, `anthropic`) honor the standard `*_BASE_URL` env var convention.

---

## Check Results

Per-query responses land under:

```
results/<model>/ovo_s_bench_l1_l4.json
```

`score.py` aggregates accuracy by main category and subcategory:

```bash
python score.py --result results/gpt-4o/ovo_s_bench_l1_l4.json --verbose
```

Output structure (`--verbose`):

```
Level  Cat   #Q   Acc
L1     1.1   158   .58
L1     1.2   193   .61
L1     1.3   278   .54
L2     2.1   198   .49
...
Overall: 0.521 (883/1695)
```

For sharded multi-GPU runs, `merge_results.py` consolidates `*_rank{N}.json` shards into a single result file (auto-invoked by `launch.py`).

---

## Tasks

OVO-S-Bench's **30 canonical task types** across 4 levels:

| Level | Main | Subcategories |
| :-: | :-: | --- |
| **L1** | 1.1 — Metric | 1.1.1 absolute distance · 1.1.2 relative scale · 1.1.3 passability affordance · 1.1.4 viewpoint height |
| **L1** | 1.2 — Topological | 1.2.1 directional enumeration · 1.2.2 occlusion · 1.2.3 support · 1.2.4 containment |
| **L1** | 1.3 — Dynamic | 1.3.1 ego-translation · 1.3.2 ego-rotation · 1.3.3 object motion · 1.3.4 relative speed · 1.3.5 motion segmentation |
| **L2** | 2.1 — Scene recognition | 2.1.1 same-scene judgment · 2.1.2 revisit detection |
| **L2** | 2.2 — Spatial memory | 2.2.1 out-of-view localization · 2.2.2 state memory · 2.2.3 first-occurrence |
| **L2** | 2.3 — Temporal reasoning | 2.3.1 chronological order · 2.3.2 trajectory backtracing |
| **L3** | 3.1 — Spatial simulation | 3.1.1 reorientation · 3.1.2 removal consequences · 3.1.3 physical feasibility |
| **L3** | 3.2 — Consistency check | 3.2.1 edited-clip detection |
| **L3** | 3.3 — Route planning | 3.3.1 shortest-route · 3.3.2 obstacle avoidance · 3.3.3 multi-stop ordering |
| **L4** | 4.1 — Allocentric direction | global cardinal directional reasoning |
| **L4** | 4.2 — Topological structure | room-graph adjacency / connectivity |
| **L4** | 4.3 — Trajectory-map alignment | BEV trajectory ↔ video matching (**image options**) |

Source distribution (top): RoomTour3D (557), Ego4D (355), annotated_videos (287), OmniWorld (155), CODa_full (137), Sekai (109), VSI-Bench (22), arkitscenes (33), honda (20), edited_videos (20).

---

## Repository layout

```
OVO-S-Bench/
├── inference.py            # Main eval entry point
├── score.py                # Per-category accuracy aggregator
├── launch.py               # Multi-GPU dynamic-sharding launcher (auto-merges)
├── precache.py             # CPU-only frame pre-extraction
├── merge_results.py        # Shard merger
├── prompts.py              # Pluggable prompt templates (default/verbose/cot)
├── annotation_utils.py     # Parquet / JSON annotation loader
├── option_utils.py         # Image-option helpers (L4.3 PNG decoding)
├── config.yaml             # 84+ pre-registered model entries
├── models/
│   ├── base.py             # BaseModel ABC
│   ├── api_models.py       # OpenAI / Gemini / Claude
│   ├── vllm_models.py      # Qwen3-VL / Qwen3.5 / Gemma4 / GLM-4.6V
│   ├── internvl_models.py
│   ├── llava_onevision_vllm_models.py
│   ├── minicpmv_models.py
│   └── extras/             # 12 wrappers requiring upstream repos
│       ├── README.md       # BYO install instructions
│       └── *_models.py     # HERMES / FluxMem / StreamingTOM / InfiniPot-V / ...
├── utils/
│   ├── frame_utils.py      # decord + OpenCV fallback, .frame_cache logic
│   └── config_utils.py     # nested → flat config flatten
├── scripts/
│   ├── eval_api.sh
│   └── eval_vllm.sh
├── data/README.md          # HF dataset download instructions
└── docs/
    ├── adding_models.md
    ├── benchmarking_protocol.md
    └── frame_caching.md
```

---

## Citation

```bibtex
@article{ovosbench2026,
  title  = {OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs},
  author = {OVO-S-Bench Team},
  year   = {2026},
  note   = {arXiv:TBD}
}
```

Machine-readable citation metadata will land in `CITATION.cff` once the arXiv ID is assigned.

---

## Contributors

Coming with the arXiv release. See the [project page](https://internlm.github.io/OVO-S-Bench/) for the current author list.

---

## Acknowledgements

OVO-S-Bench draws videos from publicly released datasets — many thanks to:
**[Ego4D](https://ego4d-data.org/)**, **[RoomTour3D](https://roomtour3d.github.io/)**, **[CODa](https://amrl.cs.utexas.edu/coda/)**, **[OmniWorld](https://omniworld.github.io/)**, **[VSI-Bench](https://github.com/vision-x-nyu/thinking-in-space)**, **[Sekai](https://huggingface.co/datasets/SekaiTrip/SekaiBench)**, **[ARKitScenes](https://github.com/apple/ARKitScenes)**, **[Honda HDD](https://usa.honda-ri.com/HDD)**, and selected **YouTube** walking tours.

The evaluation framework integrates wrappers for several token-compression and streaming MLLM research repos:
**[HERMES](https://github.com/microsoft/HERMES)**, **[FluxMem](https://github.com/FluxMem)**, **[StreamingTOM](https://github.com/StreamingTOM)**, **[InfiniPot-V](https://github.com/InfiniPot-V)**, **[InfiniteVL](https://github.com/InfiniteVL)**, **[Flash-VStream](https://github.com/IVGSZ/Flash-VStream)**, **[StreamForest](https://github.com/StreamForest)**, **[Spatial-MLLM](https://github.com/diankun-wu/Spatial-MLLM)**, **[Cambrian-S](https://github.com/cambrian-mllm/cambrian)**.

---

## License

MIT — see [LICENSE](LICENSE). Annotations released under CC-BY-4.0; source videos retain their original licenses.

---

## Star History

<a href="https://www.star-history.com/?repos=InternLM/OVO-S-Bench&type=date&legend=top-left">
  <img src="https://api.star-history.com/svg?repos=InternLM/OVO-S-Bench&type=Date&legend=top-left" alt="Star History" width="600"/>
</a>
