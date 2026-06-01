<p align="center">
  <img src="https://raw.githubusercontent.com/InternLM/OVO-S-Bench/webpage/static/images/teaser.png" alt="OVO-S-Bench teaser" width="92%"/>
</p>

<h1 align="center">OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs</h1>

<p align="center">
  <a href="https://github.com/yifei-liyifei">Yifei Li</a><sup>1,2,†</sup> &nbsp;·&nbsp;
  <a href="#">Pengyiang Liu</a><sup>3,†</sup> &nbsp;·&nbsp;
  <a href="https://yuhangzang.github.io/">Yuhang Zang</a><sup>2,*</sup> &nbsp;·&nbsp;
  Zhongyue Shi<sup>3</sup> &nbsp;·&nbsp;
  Qi Fu<sup>3</sup> &nbsp;·&nbsp;
  Hongye Hao<sup>3</sup> &nbsp;·&nbsp;
  <a href="http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/">Jiwen Lu</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Tsinghua University &nbsp;&nbsp; <sup>2</sup>Shanghai AI Lab &nbsp;&nbsp; <sup>3</sup>Beihang University<br/>
  <sup>†</sup>Equal Contribution &nbsp;&nbsp; <sup>*</sup>Project Leader
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Questions-1680-blue" alt="Questions"/>
  <img src="https://img.shields.io/badge/Videos-348-purple" alt="Videos"/>
  <img src="https://img.shields.io/badge/Levels-4-teal" alt="Levels"/>
  <img src="https://img.shields.io/badge/Task_Types-30-green" alt="Task Types"/>
  <img src="https://img.shields.io/badge/Datasets-9-yellow" alt="Datasets"/>
  <img src="https://img.shields.io/badge/Models-38-orange" alt="Models"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://huggingface.co/datasets/InternLM/OVO-S-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Dataset-yellow" alt="HF Dataset"/></a>
  <a href="https://internlm.github.io/OVO-S-Bench/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Project-Page-blue" alt="Project Page"/></a>
  <a href="https://internlm.github.io/OVO-S-Bench/"><img src="https://img.shields.io/badge/%F0%9F%8F%86_Leaderboard-OVO--S--Bench-8c2416" alt="Leaderboard"/></a>
</p>

---

## Abstract

Multimodal agents in robotics, AR, and autonomous driving must reason about places and layouts from continuous egocentric streams, often using evidence outside the current view. Existing benchmarks either evaluate offline over full videos or target events rather than spatial structure. We introduce **OVO-S-Bench**, a fully human-annotated benchmark for streaming spatial intelligence, comprising **1,680 questions over 348 source videos**. Annotation involves **12 trained annotators** (each also serving as a blind cross-reviewer) across roughly **804 person-hours** of multi-round quality assurance. Each question carries a query timestamp and an evidence interval, and at evaluation the model sees only the prefix preceding the query. Questions span four levels of increasing abstraction: **instantaneous egocentric perception**, **spatiotemporal context tracking**, **spatial simulation and reasoning**, and **allocentric mapping**. Across **38 proprietary and open-source MLLMs**, Gemini-3.1-Pro trails human experts by **27 points (59.2 vs. 86.6)**, with allocentric mapping as the dominant bottleneck. Notably, streaming and spatially fine-tuned MLLMs underperform their own backbones. We further find that chain-of-thought reasoning amplifies spatial errors when ungrounded in the stream. By exposing these limitations, OVO-S-Bench establishes a demanding testbed for next-generation streaming spatial MLLMs.

## Four-Level Streaming Spatial Taxonomy

OVO-S-Bench organizes its 1,680 questions into **four cumulative levels** by the spatial state a model must access at query time, progressing from evidence in the current view to global-map queries that require cross-viewpoint integration.

| Level | Capability | Task families |
| --- | --- | --- |
| **L1 — Instantaneous Egocentric Perception** | Answerable from frames near the query timestamp alone, without recalling any past observation | egocentric metric perception (distance, scale, clearance, viewpoint height) · local spatial relations (containment, occlusion, support, visible layout) · dynamic spatial perception (camera motion, object motion, relative speed) |
| **L2 — Spatiotemporal Context Tracking** | Evidence has appeared in the prefix but is no longer visible at query time | scene revisit recognition · spatial memory beyond the view · chronological spatial memory |
| **L3 — Spatial Simulation and Reasoning** | Operate on spatial structure rather than merely retrieve an observation | spatial simulation (reorientation, removal consequences, physical feasibility) · spatiotemporal consistency verification · spatial route planning |
| **L4 — Allocentric Spatial Mapping** | Integrate the egocentric stream into an allocentric representation and query its global structure | allocentric direction reasoning · topological structure reasoning · **trajectory-map alignment** (image options) |

The released benchmark comprises **1,680 questions over 348 source videos from 9 datasets**, organized into **30 canonical task types** across four levels. Mean prefix at query time: **8.8 min**. Evidence-span medians by level: **L1 2.0 s · L2 36.8 s · L3 2.0 s · L4 278.7 s** — reflecting the spatial persistence each level demands.

> L4.3 trajectory-matching questions ship their option images **embedded inline** as base64 data URIs in the parquet's `options` column. No separate image-asset download is needed.

## Why OVO-S-Bench?

Existing benchmarks leave the streaming-spatial regime untested. Spatial benchmarks study 3D relations, multi-view reasoning, and embodied QA but assume static or offline visual context; long-video and streaming benchmarks target event understanding, narrative memory, or response timing rather than spatial structure in a continuous visual stream.

| Capability | What we test | Why it's hard |
| --- | --- | --- |
| **🎯 Streaming protocol** | Each query carries a `query_time` $t_q$; only the prefix $[0, t_q]$ is visible — 128 uniformly-sampled frames | Models cannot peek at future evidence; many existing video MLLMs implicitly assume the whole clip |
| **🧱 Hierarchical** | 4 cumulative levels × 30 canonical task types | Each level requires a capability the previous lacks; per-level ablations isolate where reasoning breaks down |
| **🗺️ Allocentric reasoning** | L4 integrates the egocentric stream into a global map (cardinal directions, room topology, BEV trajectory matching) | Dominates the gap to human performance; 28 of 34 systems score lowest on L4 |
| **🌍 Source diversity** | 9 datasets across 5 regimes: indoor walkthroughs (RoomTour3D), egocentric activities (Ego4D), outdoor/world (Sekai, OmniWorld, YouTube), driving (CODa, Honda HDD), 3D-annotated (ARKitScenes, VSI-Bench) | Spatial reasoning must hold across drastically different visual statistics |
| **👥 Human-written** | 12 annotators · 804 person-hours · blind cross-review + text-only LLM-probe shortcut filtering | No template-generated items; distractors are visually plausible but evidentially wrong |

## Key Findings

| Finding | Numbers |
| --- | --- |
| **Significant gap with human performance.** Strongest system **Gemini-3.1-Pro** reaches **59.2** overall, far below human experts under the same streaming protocol (**86.6**; **92.2** offline). Best open-source: **Qwen3-VL-235B-A22B** at **53.6**, trailing human-streaming by **33 points**. Random (31.3) and Text-Only (37.1) baselines fall below all general-backbone systems. | 27 pts gap |
| **Allocentric mapping is the dominant bottleneck.** L4 is the lowest-scoring level for **28 of 34** systems, with a mean gap of **9.3 %** between L1–L3 and L4. Even the largest open-source backbones drop ≥10 pts (Qwen3-VL-235B-A22B: 10.6; InternVL-3.5-241B-A28B: 13.8). | 28 / 34 |
| **Narrow but uneven closed-source lead.** Only **+5.6 pts overall** (Gemini-3.1-Pro vs Qwen3-VL-235B-A22B), narrower than the 10+ pt gap on recent video benchmarks. Gap widens on memory-heavy L2 (**+5.9**), narrows on L4 (**+4.1**); on L3 the best open-source *exceeds* Gemini-3.1-Pro by **+5.3** (61.2 vs 55.9). | +5.6 / +5.3 |
| **Specialization hurts the backbone.** No streaming-architecture or spatially fine-tuned variant outperforms its comparable general backbone; **13 of 15** lag behind their own base (median −2.0, range −18.4 to +0.5). Only HERMES (+0.4) and FluxMem (+0.5) exceed base. **L4 is the most uniformly damaged level**: 13/15 regress (mean Δ = **−6.1**; Flash-VStream-7B **−16.7**, Cosmos-Reason1-7B **−12.8**). | 13 / 15 |
| **Chain-of-thought is double-edged.** Across paired thinking-mode comparisons, explicit reasoning consistently helps L2 (mean Δ = **+3.9**, 8/9 pairs positive) but slightly hurts L1 (mean Δ = **−1.0**, 6/9 pairs negative). A GPT-5.4 judge over wrong traces finds **60–80 % of CoT failures are mis-grounded visual evidence**. | +3.9 / −1.0 |
| **Retention is not the bottleneck.** For HERMES, StreamingTOM, FluxMem, mean Evidence Recall on L4 ranges **0.14 → 0.42**, much lower than L1 (0.60–0.76). Yet per-query Pearson between ER and correctness is essentially zero (**r ∈ [−0.07, 0.00]**) — compressors don't lose only useful evidence yet still don't help. | r ≈ 0 |

## News

- **2026-06** Initial release: evaluation framework (this repo) + HuggingFace dataset.
- **2026-06** Project page online: <https://internlm.github.io/OVO-S-Bench/>

---

## Leaderboard

Full interactive leaderboard with per-level breakdowns: <https://internlm.github.io/OVO-S-Bench/>

**Top systems (overall, streaming protocol):**

| Rank | System | Org | Overall | L3 best (open-source vs closed) |
| :-: | --- | --- | :-: | :-: |
| 🥇 | **Human Expert (offline)** | — | **92.2** | — |
| 🥈 | Human Expert (streaming) | — | **86.6** | — |
| 🥉 | **Gemini-3.1-Pro** | Google DeepMind | **59.2** | 55.9 |
| 4 | Qwen3-VL-235B-A22B | Alibaba Cloud | 53.6 | **61.2** |
| 5 | GPT-5.4 | OpenAI | 50.9 | — |
| 6 | Grok-4.1-Fast | xAI | 43.7 | — |
| · | Text-Only (GPT-5.4) | — | 37.1 | — |
| · | Random | — | 31.3 | — |

> Full table covering all 38 systems with per-level scores: see the project page.

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

The annotations parquet + source videos are hosted on HuggingFace Datasets:

```bash
pip install -U "huggingface_hub[cli]"
hf download InternLM/OVO-S-Bench --repo-type dataset --local-dir ./data
```

Layout you should see after download:

```
data/
├── ovo_s_bench_l1_l4.parquet     # questions, ~35 MB
└── videos/                       # source .mp4 files
    ├── Ego4D/ ...
    ├── RoomTour3D/ ...
    ├── annotated_videos/ ...
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

**Open-source MLLM via vLLM** (multi-GPU auto-sharded; default 128 uniformly-sampled frames per prefix):

```bash
GPUS=8 bash scripts/eval_vllm.sh qwen3-vl-32b data/ovo_s_bench_l1_l4.parquet
```

Both scripts call `inference.py` then `score.py`. Inference auto-resumes on rerun — interrupted jobs pick up from the last checkpoint.

### Add a new model

See [docs/adding_models.md](docs/adding_models.md). The contract is `BaseModel.inference(frames: List[PIL.Image], prompt: str) -> str`; add a config entry to `config.yaml::MODELS` and register the wrapper in `models/api_models.py` or `models/vllm_models.py`.

### Custom API endpoint

Set `*_BASE_URL` in `.env` to point at your proxy or local vLLM server. API providers (`openai`, `gemini-native`, `anthropic`) honor the standard `*_BASE_URL` env var convention.

---

## Evaluation Protocol

All systems are evaluated under a **unified streaming protocol**: each source video is truncated at the annotated query timestamp $t_q$, and the model receives **128 frames uniformly sampled from the resulting prefix** together with the question and multiple-choice options. For streaming-architecture models that implement a native sequential ingestion path, we instead feed the video at each model's published streaming rate and query the resulting compressed state. No model sees frames after $t_q$. Answers are extracted by regular expression without further post-processing.

Per-query responses land under `results/<model>/ovo_s_bench_l1_l4.json`. `score.py` aggregates accuracy by main category and subcategory:

```bash
python score.py --result results/gpt-4o/ovo_s_bench_l1_l4.json --verbose
```

For sharded multi-GPU runs, `merge_results.py` consolidates `*_rank{N}.json` shards into a single result file (auto-invoked by `launch.py`).

---

## Repository Layout

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
@article{li2026ovos,
  title  = {OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs},
  author = {Li, Yifei and Liu, Pengyiang and Zang, Yuhang and Shi, Zhongyue and Fu, Qi and Hao, Hongye and Lu, Jiwen},
  year   = {2026},
  note   = {arXiv:TBD}
}
```

A machine-readable `CITATION.cff` will land alongside the arXiv ID.

---

## Acknowledgements

OVO-S-Bench draws videos from publicly released datasets — many thanks to:
**[Ego4D](https://ego4d-data.org/)**, **[RoomTour3D](https://roomtour3d.github.io/)**, **[CODa](https://amrl.cs.utexas.edu/coda/)**, **[OmniWorld](https://omniworld.github.io/)**, **[VSI-Bench](https://github.com/vision-x-nyu/thinking-in-space)**, **[Sekai](https://huggingface.co/datasets/SekaiTrip/SekaiBench)**, **[ARKitScenes](https://github.com/apple/ARKitScenes)**, **[Honda HDD](https://usa.honda-ri.com/HDD)**, and selected **YouTube** walking tours.

The evaluation framework integrates wrappers for several token-compression and streaming MLLM research repos:
**[HERMES](https://github.com/microsoft/HERMES)**, **[FluxMem](https://github.com/FluxMem)**, **[StreamingTOM](https://github.com/StreamingTOM)**, **[InfiniPot-V](https://github.com/InfiniPot-V)**, **[InfiniteVL](https://github.com/InfiniteVL)**, **[Flash-VStream](https://github.com/IVGSZ/Flash-VStream)**, **[StreamForest](https://github.com/StreamForest)**, **[Spatial-MLLM](https://github.com/diankun-wu/Spatial-MLLM)**, **[Cambrian-S](https://github.com/cambrian-mllm/cambrian)**, **[SenseNova-SI](https://github.com/SenseTime/sensenova-si)**, **[Spatial-TTT](https://github.com/spatial-ttt)**.

---

## License

MIT — see [LICENSE](LICENSE). Annotations released under CC-BY-4.0; source videos retain their original licenses.

---

## Star History

<a href="https://www.star-history.com/?repos=InternLM/OVO-S-Bench&type=date&legend=top-left">
  <img src="https://api.star-history.com/svg?repos=InternLM/OVO-S-Bench&type=Date&legend=top-left" alt="Star History" width="600"/>
</a>
