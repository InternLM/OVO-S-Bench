# `models/extras/` — model wrappers requiring upstream repos

The wrappers in this directory cannot be installed via `pip` alone — they
depend on research repos that bring their own model definitions, CUDA kernels,
and (often) conflicting `transformers` pins. We expose them here for
reproducibility but require you to clone the upstream sources yourself.

## How upstream sources are located

For an upstream repo named `<NAME>`, each wrapper looks for the source at, in
order:

1. `$OVO_S_<NAME_UPPER>_SRC` — per-upstream env var
2. `$OVO_S_EXTRAS_SRC/<NAME>` — umbrella env var
3. `<repo_root>/extras_src/<NAME>` — default convention

So the simplest setup is:

```bash
mkdir extras_src
git clone <upstream-url> extras_src/<NAME>
```

## Installation matrix

| Wrapper                  | Upstream                                                 | Conda env hint        |
| ------------------------ | -------------------------------------------------------- | --------------------- |
| `hermes_models.py`       | HERMES (KV pruning on Qwen2.5-VL)                        | flash-attn, transformers |
| `fluxmem_models.py`      | FluxMem (3-tier visual token memory on Qwen2.5-VL)       | flash-attn, decord    |
| `streamingtom_models.py` | StreamingTOM (CTR+OQM on LLaVA-OneVision; needs `LLaVA-NeXT` subdir) | flash-attn |
| `infinipot_models.py`    | InfiniPot-V (block-wise KV compression)                  | flash-attn, decord    |
| `infinitevl_models.py`   | InfiniteVL (hybrid linear attention streaming)           | flash-linear-attention |
| `flash_vstream_models.py`| Flash-VStream (Qwen2-VL based)                           | flash-attn            |
| `streamforest_models.py` | StreamForest (LLaVA + event-memory forest)               | flash-attn            |
| `streaming_vlm_models.py`| StreamingVLM checkpoint on Qwen2.5-VL                    | flash-attn            |
| `spatial_mllm_models.py` | Spatial-MLLM                                             | flash-attn            |
| `spatial_ttt_models.py`  | Spatial-TTT (test-time training)                         | flash-attn            |
| `cambrian_models.py`     | Cambrian-S                                               | flash-attn            |
| `llava_next_video_models.py` | LLaVA-NeXT-Video                                     | standard              |

## Each wrapper at a glance

Look at the top of each `.py` file for a one-line installation hint
(`Requires: conda env: <name>`) plus the protocol assumptions.

When a wrapper fails to import (because the upstream isn't on disk), the
benchmark's offline model registry (`models/vllm_models.py:_get_offline_registry`)
prints a warning and continues — your other models stay usable.

## Adding a new extras wrapper

1. Create `models/extras/myname_models.py` subclassing `BaseModel`.
2. If you need to push an upstream repo onto `sys.path`, use
   `from ._paths import find_upstream_src` to look it up.
3. Register the new class in `models/vllm_models.py:_get_offline_registry()`
   under a `try / except ImportError` block.
4. Add a config entry under `config.yaml::MODELS`.
