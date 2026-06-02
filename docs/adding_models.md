# Adding a new model

OVO-S-Bench models are pluggable wrappers around a uniform interface:

```python
from models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        # parse config fields, initialize state

    def inference(self, frames: List[PIL.Image], prompt: str) -> str:
        """Run inference and return the raw model response."""
        ...
```

`frames` is a list of PIL Images extracted by the framework according to the
model config's `sampling_strategy` and `max_frames`. `prompt` is the
multiple-choice prompt built by `prompts.py`.

## Where to put the file

| Tier  | Location                | When                                                                   |
| ----- | ----------------------- | ---------------------------------------------------------------------- |
| Core  | `models/<name>_models.py` | Wrapper only needs `pip install` packages (vLLM, transformers, openai) |
| Extra | `models/extras/<name>_models.py` | Wrapper needs an upstream research repo to be cloned alongside     |

Extras wrappers can use `from ._paths import find_upstream_src` to resolve the
upstream source location (see `models/extras/README.md`).

## Register the model

### For API-based wrappers

Edit `models/api_models.py` and add an entry to `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = {
    ...
    "my-provider": MyModel,
}
```

### For offline / vLLM wrappers

Edit `models/vllm_models.py` and add a `try/except` block inside
`_get_offline_registry()`:

```python
try:
    from .my_models import MyModel              # or .extras.my_models
    registry["my-provider"] = MyModel
except ImportError as e:
    print(f"Warning: MyModel provider unavailable: {e}")
```

The `try/except` keeps the registry resilient: missing optional dependencies
print a warning but don't break the other models.

## Add a config entry

In `config.yaml::MODELS`, pick a category and add your model. The minimum
config is:

```yaml
MODELS:
  open-source-general-mllm:
    my-model:
      type: offline
      provider: my-provider          # matches the registry key above
      model_id: org/MyModel-Hub-Id   # or HuggingFace repo
      max_frames: 128
      frame_size: 512
      tensor_parallel_size: 1
      batch_size: 1
```

A nested `defaults` + `variants:` block (see e.g. `qwen3-vl`) lets you define
multiple sizes that share a base config.

## Test it

```bash
python inference.py --model my-model --annotation data/ovo_s_bench.parquet --limit 5
python score.py --result results/my-model/ovo_s_bench.json
```

The `--limit 5` flag is the fastest way to confirm the wrapper works
end-to-end before running on the full 1695-question set.
