"""Base model wrapper for OVO-S-Bench evaluation.

All model implementations inherit from `BaseModel` and implement
`inference(frames, prompt) -> str`.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def resolve_runtime_path(path: Optional[str]) -> Optional[str]:
    """Resolve a model checkpoint path with two-step fallback:

    1. Return `path` verbatim if it exists.
    2. If it's relative, look it up under `$OVO_S_MODEL_ROOT` (if set).
    3. Otherwise return `path` as-is and let the caller surface the error.

    Set `OVO_S_MODEL_ROOT` (in `.env`) to point at the directory containing
    your local HF checkpoints. Most config entries use HF repo ids
    (e.g. `Qwen/Qwen3-VL-32B`) which need no resolution and download on first
    use.
    """
    if not path:
        return path
    if Path(path).exists():
        return path
    model_root = os.environ.get("OVO_S_MODEL_ROOT")
    if model_root:
        candidate = Path(model_root) / path
        if candidate.exists():
            return str(candidate)
    return path


class BaseModel(ABC):
    """Abstract base class for all model wrappers."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model_id = config.get("model_id", model_name)
        self.max_frames = config.get("max_frames", 8)
        self.fps = config.get("fps", 1)
        self.frame_size = config.get("frame_size", 512)
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.0)

    @abstractmethod
    def inference(self, frames: List[Any], prompt: str) -> str:
        """Run inference on `frames` with `prompt`. Returns the raw model output."""

    def build_prompt(
        self,
        question: str,
        options: Dict[str, str],
        prompt_style: Optional[str] = None,
    ) -> str:
        """Format a multiple-choice prompt using the registered template."""
        prompt, _ = self.prepare_prompt(question, options, prompt_style=prompt_style)
        return prompt

    def prepare_prompt(
        self,
        question: str,
        options: Dict[str, Any],
        prompt_style: Optional[str] = None,
    ) -> Tuple[str, List[Tuple[str, Any]]]:
        """Build the prompt text and decode any base64 image options.

        Returns `(prompt_str, option_images)` where `option_images` is a list of
        `(label, PIL.Image)` tuples in option-label order, empty when none of the
        options is an image. Used by L4.3 trajectory-map questions, where the
        options are PNGs embedded as `data:image/png;base64,...`.
        """
        style = prompt_style or self.config.get("prompt_style", "default")
        from prompts import build_prompt
        from option_utils import (
            build_image_options_instruction,
            has_image_options,
            split_options,
        )

        if isinstance(options, dict) and has_image_options(options):
            text_options, option_images = split_options(options, frame_size=self.frame_size)
            prompt = build_prompt(question, text_options, style)
            prompt = f"{prompt}\n\n{build_image_options_instruction(option_images)}"
            return prompt, option_images

        return build_prompt(question, options, style), []
