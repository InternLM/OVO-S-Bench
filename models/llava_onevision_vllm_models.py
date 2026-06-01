"""
vLLM-based wrapper for LLaVA-OneVision (HF-converted format).

vLLM 0.17+ has native support for `LlavaOnevisionForConditionalGeneration`,
which is the HF-converted form of `lmms-lab/llava-onevision-qwen2-7b-ov`.
This wrapper feeds video frames as multi-image input via the HF chat template
and uses vLLM's batched generation path — much simpler than the lmms-lab
LLaVA-NeXT codebase and works in the standard vllm-qwen3vl conda env.

Used as §4.3.4(a) base for StreamingTOM (and as a non-Qwen-family baseline
to broaden the architectural diversity beyond Qwen2.5-VL / Qwen3-VL).
"""

import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .base import BaseModel, resolve_runtime_path

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_MODULES_CACHE"] = os.path.expanduser("~/.cache/huggingface/modules")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class VLLMLlavaOnevisionModel(BaseModel):
    """vLLM offline inference for LLaVA-OneVision (HF format)."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(
            config.get("local_path", config.get("model_id"))
        )
        self.tensor_parallel_size = config.get(
            "tensor_parallel_size", torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.85)
        self.max_images_per_prompt = config.get("max_images_per_prompt", 32)

        self.llm = None
        self.processor = None
        self.sampling_params = None

    def _init_model(self):
        if self.llm is not None:
            return
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        print(f"Loading LLaVA-OneVision (vllm) from: {self.local_path}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")

        self.processor = AutoProcessor.from_pretrained(self.local_path)

        self.llm = LLM(
            model=self.local_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": self.max_images_per_prompt,
                                 "video": 1},
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _prepare_input(self, frames: List[Image.Image], prompt: str) -> dict:
        """Build a multi-image LLaVA-OneVision chat-template prompt + frames.

        LLaVA-OneVision pools each video frame to ~196 tokens (vs. ~1380 tokens
        per anyres-9 image), so we always feed frames through the
        ``{"type": "video"}`` modality — otherwise 32 frames would blow the 32k
        context window.
        """
        # Truncate frame list to fit max_images_per_prompt budget.
        if len(frames) > self.max_images_per_prompt:
            import numpy as np
            idx = np.linspace(0, len(frames) - 1, self.max_images_per_prompt, dtype=int)
            frames = [frames[i] for i in idx]

        if not frames:
            content = [{"type": "text", "text": prompt}]
            mm_data = {}
        else:
            # Video modality: single placeholder, frame list goes into mm_data.
            content = [{"type": "video"}, {"type": "text", "text": prompt}]
            mm_data = {"video": frames}

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": text, "multi_modal_data": mm_data}

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_images_per_prompt
            )
        self._init_model()
        vllm_input = self._prepare_input(frames, prompt)
        outputs = self.llm.generate([vllm_input], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def batch_inference(self, batch_frames: List[List[Image.Image]],
                        batch_prompts: List[str],
                        batch_option_images: Optional[List[Optional[List[Any]]]] = None) -> List[str]:
        self._init_model()
        if batch_option_images is None:
            batch_option_images = [None] * len(batch_prompts)
        from option_utils import append_option_images_to_frames
        merged = [
            append_option_images_to_frames(f, oi, max_n=self.max_images_per_prompt) if oi else f
            for f, oi in zip(batch_frames, batch_option_images)
        ]
        vllm_inputs = [self._prepare_input(f, p) for f, p in zip(merged, batch_prompts)]
        outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]
