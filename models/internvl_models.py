"""
vLLM offline batch inference for InternVL 3.5 models.

InternVL3.5 uses InternVLProcessor for chat template and image handling.
With vLLM, images are passed as PIL images in multi_modal_data and
the prompt uses <image> placeholders.

Requirements:
    pip install vllm>=0.11.0 transformers
"""

import os
import torch
from typing import Any, Dict, List, Optional
from PIL import Image

from .base import BaseModel, resolve_runtime_path

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_MODULES_CACHE"] = os.path.expanduser("~/.cache/huggingface/modules")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# Official InternVL3.5 R1-style system prompt for Thinking mode. Source:
# https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/internvl/internvl_chat.py
# Used verbatim per the model card recommendation; together with
# temperature=0.6 + do_sample=True this enables <think>...</think> traces.
R1_SYSTEM_PROMPT = (
    "You are an AI assistant that rigorously follows this response protocol:\n\n"
    "1. First, conduct a detailed analysis of the question. Consider different angles, "
    "potential solutions, and reason through the problem step-by-step. Enclose this "
    "entire thinking process within <think> and </think> tags.\n\n"
    "2. After the thinking section, provide a clear, concise, and direct answer to the "
    "user's question. Separate the answer from the think section with a newline.\n\n"
    "Ensure that the thinking process is thorough but remains focused on the query. "
    "The final answer should be standalone and not reference the thinking section."
)


class VLLMInternVLModel(BaseModel):
    """vLLM offline inference for InternVL 3.5 models."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(
            config.get("local_path", config.get("model_id"))
        )
        self.tensor_parallel_size = config.get(
            "tensor_parallel_size", torch.cuda.device_count()
        )
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.85)
        self.max_images_per_prompt = config.get("max_images_per_prompt", 64)
        # Limit dynamic patches per image to fit many frames in context.
        # InternVL3.5 default max_dynamic_patch=12 → ~3072 tokens/image.
        # With max_dynamic_patch=1 → 256 tokens/image, 128 frames = 32768 < 40960.
        self.max_dynamic_patch = config.get("max_dynamic_patch", 1)

        # vLLM context budget. None = vLLM default (~40k for InternVL3.5),
        # which silently rejects requests > limit (e.g. u256: 256 frames ×
        # 256 tok/img ≈ 65k > 40k). Override via config when running u256+.
        self.max_model_len = config.get("max_model_len", None)

        # Vision input-size override. InternVL3.5 default = 448 → 256 visual
        # tokens per tile (when max_dynamic_patch=1). InternVL-3.5-38B has a
        # hard architectural ceiling of max_position_embeddings=40960 and no
        # RoPE scaling, so u256 (256 frames × 256 tokens = 65536) is
        # un-runnable at default resolution. Setting force_image_size=224
        # cuts per-tile tokens to 64 (16×16 patches × 0.25 downsample), so
        # 256 frames = 16384 tokens, fitting under 40960 with room for the
        # prompt. Trade-off: lower visual resolution per frame.
        self.force_image_size = config.get("force_image_size", None)

        # Thinking mode (R1-style system prompt + non-zero temperature). Per the
        # InternVL3.5 model card, T=0.6 + do_sample mitigate repetition during
        # the reasoning trace. We keep the same protocol's frame sampling.
        self.enable_thinking = bool(config.get("enable_thinking", False))
        self.top_p = config.get("top_p", 0.95)

        self.llm = None
        self.processor = None
        self.sampling_params = None

    def _init_model(self):
        """Lazy initialization of vLLM model."""
        if self.llm is not None:
            return

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"Loading InternVL model from: {self.local_path}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")
        if self.enable_thinking:
            print(
                f"InternVL Thinking mode: ENABLED "
                f"(temperature={self.temperature}, top_p={self.top_p}, "
                f"max_tokens={self.max_tokens})"
            )

        # Use AutoTokenizer instead of AutoProcessor to avoid
        # InternVLProcessor.__init__ crash on Qwen2Tokenizer missing
        # start_image_token attribute (transformers >=5.x bug).
        self.processor = AutoTokenizer.from_pretrained(
            self.local_path, trust_remote_code=True
        )

        # Build hf_overrides to inject force_image_size into both top-level
        # config and vision_config (the InternVL preprocessor reads the latter).
        hf_overrides = {}
        if self.force_image_size:
            hf_overrides["force_image_size"] = self.force_image_size
            hf_overrides["vision_config"] = {"image_size": self.force_image_size}

        self.llm = LLM(
            model=self.local_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": self.max_images_per_prompt},
            mm_processor_kwargs={"max_dynamic_patch": self.max_dynamic_patch},
            **({"max_model_len": self.max_model_len} if self.max_model_len else {}),
            **({"hf_overrides": hf_overrides} if hf_overrides else {}),
        )

        sp_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.enable_thinking:
            # Non-zero temperature already implies sampling in vLLM; top_p
            # restricts the nucleus to stabilize the long thinking trace.
            sp_kwargs["top_p"] = self.top_p
        self.sampling_params = SamplingParams(**sp_kwargs)

    def _prepare_input(self, frames: List[Image.Image], prompt: str) -> dict:
        """Prepare input for vLLM inference.

        InternVL's chat template expects structured content (list of dicts)
        with {"type": "image"} entries so the jinja template inserts <image>
        placeholders automatically. Using plain string with <image> triggers
        tokenizer.start_image_token which Qwen2Tokenizer doesn't have.
        """
        # Build structured content: one {"type": "image"} per frame + text
        content = []
        for _ in frames:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        messages = []
        if self.enable_thinking:
            messages.append({"role": "system", "content": R1_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        mm_data = {}
        if frames:
            mm_data["image"] = frames

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
        }

    def inference(
        self,
        frames: List[Image.Image],
        prompt: str,
        option_images: Optional[List[Any]] = None,
    ) -> str:
        """Run single inference."""
        # Image-option path (task 4.3.x): InternVL builds one <image> placeholder
        # per frame, so option PIL images appended at the end naturally become
        # extra <image> placeholders without any other change.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_images_per_prompt
            )
        self._init_model()
        vllm_input = self._prepare_input(frames, prompt)
        outputs = self.llm.generate([vllm_input], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def batch_inference(
        self,
        batch_frames: List[List[Image.Image]],
        batch_prompts: List[str],
        batch_option_images: Optional[List[Optional[List[Any]]]] = None,
    ) -> List[str]:
        """Run batch inference for better throughput."""
        self._init_model()
        if batch_option_images is None:
            batch_option_images = [None] * len(batch_prompts)
        from option_utils import append_option_images_to_frames
        merged_frames = [
            append_option_images_to_frames(f, oi, max_n=self.max_images_per_prompt) if oi else f
            for f, oi in zip(batch_frames, batch_option_images)
        ]
        vllm_inputs = [
            self._prepare_input(frames, prompt)
            for frames, prompt in zip(merged_frames, batch_prompts)
        ]
        outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]
