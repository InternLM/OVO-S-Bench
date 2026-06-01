"""
StreamingVLM offline inference for OVO-S evaluation.

The upstream StreamingVLM checkpoint is a Qwen2.5-VL-7B variant fine-tuned with
streaming-style training (sliding KV cache, position-shrink). Two upstream
inference paths exist:

1. ``streaming_vlm/inference/inference.py`` — chunked, KV-cached commentary loop
   (uses ``convert_qwen2_5_to_streaming`` + ``StreamingArgs``).
2. ``streaming_vlm/eval/ovobench/distributed_evaluate_ovobench.py`` — vanilla
   ``Qwen2_5_VLForConditionalGeneration`` over a fixed-frame video clip.

For OVO-S we follow path (2) — short MCQ video clips, single forward pass — which
is exactly how the upstream paper evaluates on OVO-Bench. This avoids the
``flash_attn_varlen_func`` import drift introduced in newer transformers releases
and keeps us aligned with the authors' own benchmark pipeline.

Frames are pre-extracted by ``frame_utils`` and passed to the model as a video
input via ``qwen_vl_utils.process_vision_info``.

Requires:
    - conda env: streamingvlm-infer
    - transformers >= 4.51, qwen_vl_utils, flash-attn (recommended)
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from ..base import BaseModel, resolve_runtime_path


def _resolve_cluster_path(path: str | None) -> str | None:
    """Backwards-compatible alias around the package-wide path resolver."""
    return resolve_runtime_path(path)


class StreamingVLMModel(BaseModel):
    """Vanilla Qwen2.5-VL-style inference on the StreamingVLM checkpoint.

    Aligned with the upstream OVO-Bench evaluation script
    (``streaming_vlm/eval/ovobench/distributed_evaluate_ovobench.py``).
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = _resolve_cluster_path(config.get("local_path", config.get("model_id")))
        self.lora_path = _resolve_cluster_path(config.get("lora_path", None))
        self.model_base = config.get("model_base", "Qwen2_5")
        self.attn_implementation = config.get("attn_implementation", "flash_attention_2")
        self.torch_dtype = config.get("torch_dtype", "auto")
        self.device = config.get("device", "cuda")

        self._model = None
        self._processor = None

    def _init_model(self):
        if self._model is not None:
            return

        from transformers import (
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
        )

        ModelCls = (
            Qwen2_5_VLForConditionalGeneration
            if self.model_base == "Qwen2_5"
            else Qwen2VLForConditionalGeneration
        )
        attn = self.attn_implementation if torch.cuda.is_available() else "eager"

        print(f"Loading StreamingVLM ({self.model_base}) from: {self.local_path}")
        print(f"  attn_implementation={attn}")

        self._model = ModelCls.from_pretrained(
            self.local_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            attn_implementation=attn,
        )

        if self.lora_path:
            from peft import PeftModel

            print(f"Loading LoRA from: {self.lora_path}")
            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
            self._model = self._model.merge_and_unload()

        self._processor = AutoProcessor.from_pretrained(self.local_path, use_fast=False)
        self._model.eval()
        print("StreamingVLM loaded.")

    @staticmethod
    def _to_pil(frame) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame
        if isinstance(frame, np.ndarray):
            return Image.fromarray(frame)
        if isinstance(frame, torch.Tensor):
            arr = frame.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            return Image.fromarray(arr.astype(np.uint8))
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def inference(self, frames: List[Any], prompt: str,
                  option_images: List[Any] = None) -> str:
        self._init_model()

        # Image-option path (task 4.3.x): append option PIL images at the tail
        # of the frame list so they ride along inside the single "video" block.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )

        from qwen_vl_utils import process_vision_info

        pil_frames = [self._to_pil(f) for f in frames]
        if not pil_frames:
            return ""

        # Pass frames as a video input — matches upstream OVO-Bench eval shape.
        content = [
            {"type": "video", "video": pil_frames},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return response
