"""
LLaVA-NeXT-Video model wrapper for OVO-S evaluation.

Uses the original lmms-lab/LLaVA-NeXT codebase (LlavaLlamaForCausalLM)
with vicuna-v1 conversation template. Video frames are passed as a single
video item with one <image> token and modalities=["video"], which triggers
the spatial_pool resampler (avg_pool2d stride=2) to reduce per-frame
tokens from 576 to 144.

Requires:
    - LLaVA-NeXT source code (from StreamingTOM/LLaVA-NeXT)
    - conda env with transformers, torch, flash-attn
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image

from ..base import BaseModel

# Path to LLaVA-NeXT source (reuse from StreamingTOM)
_LLAVA_NEXT_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "_src",
    "token-compression-methods", "StreamingTOM", "LLaVA-NeXT"
))


class LLaVANextVideoModel(BaseModel):
    """LLaVA-NeXT-Video inference using original LLaVA codebase."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = config.get("local_path", config.get("model_id"))
        self.conv_template = config.get("conv_template", "v1")
        self.clip_local_path = config.get("clip_local_path", None)

        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._context_len = None

    def _init_model(self):
        """Lazy initialization.

        Loads the model directly via LlavaLlamaForCausalLM instead of the
        builder, which has a bare ``except:`` that swallows real errors and
        reports "Model not supported" when the model name doesn't match its
        hardcoded keyword list.
        """
        if self._model is not None:
            return

        # Prevent HF downloads on cluster nodes without internet access.
        # All models must be available locally.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Add LLaVA-NeXT source to path
        if _LLAVA_NEXT_SRC not in sys.path:
            sys.path.insert(0, _LLAVA_NEXT_SRC)

        from transformers import AutoTokenizer
        from llava.model.language_model.llava_llama import (
            LlavaLlamaForCausalLM,
            LlavaConfig,
        )
        from llava.constants import (
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )

        print(f"Loading LLaVA-NeXT-Video from: {self.local_path}")
        print(f"  conv_template: {self.conv_template}")

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.local_path, use_fast=False
        )

        # Model config — override vision tower path BEFORE from_pretrained
        # so CLIPVisionTower.__init__ uses the local path directly and
        # never tries to download from HuggingFace.
        llava_cfg = LlavaConfig.from_pretrained(self.local_path)
        if self.clip_local_path and os.path.isdir(self.clip_local_path):
            print(f"  Overriding mm_vision_tower → {self.clip_local_path}")
            llava_cfg.mm_vision_tower = self.clip_local_path

        # Extend context by increasing max_position_embeddings.
        # RoPE is computed on-the-fly (not a learned table), so the model
        # can handle longer positions. Quality degrades gracefully for
        # positions beyond the original 4096 training range.
        rope_factor = self.config.get("rope_scaling_factor", None)
        if rope_factor and rope_factor > 1.0:
            new_ctx = int(4096 * rope_factor)
            llava_cfg.max_position_embeddings = new_ctx
            # LlavaConfig doesn't inherit rope_theta from LlamaConfig,
            # so we must set it explicitly for RoPE scaling to work.
            if not hasattr(llava_cfg, "rope_theta") or llava_cfg.rope_theta is None:
                llava_cfg.rope_theta = 10000.0
            llava_cfg.rope_scaling = {
                "rope_type": "dynamic",
                "factor": float(rope_factor),
                "original_max_position_embeddings": 4096,
            }
            print(f"  RoPE dynamic scaling: factor={rope_factor} → context={new_ctx}")

        load_kwargs = dict(
            low_cpu_mem_usage=True,
            config=llava_cfg,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        try:
            self._model = LlavaLlamaForCausalLM.from_pretrained(
                self.local_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
        except (ImportError, ValueError) as e:
            print(f"  flash_attention_2 unavailable ({e}), using default")
            self._model = LlavaLlamaForCausalLM.from_pretrained(
                self.local_path, **load_kwargs,
            )

        # Token embeddings
        if getattr(self._model.config, "mm_use_im_patch_token", True):
            self._tokenizer.add_tokens(
                [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
            )
        if getattr(self._model.config, "mm_use_im_start_end", False):
            self._tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
        self._model.resize_token_embeddings(len(self._tokenizer))

        # Vision tower — should already be loaded via from_pretrained
        # with the local CLIP path. Ensure it's on GPU.
        vision_tower = self._model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=torch.float16)
        self._image_processor = vision_tower.image_processor

        self._model.eval()

        # LlavaConfig defines class-level generation params
        # (temperature=0.0, do_sample=False, max_new_tokens=1024, top_p=None)
        # that conflict with transformers 5.x generation validation.
        # Override them on the instance with None so generate() ignores them,
        # then configure generation_config which is what generate() reads.
        self._model.config.temperature = None
        self._model.config.do_sample = None
        self._model.config.top_p = None
        self._model.config.max_new_tokens = None
        self._model.generation_config.do_sample = False
        self._model.generation_config.max_new_tokens = self.max_tokens

        self._context_len = getattr(
            self._model.config, "max_position_embeddings", 4096
        )
        # Ensure generation max_length matches the (possibly extended) context
        self._model.generation_config.max_length = self._context_len
        print(f"LLaVA-NeXT-Video loaded. context_len={self._context_len}")

    def _prepare_and_generate(self, frames: List[Image.Image], prompt: str) -> str:
        """Build conversation, process images, and generate response.

        For video input: uses a single <image> token with modalities=["video"]
        so the spatial_pool resampler (stride=2) reduces tokens from 576 to
        144 per frame. The images tensor is [num_frames, C, H, W] wrapped in
        a list to represent one video item.
        """
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token

        # Single <image> token for the entire video, NOT one per frame
        user_message = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        # Tokenize with image token placeholder
        input_ids = tokenizer_image_token(
            full_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._model.device)

        # Process frames directly through the image processor without
        # anyres multi-crop (which is for single-image mode). For video,
        # each frame is simply resized to 336×336.
        if frames:
            pixel_values = self._image_processor.preprocess(
                frames, return_tensors="pt"
            )["pixel_values"]  # (num_frames, C, 336, 336)
            pixel_values = pixel_values.to(
                self._model.device, dtype=torch.float16
            )
            # Wrap in list: one video item → [tensor(num_frames, C, H, W)]
            images_tensor = [pixel_values]
        else:
            images_tensor = None

        # Generate with modalities=["video"] to trigger spatial pooling
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[frames[0].size] if frames else None,
                modalities=["video"],
                use_cache=True,
            )

        # LLaVA's generate() passes inputs_embeds (not input_ids) to
        # super().generate(), so the returned output_ids contain ONLY
        # the newly generated tokens — no input prefix to skip.
        response = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return response

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """Run single inference."""
        # Image-option path (task 4.3.x): append option PIL images as trailing
        # frames; LLaVA-NeXT-Video processes them through the same per-frame
        # CLIP encoder as the video frames.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )
        self._init_model()
        return self._prepare_and_generate(frames, prompt)
