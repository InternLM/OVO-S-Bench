"""
StreamForest offline inference for OVO-S evaluation.

Wraps the upstream StreamForest model (LLaVA-style architecture with a Qwen2 LLM
backbone, persistent event-memory forest, fine-grained spatiotemporal window).
Mirrors the upstream `lmms_eval/models/streamforest.py` pipeline:
    1. load_pretrained_model -> tokenizer / model / image_processor
    2. preprocess pre-extracted frames -> [T, C, H, W] half-precision tensor
    3. build a conv (`qwen_2`) with a single <image> placeholder + time_msg
    4. tokenizer_image_token + model.generate(..., images=[tensor], modalities=["video"])

Requires:
    - conda env: streamforest (or any env with the StreamForest llava package importable)
    - source tree at extras_src/StreamForest (added to sys.path lazily; override via OVO_S_STREAMFOREST_SRC)
    - flash-attn / sdpa attention available
"""

import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from ..base import BaseModel, resolve_runtime_path
from ._paths import find_upstream_src

_STREAMFOREST_SRC = find_upstream_src("StreamForest", strict=False)


def _resolve_cluster_path(path: str | None) -> str | None:
    """Backwards-compatible alias around the package-wide path resolver."""
    return resolve_runtime_path(path)


class StreamForestModel(BaseModel):
    """LLaVA-style StreamForest wrapper using a single <image> token + video modality."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = _resolve_cluster_path(config.get("local_path", config.get("model_id")))
        self.conv_template = config.get("conv_template", "qwen_2")
        self.attn_implementation = config.get("attn_implementation", "sdpa")
        self.max_frames_num = config.get("max_frames_num", config.get("max_frames", 32))
        self.token_strategy = config.get("token_strategy", "single")
        self.time_msg = config.get("time_msg", "short")
        self.use_cache = config.get("use_cache", True)
        self.mm_local_num_frames = config.get("mm_local_num_frames", None)
        self.mm_projector_type = config.get("mm_projector_type", None)
        self.vision_encode_type = config.get("vision_encode_type", None)
        self.mm_vision_tower_override = _resolve_cluster_path(config.get("mm_vision_tower", None))
        self.device = config.get("device", "cuda:0")

        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._max_length = None
        self._stop_str = None
        self._image_token_index = None

    def _init_model(self):
        if self._model is not None:
            return

        if _STREAMFOREST_SRC not in sys.path:
            sys.path.insert(0, _STREAMFOREST_SRC)

        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from transformers import AutoConfig

        cfg_pretrained = AutoConfig.from_pretrained(
            self.local_path, trust_remote_code=True
        )

        overwrite_config = {}
        if self.mm_local_num_frames is not None:
            overwrite_config["mm_local_num_frames"] = self.mm_local_num_frames
        if self.mm_projector_type:
            overwrite_config["mm_projector_type"] = self.mm_projector_type
        if self.vision_encode_type:
            overwrite_config["vision_encode_type"] = self.vision_encode_type
        if self.mm_vision_tower_override:
            overwrite_config["mm_vision_tower"] = self.mm_vision_tower_override

        llava_args = {
            "multimodal": True,
            "attn_implementation": self.attn_implementation,
            "overwrite_config": overwrite_config,
        }

        model_name = get_model_name_from_path(self.local_path)
        print(f"Loading StreamForest from: {self.local_path}")
        print(f"  conv_template={self.conv_template}, attn={self.attn_implementation}, "
              f"max_frames_num={self.max_frames_num}, time_msg={self.time_msg}")

        try:
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    self.local_path, None, model_name,
                    device_map=self.device, **llava_args
                )
            )
        except TypeError:
            llava_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    self.local_path, None, model_name,
                    device_map=self.device, **llava_args
                )
            )

        self._model.eval()
        self._image_token_index = IMAGE_TOKEN_INDEX

        conv = conv_templates[self.conv_template].copy()
        self._stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        print("StreamForest loaded.")

    def _preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """Run the vision tower's image processor on a list of PIL frames.

        Returns a [T, C, H, W] half-precision tensor on the model device.
        """
        imgs: List[Image.Image] = []
        for f in frames:
            if isinstance(f, Image.Image):
                imgs.append(f)
            elif isinstance(f, np.ndarray):
                imgs.append(Image.fromarray(f))
            elif isinstance(f, torch.Tensor):
                arr = f.detach().cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = arr.transpose(1, 2, 0)
                imgs.append(Image.fromarray(arr.astype(np.uint8)))
            else:
                raise TypeError(f"Unsupported frame type: {type(f)}")

        if not imgs:
            raise ValueError("No frames provided to StreamForest")

        if len(imgs) > self.max_frames_num:
            idxs = np.linspace(0, len(imgs) - 1, self.max_frames_num).round().astype(int)
            imgs = [imgs[i] for i in idxs]

        processed = self._image_processor.preprocess(imgs, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        return pixel_values.half().to(self._model.device)

    def _build_time_msg(self, num_frames: int) -> str:
        """Mimic the upstream `time_msg` strings.

        For pre-extracted frames we lack absolute timestamps so we fall back to
        the `short` template (no online-progression phrasing). When `time_msg`
        is empty / None, no message is added.
        """
        msg = self.time_msg
        if not msg:
            return ""
        if msg == "short":
            return f"\nThere are {num_frames} frames uniformly sampled from the video. "
        if msg in ("short_online", "short_online_v2"):
            return (
                f"\nThe video segment contains {num_frames} frames sampled from "
                f"the recent past up to the present moment. "
            )
        return ""

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        # Image-option path (task 4.3.x): append option PIL images as trailing
        # frames so the existing _preprocess_frames + per-frame token pipeline
        # picks them up; prompt header already labels the last N as options.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames_num
            )
        self._init_model()

        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

        image_tensor = self._preprocess_frames(frames)
        n_frames = image_tensor.shape[0]

        time_msg = self._build_time_msg(n_frames)
        image_token = (
            DEFAULT_IMAGE_TOKEN if self.token_strategy == "single"
            else " ".join([DEFAULT_IMAGE_TOKEN] * n_frames)
        )
        question = f"{image_token}\n{time_msg.rstrip()} {prompt}".strip()

        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self._tokenizer, self._image_token_index, return_tensors="pt"
        ).unsqueeze(0).to(self._model.device)
        attention_mask = input_ids.ne(
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id
        ).to(self._model.device)
        pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id
        )

        stopping_criteria = KeywordsStoppingCriteria(
            [self._stop_str], self._tokenizer, input_ids
        )

        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            modalities=["video"],
            stopping_criteria=[stopping_criteria],
            use_cache=self.use_cache,
        )
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_id,
                images=[image_tensor],
                **gen_kwargs,
            )

        response = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        # Trim any trailing stop string
        if self._stop_str and response.endswith(self._stop_str):
            response = response[: -len(self._stop_str)].rstrip()
        return response
