"""
Flash-VStream (Qwen2-VL) inference wrapper for OVO-S evaluation.

The upstream Flash-VStream repository ships a custom Qwen2-VL model and
processor under ``_src/Flash-VStream/Flash-VStream-Qwen``. Those modules use a
few private Transformers symbols from the 4.45 era; this wrapper installs a
small compatibility shim before loading the upstream package so it can run in
our closest local env (``spatial-mllm``: torch 2.6, flash-attn 2.7.4).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from ..base import BaseModel, resolve_runtime_path
from ._paths import find_upstream_src

# Lazy-resolved upstream paths. The defaults point to the standard release
# conventions; override via OVO_S_FLASH_VSTREAM_SRC / OVO_S_MODEL_ROOT.
_DEFAULT_SRC = find_upstream_src("Flash-VStream", strict=False)
_DEFAULT_MODEL = os.path.join(_DEFAULT_SRC, "Flash-VStream-Qwen-7b")
_FLASH_MODELS_ALIAS = "_ovos_flash_vstream_models"


@contextmanager
def _hide_deepspeed_from_transformers():
    """Avoid importing DeepSpeed during Transformers model imports.

    In the spatial-mllm env, importing DeepSpeed on H200 can segfault while it
    probes CUDA/Triton support. Flash-VStream only needs plain inference here,
    so make Transformers treat DeepSpeed as unavailable.
    """
    original_find_spec = importlib.util.find_spec

    def find_spec_without_deepspeed(name, *args, **kwargs):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = find_spec_without_deepspeed
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec


def _resolve_cluster_path(path: str | None) -> str | None:
    """Backwards-compatible alias around the package-wide path resolver."""
    return resolve_runtime_path(path)


def _install_transformers_compat():
    """Expose the old Qwen2-VL mask helper expected by Flash-VStream."""
    with _hide_deepspeed_from_transformers():
        import transformers.models.qwen2_vl.modeling_qwen2_vl as qwen2_vl
        import transformers.models.qwen2_vl.image_processing_qwen2_vl as qwen2_vl_image
        from transformers.image_utils import is_valid_image

    if not hasattr(qwen2_vl, "_prepare_4d_causal_attention_mask_with_cache_position"):
        def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: torch.Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            device: torch.device,
            min_dtype: float,
            cache_position: torch.Tensor,
            batch_size: int,
        ):
            if attention_mask is not None and attention_mask.dim() == 4:
                return attention_mask

            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            return causal_mask

        qwen2_vl._prepare_4d_causal_attention_mask_with_cache_position = (  # type: ignore[attr-defined]
            _prepare_4d_causal_attention_mask_with_cache_position
        )

    if not hasattr(qwen2_vl_image, "make_batched_images"):
        def make_batched_images(images):
            if (
                isinstance(images, (list, tuple))
                and images
                and isinstance(images[0], (list, tuple))
                and images[0]
                and is_valid_image(images[0][0])
            ):
                return [img for img_list in images for img in img_list]
            if isinstance(images, (list, tuple)) and images and is_valid_image(images[0]):
                return images
            if is_valid_image(images):
                return [images]
            raise ValueError(f"Could not make batched images from {images}")

        qwen2_vl_image.make_batched_images = make_batched_images  # type: ignore[attr-defined]

    if not hasattr(qwen2_vl_image, "make_batched_videos"):
        def make_batched_videos(videos):
            if (
                isinstance(videos, (list, tuple))
                and videos
                and isinstance(videos[0], (list, tuple))
                and videos[0]
                and is_valid_image(videos[0][0])
            ):
                return videos
            if isinstance(videos, (list, tuple)) and videos and is_valid_image(videos[0]):
                return [videos]
            if is_valid_image(videos) and hasattr(videos, "shape") and len(videos.shape) == 4:
                return [list(videos)]
            return videos

        qwen2_vl_image.make_batched_videos = make_batched_videos  # type: ignore[attr-defined]


def _load_flash_source(src_path: str):
    """Load upstream Flash-VStream ``models`` package under a private alias."""
    src = Path(_resolve_cluster_path(src_path)).resolve()
    init_py = src / "models" / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(f"Flash-VStream source not found: {src}")

    # Prefer the upstream qwen_vl_utils bundled with Flash-VStream.
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    existing_qwen_utils = sys.modules.get("qwen_vl_utils")
    if existing_qwen_utils is not None:
        module_file = getattr(existing_qwen_utils, "__file__", "") or ""
        if not module_file.startswith(str(src)):
            del sys.modules["qwen_vl_utils"]

    _install_transformers_compat()

    loaded = sys.modules.get(_FLASH_MODELS_ALIAS)
    if loaded is not None:
        return loaded

    spec = importlib.util.spec_from_file_location(
        _FLASH_MODELS_ALIAS,
        init_py,
        submodule_search_locations=[str(src / "models")],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Flash-VStream package from {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_FLASH_MODELS_ALIAS] = module
    spec.loader.exec_module(module)
    return module


class FlashVStreamQwenModel(BaseModel):
    """OVO-S wrapper around the upstream Flash-VStream-Qwen checkpoint."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.src_path = config.get("src_path", _DEFAULT_SRC)
        self.local_path = config.get("local_path", _DEFAULT_MODEL)
        self.processor_path = config.get("processor_path", self.local_path)
        self.attn_implementation = config.get("attn_implementation", "flash_attention_2")
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.device = config.get("device", "cuda")
        self.max_pixels = config.get("max_pixels", 224 * 224)
        self.min_pixels = config.get("min_pixels", None)
        self.resized_height = config.get("resized_height", None)
        self.resized_width = config.get("resized_width", None)
        self.load_on_cpu_first = config.get("load_on_cpu_first", True)
        self.append_best_option_prefix = config.get("append_best_option_prefix", True)
        self.nframes = config.get("nframes", self.max_frames)

        self._model = None
        self._processor = None
        self._flash_memory_config = None

    @staticmethod
    def _to_pil(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            return Image.fromarray(frame).convert("RGB")
        if isinstance(frame, torch.Tensor):
            arr = frame.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def _dtype(self):
        if self.torch_dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if self.torch_dtype in ("fp16", "float16"):
            return torch.float16
        if self.torch_dtype in ("fp32", "float32"):
            return torch.float32
        return self.torch_dtype

    def _init_model(self):
        if self._model is not None:
            return

        if torch.cuda.is_available():
            # Initialize CUDA before importing/loading the custom Flash-VStream
            # stack.  In this H200 runtime, delayed lazy init can segfault after
            # Transformers/DeepSpeed/PyAV side imports are already loaded.
            torch.cuda.init()
            torch.cuda.current_device()

        flash = _load_flash_source(self.src_path)
        model_path = _resolve_cluster_path(self.local_path)
        processor_path = _resolve_cluster_path(self.processor_path)

        print(f"Loading Flash-VStream-Qwen from: {model_path}")
        print(f"  source={_resolve_cluster_path(self.src_path)}")
        print(f"  processor={processor_path}")

        model_config = flash.FlashVStreamQwen2VLConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        flash_memory_config = getattr(model_config.vision_config, "flash_memory_config", None)
        if flash_memory_config is None:
            flash_memory_config = dict(flash.DEFAULT_FLASH_MEMORY_CONFIG)
            model_config.vision_config.flash_memory_config = flash_memory_config
        for key, value in flash.DEFAULT_FLASH_MEMORY_CONFIG.items():
            flash_memory_config.setdefault(key, value)

        attn = self.attn_implementation if torch.cuda.is_available() else "eager"
        load_kwargs = dict(
            config=model_config,
            trust_remote_code=True,
            torch_dtype=self._dtype(),
            attn_implementation=attn,
        )
        if torch.cuda.is_available() and not self.load_on_cpu_first:
            load_kwargs["device_map"] = self.device
        self._model = flash.FlashVStreamQwen2VLModel.from_pretrained(
            model_path,
            **load_kwargs,
        ).eval()
        if torch.cuda.is_available() and self.load_on_cpu_first:
            # Avoid Transformers' device_map allocator warmup, which segfaults
            # for this custom model in the spatial-mllm H200 runtime.
            self._model.to(self.device)
        self._processor = flash.FlashVStreamQwen2VLProcessor.from_pretrained(
            processor_path,
            size={"shortest_edge": 1, "longest_edge": 10**9},
        )
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None:
            size = getattr(image_processor, "size", None)
            if not isinstance(size, dict) or not {"shortest_edge", "longest_edge"} <= set(size):
                # Newer Transformers validates this legacy field even though
                # Flash-VStream's custom preprocessor uses min/max pixels.
                image_processor.size = {"shortest_edge": 1, "longest_edge": 10**9}
        self._flash_memory_config = flash_memory_config
        print(f"Flash-VStream loaded. flash_memory_config={flash_memory_config}")

    def _build_video_content(self, frames: List[Image.Image]) -> Dict[str, Any]:
        content = {"type": "video", "video": frames, "max_frames": self.max_frames}
        if self.max_pixels is not None:
            content["max_pixels"] = self.max_pixels
        if self.min_pixels is not None:
            content["min_pixels"] = self.min_pixels
        if self.resized_height is not None and self.resized_width is not None:
            content["resized_height"] = self.resized_height
            content["resized_width"] = self.resized_width
        return content

    def inference(self, frames: List[Any], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        # Image-option path (task 4.3.x): append option PIL images as trailing
        # frames; _to_pil handles non-PIL conversions and the model's vision
        # encoder treats them as additional frames before the prompt.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )
        self._init_model()
        from qwen_vl_utils import process_vision_info

        pil_frames = [self._to_pil(f) for f in frames]
        if not pil_frames:
            return ""

        messages = [
            {
                "role": "user",
                "content": [
                    self._build_video_content(pil_frames),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.append_best_option_prefix:
            text += "Best option: ("

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            flash_memory_config=self._flash_memory_config,
        )
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generate_kwargs = dict(
            max_new_tokens=self.max_tokens,
            top_k=1,
            do_sample=False,
        )

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generate_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
