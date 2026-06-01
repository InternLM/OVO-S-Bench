"""
Spatial-TTT offline inference for OVO-S evaluation.
Wraps the Spatial-TTT model (Qwen3-VL-2B + LaCT test-time training layers).

Requires:
    - Spatial-TTT source code at _src/Spatial-TTT/
    - transformers>=4.57, torch, flash-attn, qwen-vl-utils, safetensors, triton
"""

import sys
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image

from ..base import BaseModel, resolve_runtime_path

# Spatial-TTT source root (contains 'models/' package with spatial_ttt.py)
_SPATIAL_TTT_SRC = str(
    Path(__file__).resolve().parent.parent.parent
    / "_src"
    / "Spatial-TTT"
    / "qwen-vl-finetune"
)


def _import_spatial_ttt_loader():
    """Import load_spatial_ttt_model from Spatial-TTT source.

    Handles the package name collision between eval/models/ and
    Spatial-TTT/qwen-vl-finetune/models/ by temporarily manipulating
    sys.path and sys.modules.
    """
    # If already imported, return cached reference
    if "spatial_ttt_models_pkg.spatial_ttt" in sys.modules:
        return sys.modules["spatial_ttt_models_pkg.spatial_ttt"].load_spatial_ttt_model

    # Step 1: Save and remove ALL 'models' entries from sys.modules
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "models" or key.startswith("models."):
            saved_modules[key] = sys.modules.pop(key)

    # Step 2: Ensure _SPATIAL_TTT_SRC is first on sys.path
    if _SPATIAL_TTT_SRC in sys.path:
        sys.path.remove(_SPATIAL_TTT_SRC)
    sys.path.insert(0, _SPATIAL_TTT_SRC)

    try:
        # Step 3: Import the Spatial-TTT models package fresh
        import importlib
        stt_models_pkg = importlib.import_module("models")
        stt_spatial_ttt = importlib.import_module("models.spatial_ttt")
        load_fn = stt_spatial_ttt.load_spatial_ttt_model

        # Step 4: Re-register under unique alias for caching
        for key in list(sys.modules.keys()):
            if key == "models" or key.startswith("models."):
                alias_key = key.replace("models", "spatial_ttt_models_pkg", 1)
                sys.modules[alias_key] = sys.modules[key]

        return load_fn

    finally:
        # Step 5: Remove the Spatial-TTT 'models' entries
        for key in list(sys.modules.keys()):
            if key == "models" or key.startswith("models."):
                sys.modules.pop(key, None)

        # Step 6: Restore the original eval/models entries
        sys.modules.update(saved_modules)

        # Step 7: Restore sys.path order
        if _SPATIAL_TTT_SRC in sys.path:
            sys.path.remove(_SPATIAL_TTT_SRC)
        sys.path.append(_SPATIAL_TTT_SRC)


class SpatialTTTModel(BaseModel):
    """Offline inference for Spatial-TTT (Qwen3-VL + LaCT TTT layers)."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.base_model_path = resolve_runtime_path(
            config.get("base_model_path", "Qwen/Qwen3-VL-2B-Instruct")
        )
        self.checkpoint_path = resolve_runtime_path(
            config.get("checkpoint_path") or config.get("local_path")
        )
        self.num_lact_heads = config.get("num_lact_heads", 4)
        self.w0_w2_low_rank = config.get("w0_w2_low_rank", 0)
        self.use_fused_kernel = config.get("use_fused_kernel", False)
        self.use_conv_layer = config.get("use_conv_layer", True)
        self.lact_chunk_size = config.get("lact_chunk_size", 2648)
        self.window_size = config.get("window_size", 2648)
        self.lact_layers = config.get(
            "lact_layers",
            "0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26",
        )
        self.resize_height = config.get("resize_height", 352)
        self.resize_width = config.get("resize_width", 480)

        self._stt_model = None
        self.processor = None

    def _init_model(self):
        """Lazy initialization of Spatial-TTT model."""
        if self._stt_model is not None:
            return

        from transformers import AutoProcessor

        load_spatial_ttt_model = _import_spatial_ttt_loader()

        print(f"Loading Spatial-TTT base model: {self.base_model_path}")
        print(f"Loading LaCT checkpoint: {self.checkpoint_path}")
        self._stt_model = load_spatial_ttt_model(
            model_path=self.base_model_path,
            num_lact_heads=self.num_lact_heads,
            w0_w2_low_rank=self.w0_w2_low_rank,
            use_fused_kernel=self.use_fused_kernel,
            use_conv_layer=self.use_conv_layer,
            lact_chunk_size=self.lact_chunk_size,
            window_size=self.window_size,
            lact_layers=self.lact_layers,
            checkpoint_path=self.checkpoint_path,
            device="cuda",
        )
        self._stt_model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.base_model_path,
            max_pixels=1605632,
            min_pixels=256 * 28 * 28,
        )
        print("Spatial-TTT loaded successfully.")

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """Run inference with Spatial-TTT."""
        # Image-option path (task 4.3.x): append option PIL images as trailing
        # frames; Spatial-TTT then duplicates each frame (including options) in
        # its image-as-video pipeline below.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )
        self._init_model()

        from qwen_vl_utils import process_vision_info

        # Duplicate each frame (Spatial-TTT evaluation convention for image-as-video)
        video_frames = []
        for f in frames:
            video_frames.extend([f, f])

        content = [
            {
                "type": "video",
                "video": video_frames,
                "resized_height": self.resize_height,
                "resized_width": self.resize_width,
                "sample_fps": 1,
            },
            {"type": "text", "text": prompt},
        ]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[texts],
            images=image_inputs,
            videos=video_inputs,
            video_metadatas=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self._stt_model.generate_with_spatial_ttt(
                inputs["input_ids"],
                pixel_values_videos=inputs.get("pixel_values_videos", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                max_new_tokens=self.max_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0].strip() if output else ""
