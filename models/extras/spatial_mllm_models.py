"""
Spatial-MLLM offline inference for OVO-S evaluation.
Supports both standard uniform sampling and space-aware (SA) frame sampling.

Requires:
    - Spatial-MLLM source code at _src/Spatial-MLLM/
    - transformers, torch, flash-attn, qwen-vl-utils
"""

import os
import sys
import importlib.util
import importlib.abc
import types
from contextlib import contextmanager
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image

from ..base import BaseModel, resolve_runtime_path

# Add Spatial-MLLM source to path
_SPATIAL_MLLM_SRC = str(Path(__file__).resolve().parent.parent.parent / "_src" / "Spatial-MLLM")
if _SPATIAL_MLLM_SRC not in sys.path:
    sys.path.insert(0, _SPATIAL_MLLM_SRC)

# VGGT internal imports use `from vggt.xxx`, which requires the external dir on sys.path
_VGGT_EXTERNAL = str(Path(__file__).resolve().parent.parent.parent / "_src" / "Spatial-MLLM" / "src" / "qwenvl" / "external")
if _VGGT_EXTERNAL not in sys.path:
    sys.path.insert(0, _VGGT_EXTERNAL)


@contextmanager
def _hide_deepspeed_from_transformers():
    """Avoid DeepSpeed CUDA probes during Transformers imports (can segfault on some GPUs)."""
    class _DeepSpeedBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "deepspeed" or fullname.startswith("deepspeed."):
                raise ModuleNotFoundError(fullname)
            return None

    original_find_spec = importlib.util.find_spec
    original_meta_path = list(sys.meta_path)
    original_deepspeed = sys.modules.get("deepspeed")
    had_deepspeed = "deepspeed" in sys.modules
    patched_modules = []

    dummy_deepspeed = types.ModuleType("deepspeed")

    class _DummyGatheredParameters:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_deepspeed.zero = types.SimpleNamespace(GatheredParameters=_DummyGatheredParameters)
    dummy_deepspeed.comm = types.SimpleNamespace(get_rank=lambda: 0)

    def find_spec_without_deepspeed(name, *args, **kwargs):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = find_spec_without_deepspeed
    sys.modules["deepspeed"] = dummy_deepspeed
    sys.meta_path.insert(0, _DeepSpeedBlocker())
    try:
        # Transformers copies is_deepspeed_available into modeling_utils during
        # import. Patch the integration module first so modeling_utils never
        # reaches `import deepspeed`, which can segfault on some GPUs.
        try:
            import transformers.integrations.deepspeed as ds_integration

            patched_modules.append(
                (ds_integration, "is_deepspeed_available", ds_integration.is_deepspeed_available)
            )
            ds_integration.is_deepspeed_available = lambda: False
        except Exception:
            pass
        try:
            import transformers.integrations as integrations

            if hasattr(integrations, "is_deepspeed_available"):
                patched_modules.append(
                    (integrations, "is_deepspeed_available", integrations.is_deepspeed_available)
                )
                integrations.is_deepspeed_available = lambda: False
        except Exception:
            pass
        if "transformers.modeling_utils" in sys.modules:
            modeling_utils = sys.modules["transformers.modeling_utils"]
            if hasattr(modeling_utils, "is_deepspeed_available"):
                patched_modules.append(
                    (modeling_utils, "is_deepspeed_available", modeling_utils.is_deepspeed_available)
                )
                modeling_utils.is_deepspeed_available = lambda: False
        yield
    finally:
        for module, attr, value in reversed(patched_modules):
            setattr(module, attr, value)
        if had_deepspeed:
            sys.modules["deepspeed"] = original_deepspeed
        else:
            sys.modules.pop("deepspeed", None)
        sys.meta_path = original_meta_path
        importlib.util.find_spec = original_find_spec


class SpatialMLLMModel(BaseModel):
    """Offline inference for Spatial-MLLM (Qwen2.5-VL + VGGT spatial encoder)."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(config.get("local_path", config.get("model_id")))
        self.top_p = config.get("top_p", 0.001)
        self.nframes = config.get("nframes", 16)

        self.model = None
        self.processor = None

    def _init_model(self):
        """Lazy initialization of Spatial-MLLM model."""
        if self.model is not None:
            return

        with _hide_deepspeed_from_transformers():
            from transformers import Qwen2_5_VLProcessor
            from src.qwenvl.model.spatial_mllm import (
                SpatialMLLMConfig,
                SpatialMLLMForConditionalGeneration,
            )

        print(f"Loading Spatial-MLLM from: {self.local_path}")
        config = SpatialMLLMConfig.from_pretrained(self.local_path)
        self.model = SpatialMLLMForConditionalGeneration.from_pretrained(
            self.local_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(self.local_path)
        print("Spatial-MLLM loaded successfully.")

    def _prepare_spatial_inputs(self, batch, video_inputs, image_inputs):
        """Prepare video_tchw / image_tchw tensors for the spatial encoder.

        The spatial encoder expects each element to be [T, C, H, W] (4D).
        """
        video_tchw = []
        image_tchw = []

        if video_inputs:
            for vi in video_inputs:
                if isinstance(vi, torch.Tensor):
                    vi = vi.float() / 255.0 if vi.dtype == torch.uint8 else vi.float()
                elif isinstance(vi, list) and all(isinstance(img, Image.Image) for img in vi):
                    vi = torch.stack([
                        torch.tensor(np.array(img)).permute(2, 0, 1) for img in vi
                    ]).float() / 255.0
                else:
                    raise ValueError(f"Unsupported video input type: {type(vi)}")
                video_tchw.append(vi)

        if image_inputs:
            for img in image_inputs:
                if isinstance(img, Image.Image):
                    t = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                    image_tchw.append(t.unsqueeze(0))  # [1, C, H, W]
                else:
                    raise ValueError(f"Unsupported image input type: {type(img)}")

        batch["video_tchw"] = video_tchw if video_tchw else None
        batch["image_tchw"] = image_tchw if image_tchw else None
        return batch

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """Run inference with Spatial-MLLM."""
        # Image-option path (task 4.3.x): append option PIL images to the
        # trailing frames so the video processor sees them; the prompt header
        # already labels the last N frames as options A, B, ... in order.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )
        self._init_model()

        from qwen_vl_utils import process_vision_info

        # Pass frames as a single video — this is how Spatial-MLLM expects input.
        # The processor will produce pixel_values_videos + video_grid_thw,
        # and the model forward will use the video_tchw path for spatial encoding.
        content = [
            {
                "type": "video",
                "video": frames,  # list of PIL Images treated as video frames
                "nframes": len(frames),
            },
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        # Tokenize
        prompts_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        batch = self.processor(
            text=[prompts_text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        # Prepare spatial encoder inputs
        batch = self._prepare_spatial_inputs(batch, video_inputs, image_inputs)
        batch.to(self.model.device)
        if batch.get("video_tchw") is not None:
            batch["video_tchw"] = [t.to(self.model.device) for t in batch["video_tchw"]]
        if batch.get("image_tchw") is not None:
            batch["image_tchw"] = [t.to(self.model.device) for t in batch["image_tchw"]]

        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=max(self.temperature, 0.01),
            top_p=self.top_p,
            use_cache=True,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(**batch, **gen_kwargs)

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(batch["input_ids"], generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output[0].strip() if output else ""


class SpatialMLLMSASamplingModel(SpatialMLLMModel):
    """Spatial-MLLM with space-aware frame sampling via VGGT.

    Overrides frame extraction to use VGGT-based maximum-coverage sampling
    instead of uniform temporal sampling.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.vggt_local_path = resolve_runtime_path(config.get("vggt_local_path"))
        self.vggt_model_path = config.get("vggt_model_path", "facebook/VGGT-1B")
        self.sa_candidate_frames = config.get("sa_candidate_frames", 128)
        self._vggt = None

    def _init_vggt(self):
        if self._vggt is not None:
            return
        with _hide_deepspeed_from_transformers():
            from src.qwenvl.external.vggt.models.vggt import VGGT
        vggt_path = self.vggt_local_path or self.vggt_model_path
        print(f"Loading VGGT from: {vggt_path}")
        self._vggt = VGGT.from_pretrained(vggt_path).to("cuda")
        self._vggt.eval()
        print("VGGT loaded successfully.")

    def extract_frames(self, video_path: str, query_time: float) -> List[Image.Image]:
        """Space-aware frame sampling using VGGT 3D geometry predictions."""
        import cv2
        from torchvision import transforms as TF
        from src.sampling.sa_sampling import (
            compute_voxel_sets,
            maximum_coverage_sampling,
        )
        from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self._init_vggt()

        # Step 1: Extract candidate frames from video up to query_time
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        end_frame = int(query_time * video_fps)
        if end_frame <= 0:
            end_frame = 1

        n_candidates = min(self.sa_candidate_frames, end_frame + 1)
        candidate_indices = np.linspace(0, end_frame, n_candidates, dtype=int)

        pil_frames = []
        frame_idx = 0
        target_set = set(candidate_indices.tolist())
        max_target = int(candidate_indices[-1])

        while frame_idx <= max_target:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb))
            frame_idx += 1
        cap.release()

        if len(pil_frames) <= self.max_frames:
            return self._resize_frames(pil_frames)

        # Step 2: Preprocess for VGGT (resize to 518px)
        to_tensor = TF.ToTensor()
        target_size = 518
        tensors = []
        for img in pil_frames:
            w, h = img.size
            if h > w:
                img = img.rotate(-90, expand=True)
                w, h = img.size
            new_w = target_size
            new_h = round(h * (new_w / w) / 14) * 14
            img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            t = to_tensor(img_resized)
            if new_h > target_size:
                start_y = (new_h - target_size) // 2
                t = t[:, start_y:start_y + target_size, :]
            tensors.append(t)

        # Pad to uniform size
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded = []
        for t in tensors:
            ph = max_h - t.shape[1]
            pw = max_w - t.shape[2]
            if ph > 0 or pw > 0:
                t = torch.nn.functional.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=1.0)
            padded.append(t)

        images_tensor = torch.stack(padded).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

        # Step 3: Run VGGT and select frames
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = self._vggt(images_tensor)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images_tensor.shape[-2:]
        )
        world_points = predictions["world_points"]
        world_points_conf = predictions["world_points_conf"]
        wp_flat = world_points.reshape(-1, 3)
        wpc_flat = world_points_conf.reshape(-1)

        threshold = np.percentile(wpc_flat.cpu().numpy(), 50)
        conf_mask = (world_points_conf >= threshold) & (world_points_conf > 0.1)
        flat_mask = (wpc_flat >= threshold) & (wpc_flat > 0.1)

        valid_pts = wp_flat[flat_mask]
        mins = valid_pts.min(dim=0)[0]
        maxs = valid_pts.max(dim=0)[0]
        voxel_size = min((maxs - mins).tolist()) / 20
        if voxel_size <= 0:
            voxel_size = 0.1

        voxel_sets = compute_voxel_sets(
            world_points, conf_mask,
            mins[0].item(), mins[1].item(), mins[2].item(), voxel_size
        )
        selected = sorted(maximum_coverage_sampling(voxel_sets, self.max_frames))

        torch.cuda.empty_cache()

        selected_frames = [pil_frames[i] for i in selected]
        return self._resize_frames(selected_frames)

    def _resize_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Resize frames to max frame_size dimension."""
        resized = []
        for img in frames:
            w, h = img.size
            if max(w, h) > self.frame_size:
                scale = self.frame_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            resized.append(img)
        return resized
