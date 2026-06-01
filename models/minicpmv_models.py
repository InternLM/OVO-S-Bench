"""Native Transformers wrapper for MiniCPM-V models."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseModel, resolve_runtime_path

# Keep Hugging Face dynamic modules in a user-writable location.
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_MODULES_CACHE"] = os.path.expanduser("~/.cache/huggingface/modules")


@dataclass
class MiniCPMVFrameBatch:
    """Frames plus MiniCPM-V temporal metadata for video chat."""

    frames: List[Image.Image]
    temporal_ids: Optional[List[List[int]]] = None

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Image.Image]:
        return iter(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class MiniCPMVModel(BaseModel):
    """MiniCPM-V 4.x inference through the official Transformers chat API.

    The wrapper uses MiniCPM-V's native `model.chat()` path rather than vLLM so
    it can pass `temporal_ids` for the 3D video resampler.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(config.get("local_path", config.get("model_id")))
        self.attn_implementation = config.get("attn_implementation", "sdpa")
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.enable_thinking = config.get("enable_thinking", False)
        self.sampling = config.get("sampling", False)
        self.max_inp_length = config.get("max_inp_length", 16384)
        self.max_slice_nums = config.get("max_slice_nums", 1)
        self.use_image_id = config.get("use_image_id", False)
        self.use_temporal_ids = config.get("use_temporal_ids", True)
        self.max_packing = int(config.get("max_packing", 3))
        self.time_scale = float(config.get("time_scale", 0.1))
        self.num_beams = config.get("num_beams", 1)
        self.repetition_penalty = config.get("repetition_penalty", 1.05)

        self.model = None
        self.tokenizer = None

    def _dtype(self):
        if self.torch_dtype in ("bfloat16", "bf16"):
            return torch.bfloat16
        if self.torch_dtype in ("float16", "fp16"):
            return torch.float16
        if self.torch_dtype in ("float32", "fp32"):
            return torch.float32
        return self.torch_dtype

    def _init_model(self):
        if self.model is not None:
            return

        from transformers import AutoModel, AutoTokenizer, PreTrainedModel

        print(f"Loading MiniCPM-V model from: {self.local_path}")
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            # MiniCPM-V 4.5 remote code was authored against Transformers 4.x.
            # Transformers 5.x computes this field differently during loading.
            PreTrainedModel.all_tied_weights_keys = {}
        model = AutoModel.from_pretrained(
            self.local_path,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=self._dtype(),
            low_cpu_mem_usage=True,
        ).eval()
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}

        if self.device == "cuda":
            model = model.cuda()
        elif self.device != "cpu":
            model = model.to(self.device)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_path, trust_remote_code=True
        )

    def extract_frames(self, video_path: str, query_time: float):
        """Extract frames and temporal IDs from [0, query_time]."""
        if self.use_temporal_ids:
            return self._extract_video_frames(video_path, query_time)

        from utils.frame_utils import extract_frames_for_query

        return extract_frames_for_query(
            video_path=video_path,
            query_time=query_time,
            max_frames=self.max_frames,
            fps=self.fps,
            frame_size=self.frame_size,
        )

    def _extract_video_frames(self, video_path: str, query_time: float) -> MiniCPMVFrameBatch:
        from utils.frame_utils import (
            _extract_frames_raw,
            _get_video_fps,
            _save_to_cache,
            _try_load_from_cache,
        )

        if not Path(video_path).exists():
            print(f"Warning: Video not found: {video_path}")
            return MiniCPMVFrameBatch([])

        video_fps = _get_video_fps(video_path)
        if video_fps <= 0:
            print(f"Warning: Invalid FPS for video: {video_path}")
            return MiniCPMVFrameBatch([])

        end_frame = max(int(query_time * video_fps), 1)
        video_duration = max(query_time, 1.0 / video_fps)
        choose_fps = min(float(self.fps), round(video_fps)) if video_fps > 0 else float(self.fps)
        max_num_frames = max(int(self.max_frames), 1)

        # Mirrors the official MiniCPM-V 4.5 dynamic packing policy, but clips
        # the video to the OVO-S query time so future frames are not visible.
        if choose_fps * int(video_duration) <= max_num_frames:
            packing_nums = 1
            choose_frames = round(choose_fps * min(max_num_frames, video_duration))
        else:
            packing_nums = math.ceil(video_duration * choose_fps / max_num_frames)
            if packing_nums <= self.max_packing:
                choose_frames = round(video_duration * choose_fps)
            else:
                choose_frames = round(max_num_frames * self.max_packing)
                packing_nums = self.max_packing

        choose_frames = max(1, min(int(choose_frames), end_frame + 1))
        target_frames = np.linspace(0, end_frame, choose_frames, dtype=int).tolist()
        target_frames = sorted(set(target_frames))

        cached = _try_load_from_cache(video_path, target_frames, self.frame_size)
        if cached is not None:
            frames = cached
        else:
            frames = _extract_frames_raw(video_path, target_frames, self.frame_size)
            if frames and len(frames) == len(target_frames):
                _save_to_cache(video_path, target_frames, frames, self.frame_size)

        if not frames:
            return MiniCPMVFrameBatch([])

        # `temporal_ids` groups adjacent frames that the MiniCPM-V 3D resampler
        # compresses together. IDs are 0.1s ticks, matching the official demo.
        frame_ts_ids = [int(round((idx / video_fps) / self.time_scale)) for idx in target_frames]
        temporal_ids = [
            frame_ts_ids[i:i + packing_nums]
            for i in range(0, len(frame_ts_ids), packing_nums)
        ]
        return MiniCPMVFrameBatch(frames=frames, temporal_ids=temporal_ids)

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        # Image-option path (task 4.3.x): MiniCPM-V's chat API accepts a plain
        # list of PIL images, so we unwrap the (optionally batched) frames and
        # append option PIL tail-images. Temporal IDs (if any) are dropped
        # since indices no longer align with the appended option frames.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames_list = getattr(frames, "frames", frames)
            frames = append_option_images_to_frames(
                frames_list, option_images, max_n=int(self.max_frames)
            )
        self._init_model()

        temporal_ids = getattr(frames, "temporal_ids", None)
        image_frames = getattr(frames, "frames", frames)
        msgs = [{"role": "user", "content": list(image_frames) + [prompt]}]

        chat_kwargs = {
            "msgs": msgs,
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_tokens,
            "sampling": self.sampling,
            "max_inp_length": self.max_inp_length,
            "enable_thinking": self.enable_thinking,
            "stream": False,
        }
        if temporal_ids:
            chat_kwargs.update(
                {
                    "use_image_id": self.use_image_id,
                    "max_slice_nums": self.max_slice_nums,
                    "temporal_ids": temporal_ids,
                }
            )
        if not self.sampling:
            chat_kwargs.update(
                {
                    "num_beams": self.num_beams,
                    "repetition_penalty": self.repetition_penalty,
                }
            )
        elif self.temperature > 0:
            chat_kwargs["temperature"] = self.temperature

        answer = self.model.chat(**chat_kwargs)
        if isinstance(answer, str):
            return answer.strip()
        return str(answer).strip()
