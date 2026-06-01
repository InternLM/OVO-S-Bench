"""
InfiniPot-V inference for OVO-S evaluation.

This wrapper follows the official InfiniPot-V Qwen2/Qwen2.5-VL path: the
upstream ``OfflineVideoEval`` prepares Qwen video inputs, uses block-wise
prefill only when the sampled video has more frames than ``block_size``, and
runs the official KV-cache compression implementation.

Requires:
    - conda env: infinipot
    - transformers, torch, flash-attn, decord
"""

import os
import sys
import torch
import time
import hashlib
import numpy as np
from typing import Dict, List, Any, Tuple
from PIL import Image
from pathlib import Path

from ..base import BaseModel, resolve_runtime_path
from ._paths import find_upstream_src

# Source path for InfiniPot-V (resolved lazily; see hermes_models.py).
_INFINIPOT_SRC = find_upstream_src("InfiniPot-V", strict=False)


def infinipot_dump_path(video_path: str, dump_dir: str, max_frames_num: int, max_pixels: int) -> str:
    """Return a stable cache path for InfiniPot preprocessed video inputs."""
    dump_root = Path(resolve_runtime_path(dump_dir) or dump_dir)
    if not dump_root.is_absolute():
        dump_root = Path.cwd() / dump_root
    key_src = f"{video_path}|max_frames={max_frames_num}|max_pixels={max_pixels}"
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    return str(dump_root / key[:2] / f"{key}_mf{max_frames_num}_px{max_pixels}.pt")


class InfiniPotModel(BaseModel):
    """Block-wise inference for InfiniPot-V with KV cache compression.

    For each video+query, the upstream evaluator samples frames from the video
    and processes them in blocks of `block_size` frames. After each block (except the last), the
    KV cache is compressed to `compress_frame_num` frames using the chosen
    compression method. This allows processing arbitrarily long videos with
    bounded memory.

    Note: InfiniPot-V's official code is an offline per-sample evaluator, not
    an incremental multi-query streamer.  OVO-S groups queries by video for
    dispatch, but each query is run independently through the official path.
    """

    supports_video_streaming = True

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(config.get("local_path", config.get("model_id")))
        self.block_size = config.get("block_size", 32)
        self.compress_frame_num = config.get("compress_frame_num", 24)
        self.compression_method = config.get("compression_method", "infinipot-v")
        self.tar_ratio = config.get("tar_ratio", 0.5)
        self.query_ratio = config.get("query_ratio", 0.25)
        self.adaptive_pooling = config.get("adaptive_pooling", False)
        self.max_frames_num = config.get("max_frames_num", 768)
        self.sample_fps = config.get("sample_fps", 1.0)
        self.load_dumped = config.get("load_dumped", False)
        self.dump_dir = config.get("dump_dir")

        # Allow disabling the upstream streaming path so the project's
        # pre-cached uniform frames are consumed via inference(frames, prompt).
        # Used by the infinipot-n128 ablation and the 4.3 image-option path
        # (upstream prepare_video_input can't carry extra option images).
        self.supports_video_streaming = config.get("use_streaming", True)

        self._evaluator = None

    def _init_model(self):
        """Lazy initialization of InfiniPot-V evaluator."""
        if self._evaluator is not None:
            return

        # Add InfiniPot-V source to path
        if _INFINIPOT_SRC not in sys.path:
            sys.path.insert(0, _INFINIPOT_SRC)

        import qwen_inference_ovu as infinipot_ovu
        from qwen_inference_ovu import OfflineVideoEval

        # The upstream file references MAX_GEN_TOKENS as a module global in
        # generate(); expose the intended class constant for short videos that
        # fall back to standard generation.
        if not hasattr(infinipot_ovu, "MAX_GEN_TOKENS"):
            infinipot_ovu.MAX_GEN_TOKENS = OfflineVideoEval.MAX_GEN_TOKENS

        print(f"Loading InfiniPot-V from: {self.local_path}")
        print(f"  block_size={self.block_size}, compress_frame_num={self.compress_frame_num}, "
              f"method={self.compression_method}")

        self._evaluator = OfflineVideoEval(
            model_path=self.local_path,
            max_frames_num=self.max_frames_num,
            block_size=self.block_size,
            compress_frame_num=self.compress_frame_num,
            compression_method=self.compression_method,
            tar_ratio=self.tar_ratio,
            query_ratio=self.query_ratio,
            adaptive_pooling=self.adaptive_pooling,
            load_dumped=self.load_dumped,
            per_frame=False,
            verbose=False,
        )
        if self.dump_dir:
            max_pixels = int(getattr(OfflineVideoEval, "DEFAULT_MAX_PIXELS", 128 * 28 * 28))

            def _cached_dump_path(video_path: str) -> str:
                return infinipot_dump_path(
                    video_path,
                    self.dump_dir,
                    int(self.max_frames_num),
                    max_pixels,
                )

            self._evaluator._get_dump_path = _cached_dump_path
        print("InfiniPot-V loaded.")

    def _process_single_query(self, video_path: str, prompt: str, query_time: float) -> str:
        """Process one query through the official OfflineVideoEval path."""
        # Official InfiniPot-V samples from the provided video path; it does
        # not expose a query_time crop parameter.
        inputs = self._evaluator.prepare_video_input(
            video_path=video_path,
            question_text=prompt,
        )

        # Match the official evaluator: block processing is used only when the
        # sampled frame count exceeds block_size; shorter videos use the model's
        # standard generate path.
        frame_count = 0
        if inputs.get("video_grid_thw") is not None:
            frame_count = int(inputs["video_grid_thw"][0, 0].item())
        use_block = (
            self.block_size > 0
            and frame_count > self.block_size
            and self.compress_frame_num > 0
            and inputs.get("pixel_values_videos") is not None
        )

        if use_block:
            response = self._evaluator.block_process(inputs)
        else:
            response = self._evaluator.generate(inputs)

        return response

    def stream_video_inference(
        self,
        video_path: str,
        queries: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Process all queries for a video.

        InfiniPot-V is not natively incremental. Each query re-processes the
        video independently, matching the official per-sample evaluator.

        Args:
            video_path: Path to the video file.
            queries: List of query dicts sorted by query_time.

        Returns:
            List of (query, response_str) pairs.
        """
        self._init_model()

        results: List[Tuple[Dict[str, Any], str]] = []
        t0 = time.time()

        for q in queries:
            try:
                resp = self._process_single_query(
                    video_path, q["prompt"], q["query_time"]
                )
            except Exception as e:
                print(f"  InfiniPot-V error on query {q.get('query_id', '?')}: {e}")
                resp = ""
            results.append((q, resp))

        elapsed = time.time() - t0
        print(
            f"  InfiniPot-V processed {len(results)} queries in {elapsed:.1f}s"
        )
        return results

    def inference(self, frames: List[Any], prompt: str,
                  option_images: List[Any] = None) -> str:
        """Frame-input mode: run stock Qwen2.5-VL.generate on a pre-extracted
        frame list. Used when ``use_streaming: false`` (e.g. infinipot-n128 or
        the 4.3 image-option path).

        Note: this path bypasses InfiniPot's KV-cache compression. The
        compression mechanism targets long video streaming; for single-query
        evaluation with pre-cached uniform frames (and optional image-option
        tail), running stock Qwen2.5-VL.generate is the correct fallback.

        Args:
            frames: pre-extracted PIL frames (already at frame_size).
            prompt: full multiple-choice prompt string.
            option_images: optional [(label, PIL.Image), ...] for task 4.3.x;
                appended to the tail of frames so the model sees them as the
                last N "video" frames (prompt instruction explains the layout).
        """
        self._init_model()
        if not frames:
            return ""

        # Image-option path (task 4.3.x): pack option PIL images at the tail of
        # the frame list so they ride along inside the single "video" block.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )

        # Use the upstream evaluator's loaded model+processor and run a
        # plain Qwen2.5-VL multi-modal generate. process_vision_info accepts
        # a list of PIL.Image under the "video" key (treats each as a frame).
        import qwen_vl_utils as _qvu
        process_vision_info = _qvu.process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "fps": self.sample_fps,
                },
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._evaluator.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._evaluator.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._evaluator.model.device)

        with torch.inference_mode():
            gen_ids = self._evaluator.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
        response = self._evaluator.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response.strip()
