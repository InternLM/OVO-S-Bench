"""
FluxMem streaming inference for OVO-S evaluation.

Uses FluxMem's three-tier visual token memory management on top of a custom
Qwen2.5-VL model. Short-term tokens are kept at full resolution, mid-term
tokens are pruned via similarity-based dropping, and long-term tokens are
compressed via spatial clustering.

For each query, the video is processed from the start up to query_time with
FluxMem-enabled generation. This is not natively incremental — each query
re-processes the video prefix.

Requires:
    - conda env: fluxmem
    - transformers, torch, flash-attn, moviepy, decord
"""

import os
import sys
import torch
import time
from typing import Dict, List, Any, Tuple
from PIL import Image

from ..base import BaseModel, resolve_runtime_path
from ._paths import find_upstream_src

# Source paths for FluxMem (resolved lazily; see hermes_models.py).
_FLUXMEM_SRC = find_upstream_src("FluxMem", strict=False)
_FLUXMEM_MODEL_SRC = os.path.join(_FLUXMEM_SRC, "models", "qwen2-5-vl", "src")
_FLUXMEM_UTILS_SRC = os.path.join(_FLUXMEM_SRC, "models", "qwen-vl-utils", "src")

# Default vision resolution constraints
MIN_PIXELS = 16 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
MIN_FRAMES = 4
MAX_FRAMES = 256


def _format_official_mc_prompt(question: str, options: Dict[str, str]) -> str:
    """Match FluxMem's OVO-Bench multiple-choice prompt format."""
    formatted_options = "; ".join(
        f"{letter}. {text}" for letter, text in sorted(options.items())
    ) + ";"
    return f"""
            Question: {question}
            Options:
            {formatted_options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
            Do not include any additional text or explanation in your response.
        """


class FluxMemModel(BaseModel):
    """FluxMem inference with three-tier visual token memory.

    Two modes:
      * Streaming (default): for each query the video is clipped to
        [0, query_time] and processed via stream_video_inference().
      * Frame-input (set ``use_streaming: false`` in config): pre-extracted
        frames are passed via inference(frames, prompt). Used for the
        uniform-sample ablation (nframes=128/256) where the project's
        frame cache is the source of truth.
    """

    supports_video_streaming = True

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(config.get("local_path", config.get("model_id")))
        self.short_frames = config.get("short_frames", 8)
        self.medium_frames = config.get("medium_frames", 16)
        self.sample_fps = config.get("sample_fps", 1.0)
        self.max_pixels = config.get("max_pixels", MAX_PIXELS)
        self.min_pixels = config.get("min_pixels", MIN_PIXELS)
        self.max_num_frames = config.get(
            "max_num_frames", config.get("max_frames", MAX_FRAMES)
        )
        self.min_frames = config.get("min_frames", MIN_FRAMES)
        self.frame_sampling = config.get("frame_sampling", "uniform")
        self.pair_sim_threshold = config.get("pair_sim_threshold", None)
        self.save_path = config.get("save_path", None)
        self.time_window_size = config.get("time_window_size", None)
        self.anchor_end = config.get("anchor_end", False)
        self.nframes = config.get("nframes", config.get("max_frames", MAX_FRAMES))

        # Allow disabling streaming so the project's pre-cached uniform
        # frames are used via inference(frames, prompt) instead.
        self.supports_video_streaming = config.get("use_streaming", True)

        self._model = None
        self._processor = None
        self._process_vision_info = None

    def build_prompt(self, question: str, options: Dict[str, str],
                     prompt_style: str = None) -> str:
        """Use the official FluxMem OVO-Bench prompt by default.

        Passing a prompt_style keeps the repository-level prompt override
        behavior for ablations.
        """
        if prompt_style:
            return super().build_prompt(question, options, prompt_style)
        if options:
            return _format_official_mc_prompt(question, options)
        return question

    def _init_model(self):
        """Lazy initialization of FluxMem model and processor."""
        if self._model is not None:
            return

        # Add FluxMem source directories to path
        for src_path in [_FLUXMEM_MODEL_SRC, _FLUXMEM_UTILS_SRC]:
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

        from qwen2_5_vl_fluxmem import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )
        from qwen_vl_utils_fluxmem import process_vision_info

        print(f"Loading FluxMem (Qwen2.5-VL) from: {self.local_path}")
        print(f"  short_frames={self.short_frames}, medium_frames={self.medium_frames}")

        torch.manual_seed(1234)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.local_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            self.local_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self._process_vision_info = process_vision_info
        print("FluxMem loaded.")

    def _build_messages(self, video_path: str, prompt: str,
                        video_end: float = None,
                        sample_id: str = None) -> list:
        """Build Qwen2.5-VL chat messages with video content."""
        video_start = 0.0
        anchor_end = self.anchor_end
        if (
            video_end is not None
            and self.time_window_size is not None
            and self.time_window_size > 0
        ):
            video_start = max(0.0, video_end - float(self.time_window_size))
            anchor_end = True

        video_content = {
            "type": "video",
            "video": video_path,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "min_frames": self.min_frames,
            "max_frames": self.max_num_frames,
            "fps": self.sample_fps,
            "anchor_end": anchor_end,
            "sample_id": sample_id,
        }
        if video_end is not None:
            video_content["video_start"] = video_start
            video_content["video_end"] = video_end

        return [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _generate_response(self, messages: list) -> str:
        """Run FluxMem-enabled generation on prepared messages."""
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self._model.device)

        generate_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=False,
            temperature=0.0,
            use_fluxmem=True,
            memory_drop_method=self.frame_sampling,
            short_frames=self.short_frames,
            medium_frames=self.medium_frames,
        )
        if self.pair_sim_threshold is not None:
            generate_kwargs["pair_sim_threshold"] = self.pair_sim_threshold
        if self.save_path is not None:
            generate_kwargs["save_path"] = self.save_path

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **generate_kwargs)

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def stream_video_inference(
        self,
        video_path: str,
        queries: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Process all queries for a video.

        For each query, the video is clipped to [0, query_time] and processed
        with FluxMem-enabled generation.

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
                messages = self._build_messages(
                    video_path,
                    q["prompt"],
                    video_end=q["query_time"],
                    sample_id=q.get("query_id"),
                )
                resp = self._generate_response(messages)
            except Exception as e:
                print(f"  FluxMem error on query {q.get('query_id', '?')}: {e}")
                resp = ""
            results.append((q, resp))
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(
            f"  FluxMem processed {len(results)} queries in {elapsed:.1f}s"
        )
        return results

    def inference(self, frames: List[Any], prompt: str,
                  option_images: List[Any] = None) -> str:
        """Frame-input mode: run FluxMem on a pre-extracted frame list.

        Used when ``use_streaming: false`` is set in config — the wrapper
        becomes a single-shot model that consumes the project's cached
        uniform-sampled frames (e.g. via extract_frames_fixed_count).

        Args:
            frames: list of PIL.Image frames (already at frame_size).
            prompt: full multiple-choice prompt string.
            option_images: optional [(label, PIL.Image), ...] for task 4.3.x;
                appended to the tail of frames so the model sees them as the
                last N "video" frames (prompt instruction explains the layout).

        Returns:
            Decoded model response.
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

        # Build a Qwen2.5-VL message with the frame list as the video input.
        # The fluxmem fork's process_vision_info accepts a list of PIL Images
        # under the "video" key, treating each entry as one sampled frame.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "fps": self.sample_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            return self._generate_response(messages)
        except Exception as e:
            print(f"FluxMem frame-input inference error: {e}")
            return ""
