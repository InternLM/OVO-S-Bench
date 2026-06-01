"""
HERMES streaming inference for OVO-S evaluation.

Uses HERMES's hierarchical KV cache management with attention-based pruning
for constant-memory video streaming on top of Qwen2.5-VL.

For each video, frames are streamed in chunks; after each chunk the cache is
compressed via predict_and_compress(). At each query_time, question_answering()
generates an answer using the current compressed cache state.

Requires:
    - conda env: hermes
    - transformers, torch, flash-attn
"""

import os
import sys
import cv2
import math
import torch
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from PIL import Image

from ..base import BaseModel, resolve_runtime_path
from ._paths import find_upstream_src

# Source path for HERMES. Resolved lazily (strict=False) so a missing upstream
# checkout doesn't break import-time discovery in vllm_models._get_offline_registry.
_HERMES_SRC = find_upstream_src("HERMES", strict=False)


def _hermes_get_prompt(query: str, mc: bool = False) -> str:
    """Match HERMES Abstract_Hermes.get_prompt exactly."""
    prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
    if mc:
        prompt += "Best option: ("
    return prompt


class HermesModel(BaseModel):
    """Streaming inference for HERMES with hierarchical KV cache compression.

    For each video, frames are streamed at `sample_fps` into the model.
    After each chunk of frames, predict_and_compress() prunes the KV cache.
    At each query_time, question_answering() generates an answer from the
    compressed cache, then the cache is restored for continued streaming.
    """

    supports_video_streaming = True

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(config.get("local_path", config.get("model_id")))
        self.kv_size = config.get("kv_size", 6000)
        self.sample_fps = config.get("sample_fps", 1.0)
        self.encode_chunk_size = config.get("encode_chunk_size", 16)
        self.streaming = config.get("streaming", True)
        self.mc_max_new_tokens = config.get("mc_max_new_tokens", 16)
        self.open_max_new_tokens = config.get("open_max_new_tokens", 256)
        self.nframes = config.get("nframes", config.get("max_frames", 128))

        # Allow disabling video-streaming so the project's pre-cached uniform
        # frames are consumed via inference(frames, prompt) instead.
        self.supports_video_streaming = config.get("use_streaming", True)

        self._model = None
        self._processor = None

    def build_prompt(self, question: str, options: Dict[str, str],
                     prompt_style: str = None) -> Dict[str, Any]:
        """Build the official HERMES QA input dict.

        The HERMES repo does not pass the benchmark's generic prompt directly.
        It wraps multiple-choice questions as "(A) option" blocks and appends
        "Best option: (" to the assistant prefix.
        """
        if prompt_style:
            question = super().build_prompt(question, options, prompt_style)
            return {
                "question": question,
                "prompt": _hermes_get_prompt(question),
                "max_new_tokens": self.open_max_new_tokens,
            }

        if options:
            formatted_choices = "\n".join(
                f"({letter}) {text}" for letter, text in sorted(options.items())
            )
            formatted_question = (
                f"Question: {question}\n"
                f"Options:\n{formatted_choices}\n"
                "Only give the best option."
            )
            return {
                "question": question,
                "formatted_question": formatted_question,
                "prompt": _hermes_get_prompt(formatted_question, mc=True),
                "max_new_tokens": self.mc_max_new_tokens,
            }

        return {
            "question": question,
            "prompt": _hermes_get_prompt(question),
            "max_new_tokens": self.open_max_new_tokens,
        }

    def _init_model(self):
        """Lazy initialization of HERMES model and processor."""
        if self._model is not None:
            return

        # Resolve HERMES's `inference.qwenvl_hermes` import. Two things can
        # shadow HERMES's `inference` namespace package:
        #   (1) the project's top-level `inference.py` (already in sys.modules)
        #   (2) the project root absolute path in sys.path (inserted by
        #       inference.py at startup), which has higher priority than the
        #       cwd-relative HERMES_SRC.
        # Drop both temporarily, prepend HERMES_SRC, chdir into it, import.
        import importlib
        saved_inference_modules = {
            name: sys.modules[name]
            for name in list(sys.modules)
            if name == "inference" or name.startswith("inference.")
        }
        for mod_name in [m for m in list(sys.modules)
                         if m == "inference" or m.startswith("inference.")]:
            del sys.modules[mod_name]
        importlib.invalidate_caches()

        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _saved_path = list(sys.path)
        sys.path = [p for p in sys.path
                    if os.path.abspath(p) != _project_root and p not in ("", ".")]
        sys.path.insert(0, _HERMES_SRC)

        _orig_cwd = os.getcwd()
        os.chdir(_HERMES_SRC)
        try:
            from inference.qwenvl_hermes import load_model
        finally:
            os.chdir(_orig_cwd)
            sys.path = _saved_path
            # HERMES's upstream package is also named `inference`.  Leaving it
            # in sys.modules breaks the persistent suite when the next item
            # imports this project's top-level inference.py again.
            for mod_name in [m for m in list(sys.modules)
                             if m == "inference" or m.startswith("inference.")]:
                if mod_name not in saved_inference_modules:
                    del sys.modules[mod_name]
            sys.modules.update(saved_inference_modules)
            importlib.invalidate_caches()

        print(f"Loading HERMES (Qwen2.5-VL) from: {self.local_path}")
        print(f"  kv_size={self.kv_size}, sample_fps={self.sample_fps}, "
              f"streaming={self.streaming}")

        self._model, self._processor = load_model(
            model_path=self.local_path,
            kv_size=self.kv_size,
            streaming=self.streaming,
            sample_fps=self.sample_fps,
        )
        print("HERMES loaded (streaming mode).")

    def _load_video_official(self, video_path: str) -> np.ndarray:
        """Load/samples frames like HERMES video_qa.base.BaseVQA.load_video."""
        if video_path.endswith(".npy"):
            video = np.load(video_path)
            num_frames = len(video)
            frame_idx = np.linspace(
                0, num_frames - 1, int(num_frames * self.sample_fps), dtype=int
            ).tolist()
            return video[frame_idx]

        try:
            from decord import VideoReader

            vr = VideoReader(video_path, num_threads=1)
            fps = round(vr.get_avg_fps())
            total_frames = len(vr)
            sample_step = int(fps / self.sample_fps)
            if sample_step <= 0:
                raise ValueError(
                    f"Invalid sample step {sample_step} for fps={fps}, "
                    f"sample_fps={self.sample_fps}"
                )
            frame_idx = [i for i in range(0, total_frames, sample_step)]
            return vr.get_batch(frame_idx).asnumpy()
        except ImportError:
            print("Warning: decord is unavailable; falling back to cv2 sampling.")
            frames = [frame for _, frame in self._extract_video_frames(video_path)]
            if frames:
                return np.stack(frames, axis=0)
            return np.empty((0, 0, 0, 3), dtype=np.uint8)

    def _extract_video_frames(self, video_path: str, max_time: float = float("inf")):
        """Generator yielding (time_sec, numpy_frame_HWC) at sample_fps."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS for video: {video_path}")
        frame_interval = max(1, round(video_fps / self.sample_fps))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                time_sec = frame_idx / video_fps
                if time_sec > max_time:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield time_sec, rgb
            frame_idx += 1
        cap.release()

    def _format_question(self, prompt: Any) -> Dict[str, Any]:
        """Format prompt into the dict expected by HERMES question_answering()."""
        if isinstance(prompt, dict):
            return prompt
        return {
            "question": prompt,
            "prompt": _hermes_get_prompt(prompt),
            "max_new_tokens": self.open_max_new_tokens,
        }

    @staticmethod
    def _answer_max_new_tokens(input_text: Dict[str, Any], default_tokens: int) -> int:
        return int(input_text.get("max_new_tokens", default_tokens))

    def stream_video_inference(
        self,
        video_path: str,
        queries: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Stream a video once and answer all queries at their query_times.

        Args:
            video_path: Path to the video file.
            queries: List of query dicts sorted by query_time. Each must have
                     'query_time' and 'prompt' keys.

        Returns:
            List of (query, response_str) in the same order as input queries.
        """
        self._init_model()

        results: List[Tuple[Dict[str, Any], str]] = []
        frames_fed = 0

        # Clear cache and initialize for new video
        self._model.clear_cache()
        self._model.encode_init_prompt()

        t0 = time.time()
        video_tensor = torch.from_numpy(self._load_video_official(video_path))
        current_frame_idx = 0

        with torch.inference_mode():
            for q in queries:
                # HERMES official streaming eval encodes sampled frame indices
                # up to ceil(end_time * sample_fps), not by original frame time.
                end_frame_idx = math.ceil(q["query_time"] * self.sample_fps)
                end_frame_idx = min(end_frame_idx, len(video_tensor))

                while current_frame_idx < end_frame_idx:
                    next_encode_end = min(
                        current_frame_idx + self.encode_chunk_size,
                        end_frame_idx,
                    )
                    if next_encode_end > current_frame_idx:
                        print(
                            f"Encoding frames {current_frame_idx} "
                            f"to {next_encode_end - 1}"
                        )
                        video_chunk = video_tensor[current_frame_idx:next_encode_end]
                        self._model.encode_video_chunk(video_chunk)
                        current_frame_idx = next_encode_end
                        frames_fed += int(video_chunk.shape[0])

                        self._model.predict_and_compress()

                input_text = self._format_question(q["prompt"])
                resp = self._model.question_answering(
                    input_text,
                    max_new_tokens=self._answer_max_new_tokens(input_text, self.max_tokens),
                    temperature=self.temperature,
                )
                results.append((q, resp.replace("\n", "")))

        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(
            f"  HERMES streamed {frames_fed} frames, "
            f"answered {len(results)} queries in {elapsed:.1f}s"
        )
        return results

    def inference(self, frames: List[Any], prompt: Any,
                  option_images: List[Any] = None) -> str:
        """Fallback: single-query inference (used if not grouped by video)."""
        self._init_model()

        # Image-option path (task 4.3.x): append option PIL images at the
        # tail of the frame list so they get encoded as the last frames of
        # the video stream (prompt instruction explains the layout).
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )

        self._model.clear_cache()
        self._model.encode_init_prompt()

        with torch.inference_mode():
            # Convert frames to numpy array
            frame_arrays = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_arrays.append(np.array(frame))
                elif isinstance(frame, np.ndarray):
                    frame_arrays.append(frame)
                else:
                    frame_arrays.append(np.array(frame))

            if frame_arrays:
                # Encode all frames in chunks
                for i in range(0, len(frame_arrays), self.encode_chunk_size):
                    chunk = frame_arrays[i:i + self.encode_chunk_size]
                    chunk_tensor = torch.from_numpy(np.stack(chunk, axis=0))
                    self._model.encode_video_chunk(chunk_tensor)
                    self._model.predict_and_compress()

            input_text = self._format_question(prompt)
            if option_images:
                # Image-option MC (task 4.3.x): force mc=True so the assistant
                # prefix is "Best option: (" — without it the model often emits
                # EOS at step 0 on long image-option prompts, yielding empty
                # responses. Use mc_max_new_tokens (short budget) to match the
                # MC text path.
                input_text = {
                    "question": prompt,
                    "prompt": _hermes_get_prompt(prompt, mc=True),
                    "max_new_tokens": self.mc_max_new_tokens,
                }
            resp = self._model.question_answering(
                input_text,
                max_new_tokens=self._answer_max_new_tokens(input_text, self.max_tokens),
                temperature=self.temperature,
            )

        torch.cuda.empty_cache()
        return resp.replace("\n", "")
