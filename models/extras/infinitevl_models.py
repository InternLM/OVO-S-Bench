"""
InfiniteVL streaming inference for OVO-S evaluation.

Uses InfiniteVL's hybrid linear attention for constant-memory video streaming.
For each video, frames are streamed once; at each query_time the cache is cloned
and a QA branch is performed without corrupting the main stream state.

Requires:
    - conda env: infinitevl
    - transformers, torch, flash-attn, flash-linear-attention, causal-conv1d
"""

import sys
import cv2
import torch
import time
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from ..base import BaseModel
from ._paths import find_upstream_src

# Add InfiniteVL source to path (lazy-resolved; see hermes_models.py).
_INFINITEVL_SRC = find_upstream_src("InfiniteVL", strict=False)
if _INFINITEVL_SRC not in sys.path:
    sys.path.insert(0, _INFINITEVL_SRC)

IMG_TOKENS_PER_FRAME = 256
FRAME_RESIZE = (448, 448)
DTYPE = torch.bfloat16


def _clone_inference_cache(model, src_cache):
    """Deep-copy the streaming cache state for branching QA."""
    dst_cache = model.allocate_inference_cache(batch_size=1)
    for dst_layer, src_layer in zip(dst_cache.layers, src_cache.layers):
        if getattr(src_layer, "is_sliding", False) and hasattr(src_layer, "_buf_keys"):
            dst_layer.size = src_layer.size
            dst_layer.cumulative_length = src_layer.cumulative_length
            if getattr(dst_layer, "capacity", 0) > 0 and src_layer.size > 0:
                L = src_layer.size
                dst_layer._buf_keys[:, :, :L, :].copy_(src_layer._buf_keys[:, :, :L, :])
                dst_layer._buf_values[:, :, :L, :].copy_(src_layer._buf_values[:, :, :L, :])
                dst_layer.keys = dst_layer._buf_keys[:, :, :L, :]
                dst_layer.values = dst_layer._buf_values[:, :, :L, :]
            elif getattr(dst_layer, "capacity", 0) > 0:
                dst_layer.keys = dst_layer._buf_keys[:, :, :0, :]
                dst_layer.values = dst_layer._buf_values[:, :, :0, :]
        elif hasattr(src_layer, "recurrent_state") and hasattr(src_layer, "conv_state_q"):
            if getattr(src_layer, "conv_state_q", None) is not None:
                dst_layer.conv_state_q.copy_(src_layer.conv_state_q)
                dst_layer.conv_state_k.copy_(src_layer.conv_state_k)
                dst_layer.conv_state_v.copy_(src_layer.conv_state_v)
            if getattr(src_layer, "recurrent_state", None) is not None:
                dst_layer.recurrent_state.copy_(src_layer.recurrent_state)
            dst_layer.seq_len = src_layer.seq_len
            dst_layer.start = src_layer.start
    return dst_cache


class InfiniteVLModel(BaseModel):
    """Streaming inference for InfiniteVL with per-video cache reuse.

    For each video, frames are streamed at `stream_fps` into a constant-memory
    cache. At each query_time the cache is cloned for QA without corrupting the
    main stream. Multiple queries on the same video share one streaming pass.
    """

    supports_video_streaming = True  # flag for inference.py dispatcher

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = config.get("local_path", config.get("model_id"))
        self.stream_fps = config.get("stream_fps", 1)

        self._model = None
        self._processor = None
        # Streaming template state (computed once after model load)
        self._first_frame_input_ids = None
        self._stream_frame_input_ids = None
        self._pos_base_full = None
        self._pos_base_stream = None
        self._grid_thw = None
        self._pixel_values_ref = None
        self._second_per_grid_ts = 1.0
        self._tokens_per_grid = 1

    def _init_model(self):
        """Lazy initialization of model, processor, and streaming templates."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading InfiniteVL from: {self.local_path}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.local_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            device_map=None,
        ).to("cuda").eval()

        self._processor = AutoProcessor.from_pretrained(
            self.local_path, trust_remote_code=True
        )
        self._setup_streaming_templates()
        self._warmup()
        print("InfiniteVL loaded (streaming mode).")

    # ------------------------------------------------------------------
    # Streaming infrastructure setup
    # ------------------------------------------------------------------
    def _build_image_inputs(self, image_pil: Image.Image):
        content = [{"type": "image", "image": image_pil}]
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return self._processor(text=[text], images=[image_pil], return_tensors="pt")

    def _build_text_query_inputs(self, question: str):
        content = [{"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._processor(text=[text], images=None, return_tensors="pt")

    def _setup_streaming_templates(self):
        """Compute reusable token templates and RoPE base from a dummy frame."""
        device = next(self._model.parameters()).device
        dummy = Image.new("RGB", FRAME_RESIZE, (128, 128, 128))
        warm = self._build_image_inputs(dummy)
        warm = {k: v.to(device) for k, v in warm.items()}

        full_ids = warm["input_ids"]
        self._pixel_values_ref = warm["pixel_values"]
        self._grid_thw = warm["image_grid_thw"]

        # Precompute vision module window buffers for streaming
        visual = getattr(self._model, "visual", None)
        if visual is not None and hasattr(visual, "set_graph_bucket"):
            visual.set_graph_bucket(self._grid_thw)
            if hasattr(visual, "precompute_window_buffers"):
                visual.precompute_window_buffers()
            if hasattr(visual, "precompute_full_cu_seqlens"):
                visual.precompute_full_cu_seqlens()

        vstart_id = self._model.config.vision_start_token_id
        vstart_pos = (full_ids[0] == vstart_id).nonzero(as_tuple=False)
        vstart_idx = vstart_pos[0].item()
        img_start = vstart_idx + 1
        image_span = full_ids[:, img_start: img_start + IMG_TOKENS_PER_FRAME]

        self._first_frame_input_ids = torch.cat(
            [full_ids[:, vstart_idx: vstart_idx + 1], image_span], dim=1
        ).to(device)
        self._stream_frame_input_ids = image_span.clone().to(device)

        pos_base_full, rope_deltas = self._model.model.get_rope_index(
            input_ids=self._first_frame_input_ids,
            image_grid_thw=self._grid_thw,
            video_grid_thw=None,
            attention_mask=None,
        )
        self._model.model.rope_deltas = rope_deltas
        self._pos_base_full = pos_base_full.to(device)
        self._pos_base_stream = self._pos_base_full[:, :, 1:].clone()

        self._second_per_grid_ts = float(
            getattr(self._model.config, "second_per_grid_ts", 1.0)
        )
        tokens_per_second = 1.0
        if hasattr(self._model.config, "vision_config"):
            tokens_per_second = float(
                getattr(self._model.config.vision_config, "tokens_per_second", 1.0)
            )
        self._tokens_per_grid = max(
            int(round(self._second_per_grid_ts * tokens_per_second)), 1
        )

    def _warmup(self):
        """Run dummy forward passes to compile kernels."""
        device = next(self._model.parameters()).device
        first_len = self._first_frame_input_ids.shape[1]
        stream_len = self._stream_frame_input_ids.shape[1]

        cache = self._model.allocate_inference_cache(batch_size=1)
        with torch.inference_mode():
            cp = torch.arange(0, first_len, dtype=torch.long, device=device).view(1, -1)
            self._model(
                input_ids=self._first_frame_input_ids,
                position_ids=self._pos_base_full,
                pixel_values=self._pixel_values_ref,
                image_grid_thw=self._grid_thw,
                use_cache=True,
                past_key_values=cache,
                cache_position=cp,
                return_dict=True,
            )
            cp2 = torch.arange(0, stream_len, dtype=torch.long, device=device)
            self._model(
                input_ids=self._stream_frame_input_ids,
                position_ids=self._pos_base_stream,
                pixel_values=torch.empty_like(self._pixel_values_ref),
                image_grid_thw=self._grid_thw,
                use_cache=True,
                past_key_values=cache,
                cache_position=cp2,
                return_dict=True,
            )
        del cache
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Core streaming logic
    # ------------------------------------------------------------------
    def _extract_video_frames(self, video_path: str):
        """Generator yielding (time_sec, PIL Image) at stream_fps."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS for video: {video_path}")
        frame_interval = max(1, round(video_fps / self.stream_fps))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                time_sec = frame_idx / video_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize(FRAME_RESIZE, Image.Resampling.BICUBIC)
                yield time_sec, pil
            frame_idx += 1
        cap.release()

    def _stream_one_frame(
        self, frame_pil: Image.Image, frame_time: float,
        stream_cache, cum_tokens: int, pos_max: torch.Tensor,
        is_first: bool,
    ) -> Tuple[int, torch.Tensor]:
        """Feed one frame into the streaming cache. Returns updated (cum_tokens, pos_max)."""
        device = next(self._model.parameters()).device
        batch = self._build_image_inputs(frame_pil)
        pixel_values = batch["pixel_values"].to(device)

        if is_first:
            input_ids = self._first_frame_input_ids
            pos_ids = self._pos_base_full.clone()
            frame_len = input_ids.shape[1]
        else:
            input_ids = self._stream_frame_input_ids
            pos_ids = self._pos_base_stream.clone()
            frame_len = input_ids.shape[1]
            # Apply time-aware position offset
            grid_t = int(frame_time / self._second_per_grid_ts)
            t_offset = grid_t * self._tokens_per_grid
            if t_offset != 0:
                p0 = pos_ids.view(3, -1)[0]
                inc = torch.full(
                    (IMG_TOKENS_PER_FRAME,), t_offset,
                    device=device, dtype=p0.dtype,
                )
                p0.index_add_(
                    0, torch.arange(IMG_TOKENS_PER_FRAME, device=device), inc
                )

        pos_max = torch.maximum(pos_max, pos_ids.max().to(torch.long))
        cache_pos = torch.arange(
            cum_tokens, cum_tokens + frame_len, dtype=torch.long, device=device
        ).view(1, -1) if is_first else torch.arange(
            cum_tokens, cum_tokens + frame_len, dtype=torch.long, device=device
        )

        self._model(
            input_ids=input_ids,
            position_ids=pos_ids,
            pixel_values=pixel_values,
            image_grid_thw=self._grid_thw,
            use_cache=True,
            past_key_values=stream_cache,
            cache_position=cache_pos,
            return_dict=True,
        )
        return cum_tokens + frame_len, pos_max

    def _qa_from_cache(
        self, prompt: str, stream_cache, cum_tokens: int, pos_max: torch.Tensor,
    ) -> str:
        """Clone cache, run QA, return answer string."""
        device = next(self._model.parameters()).device
        qa_cache = _clone_inference_cache(self._model, stream_cache)
        qa_cum = cum_tokens
        qa_pos_max = pos_max.clone()

        # Prepare question tokens
        q_batch = self._build_text_query_inputs(prompt)
        q_input_ids = q_batch["input_ids"].to(device)
        vend_token = torch.full(
            (1, 1), self._model.config.vision_end_token_id,
            dtype=q_input_ids.dtype, device=device,
        )
        q_ids = torch.cat([vend_token, q_input_ids], dim=1)
        q_len = q_ids.shape[1]

        # Position IDs for question
        q_cache_pos = torch.arange(
            qa_cum, qa_cum + q_len, dtype=torch.long, device=device
        ).view(1, -1)
        start_pos = qa_pos_max + 1
        q_pos_1d = start_pos + torch.arange(q_len, device=device, dtype=torch.long)
        q_pos_ids = q_pos_1d.to(self._pos_base_full.dtype).view(1, 1, -1).expand(3, 1, -1)
        qa_pos_max = qa_pos_max + q_len

        # Question prefill
        q_out = self._model(
            input_ids=q_ids,
            past_key_values=qa_cache,
            use_cache=True,
            cache_position=q_cache_pos,
            position_ids=q_pos_ids,
            return_dict=True,
        )
        qa_cum += q_len

        # Greedy decode
        next_token = torch.argmax(q_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = []
        eos_id = self._processor.tokenizer.eos_token_id
        for _ in range(self.max_tokens):
            if next_token.item() == eos_id:
                break
            generated.append(next_token)
            step_cache_pos = torch.tensor([qa_cum], dtype=torch.long, device=device)
            qa_pos_max = qa_pos_max + 1
            step_pos = torch.full(
                (3, 1, 1), qa_pos_max.item(),
                dtype=self._pos_base_full.dtype, device=device,
            )
            step_out = self._model(
                input_ids=next_token,
                past_key_values=qa_cache,
                use_cache=True,
                cache_position=step_cache_pos,
                position_ids=step_pos,
                return_dict=True,
            )
            qa_cum += 1
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)

        del qa_cache
        if generated:
            ids = torch.cat(generated, dim=1)[0]
            return self._processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stream_video_inference(
        self,
        video_path: str,
        queries: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Stream a video once and answer all queries (sorted by query_time).

        Args:
            video_path: Path to the video file.
            queries: List of query dicts, each must have 'query_time' and a
                     pre-built 'prompt' key. Must be sorted by query_time.

        Returns:
            List of (query, response_str) in the same order as input queries.
        """
        self._init_model()
        device = next(self._model.parameters()).device

        stream_cache = self._model.allocate_inference_cache(batch_size=1)
        cum_tokens = 0
        pos_max = self._pos_base_full.max().to(torch.long)

        results: List[Tuple[Dict[str, Any], str]] = []
        qi = 0  # index into queries
        is_first = True
        frames_fed = 0

        t0 = time.time()
        with torch.inference_mode():
            for frame_time, frame_pil in self._extract_video_frames(video_path):
                # Answer all queries whose query_time <= current frame_time
                while qi < len(queries) and queries[qi]["query_time"] <= frame_time:
                    resp = self._qa_from_cache(
                        queries[qi]["prompt"], stream_cache, cum_tokens, pos_max,
                    )
                    results.append((queries[qi], resp))
                    qi += 1

                if qi >= len(queries):
                    break  # all queries answered

                # Stream this frame
                cum_tokens, pos_max = self._stream_one_frame(
                    frame_pil, frame_time, stream_cache,
                    cum_tokens, pos_max, is_first,
                )
                is_first = False
                frames_fed += 1

            # Handle remaining queries (query_time beyond video end)
            while qi < len(queries):
                resp = self._qa_from_cache(
                    queries[qi]["prompt"], stream_cache, cum_tokens, pos_max,
                )
                results.append((queries[qi], resp))
                qi += 1

        del stream_cache
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(
            f"  InfiniteVL streamed {frames_fed} frames, "
            f"answered {len(results)} queries in {elapsed:.1f}s"
        )
        return results

    def inference(self, frames: List[Image.Image], prompt: str) -> str:
        """Fallback: single-query prefill (used if framework doesn't group by video)."""
        self._init_model()
        device = next(self._model.parameters()).device

        stream_cache = self._model.allocate_inference_cache(batch_size=1)
        cum_tokens = 0
        pos_max = self._pos_base_full.max().to(torch.long)

        with torch.inference_mode():
            for i, frame in enumerate(frames):
                frame_resized = frame.resize(FRAME_RESIZE, Image.Resampling.BICUBIC)
                cum_tokens, pos_max = self._stream_one_frame(
                    frame_resized, float(i) / max(self.stream_fps, 1),
                    stream_cache, cum_tokens, pos_max, is_first=(i == 0),
                )
            resp = self._qa_from_cache(prompt, stream_cache, cum_tokens, pos_max)

        del stream_cache
        torch.cuda.empty_cache()
        return resp
