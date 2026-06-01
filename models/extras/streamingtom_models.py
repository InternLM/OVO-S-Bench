"""
StreamingTOM inference for OVO-S evaluation.

This wrapper follows the official StreamingTOM/LLaVA-OneVision code path:
LLaVA-NeXT loads ``llava-onevision-qwen2-7b-ov``, StreamingTOM patches the
model's ``generate`` method, frames are preprocessed by LLaVA's own image
processor, prompts are wrapped with the LLaVA ``qwen_1_5`` conversation template,
and generation is invoked through the patched ``model.generate``.

Requires:
    - conda env: streamingtom
    - transformers, torch, flash-attn, llava (LLaVA-OneVision)
"""

import os
import sys
import cv2
import torch
import time
from typing import Dict, List, Any, Tuple
from PIL import Image

from ..base import BaseModel, resolve_runtime_path
from ._paths import find_upstream_src

# Source paths for StreamingTOM and its vendored LLaVA-NeXT checkout
# (resolved lazily; see hermes_models.py).
_STREAMINGTOM_SRC = find_upstream_src("StreamingTOM", strict=False)
_LLAVA_NEXT_SRC = os.path.join(_STREAMINGTOM_SRC, "LLaVA-NeXT")


class StreamingTOMModel(BaseModel):
    """Official StreamingTOM wrapper for LLaVA-OneVision.

    The official repository exposes StreamingTOM by monkey-patching
    LLaVA-OneVision's ``generate``.  To keep behavior aligned, this class does
    not manually call the lower-level pipeline for normal inference; it prepares
    inputs the same way as the official lmms-eval adapter and then calls the
    patched ``generate``.
    """

    # The official LLaVA-OneVision evaluation is per sample.  We still implement
    # stream_video_inference so OVO-S can group by video, but each query is run
    # through the official patched generate path independently.
    supports_video_streaming = True

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(
            config.get("local_path", config.get("model_id"))
        )
        self.nframes = int(config.get("nframes", self.max_frames))
        # Default to the official video-grouped path, but allow n128 cached
        # evaluation to use the common fixed-frame pipeline to avoid long CPU
        # OpenCV decode periods that leave the GPU idle.
        self.supports_video_streaming = bool(config.get("use_streaming", True))
        # When false, load the model via the same vendored LLaVA-NeXT path but
        # skip the streamingtom KV-compression patch — used by §4.3.4(a) to
        # evaluate the vanilla LLaVA-Video / LLaVA-OneVision baseline.
        self.use_streamingtom_patch = bool(config.get("use_streamingtom_patch", True))
        self.stream_fps = config.get("stream_fps", 1.0)
        self.encoder_batch_size = config.get("encoder_batch_size", 32)
        self.conv_template = config.get("conv_template", "qwen_1_5")
        self.model_name_for_loader = config.get("loader_model_name", "llava_qwen")
        self.mm_spatial_pool_stride = config.get("mm_spatial_pool_stride", 2)
        self.mm_spatial_pool_mode = config.get("mm_spatial_pool_mode", "bilinear")
        self.mm_vision_tower = resolve_runtime_path(config.get("mm_vision_tower"))

        # CTR parameters
        self.ctr_retain_tokens = config.get("ctr_retain_tokens", 50)
        self.ctr_similarity_threshold = config.get("ctr_similarity_threshold", 0.9)
        self.ctr_k = config.get("ctr_k", 7)
        self.ctr_beta = config.get("ctr_beta", 0.6)

        # OQM parameters
        self.oqm_enable_quantization = config.get("oqm_enable_quantization", True)
        self.oqm_quantization_bits = config.get("oqm_quantization_bits", 4)
        self.oqm_retrieval_max_tokens = config.get("oqm_retrieval_max_tokens", 12544)
        self.oqm_sliding_window_size = config.get("oqm_sliding_window_size", 4800)
        self.oqm_init_token_count = config.get("oqm_init_token_count", 14)

        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._config = None
        self._llava_imports = None

    def _set_official_env(self):
        """Set the same env-driven config knobs used by StreamingTOM."""
        os.environ["WRAPPER"] = "streamingtom"
        os.environ["CTR_RETAIN_TOKENS"] = str(self.ctr_retain_tokens)
        os.environ["CTR_SIMILARITY_THRESHOLD"] = str(self.ctr_similarity_threshold)
        os.environ["CTR_K"] = str(self.ctr_k)
        os.environ["CTR_BETA"] = str(self.ctr_beta)
        os.environ["OQM_ENABLE_QUANTIZATION"] = "1" if self.oqm_enable_quantization else "0"
        os.environ["OQM_QUANTIZATION_BITS"] = str(self.oqm_quantization_bits)
        os.environ["OQM_RETRIEVAL_MAX_TOKENS"] = str(self.oqm_retrieval_max_tokens)
        os.environ["OQM_GROUP_SIZE"] = str(self.ctr_retain_tokens)
        os.environ["OQM_INIT_TOKEN_COUNT"] = str(self.oqm_init_token_count)
        os.environ["OQM_SLIDING_WINDOW_SIZE"] = str(self.oqm_sliding_window_size)
        os.environ["STREAMING_ENCODER_BATCH_SIZE"] = str(self.encoder_batch_size)
        os.environ.setdefault("STREAMINGTOM_USE_FULL_PROMPT", "0")

    def _init_model(self):
        """Lazy initialization of LLaVA-OneVision with StreamingTOM patch."""
        if self._model is not None:
            return

        # Cluster nodes have no internet — force offline so LLaVA-NeXT internal
        # code doesn't try to validate the local_path as a HF Hub repo_id and
        # raise "Repo id must be in form 'repo_name'".
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._set_official_env()

        # Prefer the vendored LLaVA-NeXT checkout used by the official repo.
        for path in (_LLAVA_NEXT_SRC, _STREAMINGTOM_SRC):
            if path not in sys.path:
                sys.path.insert(0, path)

        from llava.model.builder import load_pretrained_model
        if getattr(self, "use_streamingtom_patch", True):
            from streamingtom.main import streamingtom
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        self._llava_imports = {
            "SeparatorStyle": SeparatorStyle,
            "conv_templates": conv_templates,
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
        }

        print(f"Loading StreamingTOM LLaVA-OneVision from: {self.local_path}")
        print(
            f"  ctr_retain_tokens={self.ctr_retain_tokens}, ctr_k={self.ctr_k}, "
            f"ctr_beta={self.ctr_beta}, oqm_bits={self.oqm_quantization_bits}, "
            f"stream_fps={self.stream_fps}"
        )

        overwrite_config = {
            "mm_spatial_pool_stride": self.mm_spatial_pool_stride,
            "mm_spatial_pool_mode": self.mm_spatial_pool_mode,
        }
        if self.mm_vision_tower:
            overwrite_config["mm_vision_tower"] = self.mm_vision_tower
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.local_path,
            None,
            self.model_name_for_loader,
            device_map="cuda:0",
            overwrite_config=overwrite_config,
        )

        model.tokenizer = tokenizer
        model.fps = self.stream_fps
        # `use_streamingtom_patch=false` makes this wrapper run vanilla
        # LLaVA-Video/OneVision (skip the streamingtom KV-compression patch).
        # Used by §4.3.4(a) to evaluate the vanilla LlavaQwenForCausalLM base
        # via the same vendored LLaVA-NeXT loader (which supports `grid`
        # mm_newline_position that LLaVA-Video uses).
        if getattr(self, "use_streamingtom_patch", True):
            from streamingtom.main import streamingtom
            self._model = streamingtom(model, "llava")
        else:
            self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._config = self._model.config
        self._model.eval()
        if getattr(self, "use_streamingtom_patch", True):
            print("StreamingTOM loaded using official patched generate path.")
        else:
            print("LLaVA-Video/OneVision vanilla loaded (no streamingtom patch).")

    def _extract_video_frames(self, video_path: str, max_time: float = float("inf")) -> List[Image.Image]:
        """Extract PIL frames at stream_fps, matching the frames supplied to generate."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS for video: {video_path}")
        frame_interval = max(1, round(video_fps / self.stream_fps))
        frame_idx = 0
        frames: List[Image.Image] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                time_sec = frame_idx / video_fps
                if time_sec > max_time:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            frame_idx += 1
        cap.release()
        return frames

    def _build_llava_prompt(self, prompt: str) -> torch.Tensor:
        """Wrap a text prompt exactly like the official LLaVA-OneVision adapter."""
        imports = self._llava_imports
        conv = imports["conv_templates"][self.conv_template].copy()
        question = f'{imports["DEFAULT_IMAGE_TOKEN"]}\n{prompt}'
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = imports["tokenizer_image_token"](
            prompt_question,
            self._tokenizer,
            imports["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0)
        return input_ids.to(self._device)

    @property
    def _device(self):
        return next(self._model.parameters()).device

    def _preprocess_frames(self, frames: List[Any]) -> torch.Tensor:
        """Use LLaVA's own image preprocessing and return a [T,C,H,W] tensor."""
        imports = self._llava_imports
        pil_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                pil_frames.append(frame.convert("RGB"))
            else:
                pil_frames.append(Image.fromarray(frame).convert("RGB"))

        # The official video path feeds raw per-frame SigLIP pixels [T,C,H,W].
        # LLaVA anyres process_images would add a patch/crop dimension, which
        # StreamingTOM's frame generator does not accept.
        image_tensor = self._image_processor.preprocess(pil_frames, return_tensors="pt")["pixel_values"]
        return image_tensor.to(dtype=torch.float16, device=self._device)

    def _generate_from_frames(self, frames: List[Any], prompt: str) -> str:
        self._init_model()
        if not frames:
            return ""

        input_ids = self._build_llava_prompt(prompt)
        pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_id).to(self._device)
        image_tensor = self._preprocess_frames(frames)

        imports = self._llava_imports
        conv = imports["conv_templates"][self.conv_template].copy()
        stop_str = conv.sep if conv.sep_style != imports["SeparatorStyle"].TWO else conv.sep2
        stopping_criteria = imports["KeywordsStoppingCriteria"]([stop_str], self._tokenizer, input_ids)

        self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
        self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        gen_kwargs = {
            "attention_mask": attention_mask,
            "pad_token_id": pad_token_id,
            "images": [image_tensor],
            "modalities": ["video"],
            "stopping_criteria": [stopping_criteria],
            "use_cache": True,
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)
        text = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()

    def stream_video_inference(
        self,
        video_path: str,
        queries: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Run each query through the official single-sample generate path.

        Frames are sampled up to each OVO-S query time so the benchmark remains
        causal, while the model-side compression/generation path is the official
        StreamingTOM path.
        """
        self._init_model()
        results: List[Tuple[Dict[str, Any], str]] = []
        t0 = time.time()
        total_frames = 0

        for q in queries:
            try:
                frames = self._extract_video_frames(video_path, max_time=q["query_time"])
                total_frames += len(frames)
                resp = self._generate_from_frames(frames, q["prompt"])
            except Exception as e:
                print(f"  StreamingTOM error on query {q.get('query_id', '?')}: {e}")
                resp = ""
            results.append((q, resp))

        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(
            f"  StreamingTOM processed {len(results)} queries, "
            f"{total_frames} sampled frames total in {elapsed:.1f}s"
        )
        return results

    def inference(self, frames: List[Any], prompt: str, option_images: List[Any] = None) -> str:
        """Single-query inference from pre-extracted frames."""
        if option_images:
            from option_utils import append_option_images_to_frames

            frames = append_option_images_to_frames(
                frames,
                option_images,
                max_n=int(self.config.get("max_images_per_prompt", self.max_frames)),
            )
        return self._generate_from_frames(frames, prompt)
