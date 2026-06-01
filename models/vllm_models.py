"""
vLLM offline batch inference for Qwen VL models.
Uses direct LLM class instead of server-based approach.

Requirements:
    pip install vllm>=0.11.0 qwen-vl-utils>=0.0.14 transformers
"""

import os
import torch
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from .base import BaseModel, resolve_runtime_path

# Set HF cache to user directory to avoid permission issues
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_MODULES_CACHE"] = os.path.expanduser("~/.cache/huggingface/modules")

# Set multiprocessing method for vLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _flatten_option_images(option_images: Optional[List[Any]]) -> List[Image.Image]:
    """Accept either [(label, PIL.Image), ...] or [PIL.Image, ...] and return PIL.Image list."""
    if not option_images:
        return []
    flat: List[Image.Image] = []
    for item in option_images:
        if isinstance(item, tuple) and len(item) == 2:
            flat.append(item[1])
        else:
            flat.append(item)
    return flat


class _VLLMQwenBase(BaseModel):
    """Shared base for Qwen3-VL and Qwen3.5 vLLM models.

    Uses qwen_vl_utils.process_vision_info (>= 0.0.14) for standard
    multi-modal input construction in both image and video modes.
    """

    # Subclasses can override for logging
    _display_family = "Qwen"
    _LLM_KWARG_KEYS = {
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "enforce_eager",
        "dtype",
        "disable_mm_preprocessor_cache",
    }
    _PROCESSOR_KWARG_KEYS = {"min_pixels", "max_pixels", "use_fast"}

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = self._resolve_local_path(
            config.get("local_path", config.get("model_id"))
        )
        self.tensor_parallel_size = config.get(
            "tensor_parallel_size", torch.cuda.device_count()
        )
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.85)
        self.max_images_per_prompt = config.get("max_images_per_prompt", 64)
        # Reserve a small image budget for option images even in video mode
        # (task 4.3.x: A/B/C/D bird's-eye trajectory maps).
        self.max_option_images = int(config.get("max_option_images", 6))
        self.nframes = config.get("nframes", 0)
        self.use_video_input = self.nframes > 0
        # Thinking mode: None=use model default, True/False=explicit control
        self.enable_thinking = config.get("enable_thinking", None)
        self.system_prompt = config.get("system_prompt")
        self.top_p = config.get("top_p")
        self.repetition_penalty = config.get("repetition_penalty")
        self.processor_kwargs = dict(config.get("processor_kwargs") or {})
        for key in self._PROCESSOR_KWARG_KEYS:
            if key in config:
                self.processor_kwargs[key] = config[key]

        self.llm = None
        self.processor = None
        self.sampling_params = None

    @staticmethod
    def _resolve_local_path(path: str) -> str:
        """Resolve a local checkpoint path, falling back to OVO_S_MODEL_ROOT."""
        if not path:
            return path
        return resolve_runtime_path(path)

    def _messages_with_optional_system(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build chat messages, optionally inserting a model-specific system prompt."""
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })
        messages.append({"role": "user", "content": content})
        return messages

    def _init_model(self):
        """Lazy initialization of vLLM model."""
        if self.llm is not None:
            return

        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        print(f"Loading {self._display_family} model from: {self.local_path}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")
        if self.use_video_input:
            print(f"Video input mode: {self.nframes} fixed frames")
        if self.enable_thinking is not None:
            print(f"Thinking mode: {'enabled' if self.enable_thinking else 'disabled'}")

        self.processor = AutoProcessor.from_pretrained(
            self.local_path, trust_remote_code=True, **self.processor_kwargs
        )

        mm_limit = (
            {"image": max(self.max_option_images, 0), "video": 1}
            if self.use_video_input
            else {"image": self.max_images_per_prompt, "video": 0}
        )
        llm_kwargs = dict(
            model=self.local_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt=mm_limit,
        )
        for key in self._LLM_KWARG_KEYS:
            if key in self.config:
                llm_kwargs[key] = self.config[key]
        llm_kwargs.setdefault("disable_mm_preprocessor_cache", True)
        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError as exc:
            if "disable_mm_preprocessor_cache" not in llm_kwargs:
                raise
            print(
                "Retrying Qwen vLLM init without "
                f"disable_mm_preprocessor_cache: {exc}"
            )
            llm_kwargs.pop("disable_mm_preprocessor_cache", None)
            self.llm = LLM(**llm_kwargs)

        sampling_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            sampling_kwargs["top_p"] = self.top_p
        if self.repetition_penalty is not None:
            sampling_kwargs["repetition_penalty"] = self.repetition_penalty
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def extract_frames(self, video_path: str, query_time: float):
        """Override frame extraction for video input mode."""
        from utils.frame_utils import extract_frames_fixed_count, extract_frames_for_query

        if self.use_video_input:
            return extract_frames_fixed_count(
                video_path, query_time,
                nframes=self.nframes,
                frame_size=self.frame_size,
            )
        return extract_frames_for_query(
            video_path, query_time,
            max_frames=self.max_frames,
            fps=self.fps,
            frame_size=self.frame_size,
        )

    def _prepare_input(
        self,
        frames: List[Image.Image],
        prompt: str,
        option_images: Optional[List[Any]] = None,
    ) -> dict:
        """Prepare input for vLLM inference.

        Uses qwen_vl_utils.process_vision_info (>= 0.0.14) to build
        standard multi-modal inputs with proper video metadata.

        When *option_images* is provided (list of (label, PIL.Image) or plain
        PIL.Image entries), each option image is appended as its own
        `{"type": "image", ...}` block AFTER the video / frame blocks and
        BEFORE the text prompt so the model sees them in label order.
        """
        from qwen_vl_utils import process_vision_info

        frames = frames or []
        opt_imgs = _flatten_option_images(option_images)
        if opt_imgs and len(opt_imgs) > self.max_option_images:
            # Cap so vLLM's limit_mm_per_prompt is not exceeded.
            opt_imgs = opt_imgs[: self.max_option_images]

        if not frames and not opt_imgs:
            messages = self._messages_with_optional_system(
                [{"type": "text", "text": prompt}]
            )
            chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if self.enable_thinking is not None:
                chat_kwargs["enable_thinking"] = self.enable_thinking
            text = self.processor.apply_chat_template(messages, **chat_kwargs)
            return {"prompt": text}

        content: List[Dict[str, Any]] = []
        if frames:
            if self.use_video_input:
                content.append({
                    "type": "video",
                    "video": frames,    # list[PIL.Image] — natively supported
                    "fps": self.fps,    # sampling rate from config (default 2.0)
                })
            else:
                content.extend({"type": "image", "image": f} for f in frames)

        for img in opt_imgs:
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})

        messages = self._messages_with_optional_system(content)
        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = self.enable_thinking
        text = self.processor.apply_chat_template(messages, **chat_kwargs)

        # Use process_vision_info for standard multi-modal data construction
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if self.use_video_input:
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            if image_inputs is not None:
                # Option images live in image_inputs even when the video block
                # is the dominant modality. Include them so the engine has
                # one image entry per option.
                mm_data["image"] = image_inputs
        else:
            video_kwargs = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

        vllm_input = {
            "prompt": text,
            "multi_modal_data": mm_data,
        }
        if video_kwargs:
            vllm_input["mm_processor_kwargs"] = video_kwargs
        return vllm_input

    def inference(
        self,
        frames: List[Image.Image],
        prompt: str,
        option_images: Optional[List[Any]] = None,
    ) -> str:
        """Run single inference."""
        self._init_model()
        vllm_input = self._prepare_input(frames, prompt, option_images=option_images)
        outputs = self.llm.generate([vllm_input], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def batch_inference(
        self,
        batch_frames: List[List[Image.Image]],
        batch_prompts: List[str],
        batch_option_images: Optional[List[Optional[List[Any]]]] = None,
    ) -> List[str]:
        """Run batch inference for better throughput."""
        self._init_model()
        if batch_option_images is None:
            batch_option_images = [None] * len(batch_prompts)
        vllm_inputs = [
            self._prepare_input(frames, prompt, option_images=opt_imgs)
            for frames, prompt, opt_imgs in zip(
                batch_frames, batch_prompts, batch_option_images
            )
        ]
        outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]


class VLLMQwenModel(_VLLMQwenBase):
    """vLLM offline inference for Qwen3-VL models."""

    _display_family = "Qwen3-VL"


class VLLMQwen35Model(_VLLMQwenBase):
    """vLLM offline inference for Qwen3.5 models.

    Qwen3.5 integrates vision via early fusion (no separate -VL variant).
    Supports both dense (4B, 27B) and MoE (122B-A10B) variants.
    """

    _display_family = "Qwen3.5"


class VLLMQwen25VLModel(_VLLMQwenBase):
    """vLLM offline inference for Qwen2.5-VL models.

    Differences from Qwen3-VL:
    - process_vision_info may return 2 values on older qwen_vl_utils (<0.0.14);
      falls back gracefully when 3-value API is unavailable.
    - apply_chat_template does not support enable_thinking.
    """

    _display_family = "Qwen2.5-VL"

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        # Qwen2.5-VL does not support thinking mode
        self.enable_thinking = None

    def _prepare_input(
        self,
        frames: List[Image.Image],
        prompt: str,
        option_images: Optional[List[Any]] = None,
    ) -> dict:
        """Prepare input for vLLM inference.

        Uses process_vision_info with graceful fallback for older qwen_vl_utils
        versions that return only (image_inputs, video_inputs) without video_kwargs.
        """
        from qwen_vl_utils import process_vision_info

        frames = frames or []
        opt_imgs = _flatten_option_images(option_images)
        if opt_imgs and len(opt_imgs) > self.max_option_images:
            opt_imgs = opt_imgs[: self.max_option_images]

        if not frames and not opt_imgs:
            messages = self._messages_with_optional_system(
                [{"type": "text", "text": prompt}]
            )
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {"prompt": text}

        content: List[Dict[str, Any]] = []
        if frames:
            if self.use_video_input:
                content.append({
                    "type": "video",
                    "video": frames,    # list[PIL.Image]
                    "fps": self.fps,
                })
            else:
                content.extend({"type": "image", "image": f} for f in frames)

        for img in opt_imgs:
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})

        messages = self._messages_with_optional_system(content)
        # Qwen2.5-VL: no enable_thinking support in apply_chat_template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Try newer qwen_vl_utils (>= 0.0.14) API first (3 return values);
        # fall back to older API (2 return values) for Qwen2.5-VL compatibility.
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            video_kwargs = {}

        mm_data = {}
        if self.use_video_input:
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            if image_inputs is not None:
                mm_data["image"] = image_inputs
        else:
            video_kwargs = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

        vllm_input = {"prompt": text, "multi_modal_data": mm_data}
        if video_kwargs:
            vllm_input["mm_processor_kwargs"] = video_kwargs
        return vllm_input


class VLLMGemma4Model(BaseModel):
    """vLLM offline inference for Google Gemma4 multimodal models."""

    _LLM_KWARG_KEYS = {
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "enforce_eager",
        "dtype",
        "disable_mm_preprocessor_cache",
    }

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = _VLLMQwenBase._resolve_local_path(
            config.get("local_path", config.get("model_id"))
        )
        self.tensor_parallel_size = config.get(
            "tensor_parallel_size", torch.cuda.device_count()
        )
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.92)
        self.max_images_per_prompt = int(config.get("max_images_per_prompt", 128))
        self.max_soft_tokens = int(config.get("max_soft_tokens", 70))
        self.enable_thinking = config.get("enable_thinking", False)
        self.nframes = int(config.get("nframes", 0))

        self.llm = None
        self.processor = None
        self.sampling_params = None

    def _init_model(self):
        """Lazy initialization of the Gemma4 vLLM engine."""
        if self.llm is not None:
            return

        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        print(f"Loading Gemma4 model from: {self.local_path}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")
        print(f"max_images_per_prompt={self.max_images_per_prompt}")
        print(f"max_soft_tokens={self.max_soft_tokens}")
        print(f"Thinking mode: {'enabled' if self.enable_thinking else 'disabled'}")

        self.processor = AutoProcessor.from_pretrained(
            self.local_path, trust_remote_code=True
        )

        llm_kwargs = {
            "model": self.local_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
            "limit_mm_per_prompt": {
                "image": self.max_images_per_prompt,
                "audio": 0,
                "video": 0,
            },
            "mm_processor_kwargs": {"max_soft_tokens": self.max_soft_tokens},
        }
        for key in self._LLM_KWARG_KEYS:
            if key in self.config:
                llm_kwargs[key] = self.config[key]

        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError as exc:
            if "disable_mm_preprocessor_cache" not in llm_kwargs:
                raise
            print(f"Retrying Gemma4 LLM init without disable_mm_preprocessor_cache: {exc}")
            llm_kwargs.pop("disable_mm_preprocessor_cache", None)
            self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def extract_frames(self, video_path: str, query_time: float):
        """Use fixed-count frames when nframes is configured for OVO-S."""
        from utils.frame_utils import extract_frames_fixed_count, extract_frames_for_query

        if self.nframes > 0:
            return extract_frames_fixed_count(
                video_path,
                query_time,
                nframes=self.nframes,
                frame_size=self.frame_size,
            )
        return extract_frames_for_query(
            video_path,
            query_time,
            max_frames=self.max_frames,
            fps=self.fps,
            frame_size=self.frame_size,
        )

    def _prepare_input(self, frames: List[Image.Image], prompt: str) -> dict:
        frames = list(frames or [])
        if len(frames) > self.max_images_per_prompt:
            frames = frames[-self.max_images_per_prompt:]

        content = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        vllm_input = {"prompt": text}
        if frames:
            vllm_input["multi_modal_data"] = {"image": frames}
            vllm_input["mm_processor_kwargs"] = {
                "max_soft_tokens": self.max_soft_tokens
            }
        return vllm_input

    @staticmethod
    def _clean_output(text: str) -> str:
        try:
            from vllm.reasoning.gemma4_utils import parse_thinking_output

            parsed = parse_thinking_output(text)
            return (parsed.get("answer") or text).strip()
        except Exception:
            return text.strip()

    def inference(
        self,
        frames: List[Image.Image],
        prompt: str,
        option_images: Optional[List[Any]] = None,
    ) -> str:
        # Image-option path (task 4.3.x): treat each option image as one extra
        # trailing frame. Gemma4 has no separate video block, so this fits the
        # native image-list path with no further changes.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_images_per_prompt
            )
        self._init_model()
        outputs = self.llm.generate(
            [self._prepare_input(frames, prompt)],
            sampling_params=self.sampling_params,
        )
        return self._clean_output(outputs[0].outputs[0].text)

    def batch_inference(
        self,
        batch_frames: List[List[Image.Image]],
        batch_prompts: List[str],
        batch_option_images: Optional[List[Optional[List[Any]]]] = None,
    ) -> List[str]:
        self._init_model()
        if batch_option_images is None:
            batch_option_images = [None] * len(batch_prompts)
        # Same minimal path as inference(): trailing-frame append per batch item.
        from option_utils import append_option_images_to_frames
        merged_frames = [
            append_option_images_to_frames(f, oi, max_n=self.max_images_per_prompt) if oi else f
            for f, oi in zip(batch_frames, batch_option_images)
        ]
        vllm_inputs = [
            self._prepare_input(frames, prompt)
            for frames, prompt in zip(merged_frames, batch_prompts)
        ]
        outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
        return [self._clean_output(out.outputs[0].text) for out in outputs]


# Model registry for offline models (vLLM and native)
VLLM_MODEL_REGISTRY = {
    "gemma4": VLLMGemma4Model,
    "qwen": VLLMQwenModel,
    "qwen3.5": VLLMQwen35Model,
    "qwen2.5-vl": VLLMQwen25VLModel,
}


def _get_offline_registry():
    """Lazily build the full offline model registry including native providers."""
    registry = dict(VLLM_MODEL_REGISTRY)
    try:
        from .minicpmv_models import MiniCPMVModel
        registry["minicpm-v"] = MiniCPMVModel
    except ImportError as e:
        print(f"Warning: MiniCPM-V provider unavailable: {e}")
    try:
        from .internvl_models import VLLMInternVLModel
        registry["internvl"] = VLLMInternVLModel
    except ImportError as e:
        print(f"Warning: InternVL provider unavailable: {e}")
    try:
        from .extras.spatial_mllm_models import SpatialMLLMModel, SpatialMLLMSASamplingModel
        registry["spatial-mllm"] = SpatialMLLMModel
        registry["spatial-mllm-sa"] = SpatialMLLMSASamplingModel
    except ImportError as e:
        print(f"Warning: Spatial-MLLM provider unavailable: {e}")
    try:
        from .extras.cambrian_models import CambrianModel, CambrianLFPModel
        registry["cambrian-mllm"] = CambrianModel
        registry["cambrian-mllm-lfp"] = CambrianLFPModel
    except ImportError as e:
        print(f"Warning: Cambrian-S provider unavailable: {e}")
    try:
        from .extras.streaming_vlm_models import StreamingVLMModel
        registry["streaming-vlm"] = StreamingVLMModel
    except ImportError as e:
        print(f"Warning: StreamingVLM provider unavailable: {e}")
    try:
        from .extras.infinitevl_models import InfiniteVLModel
        registry["infinitevl"] = InfiniteVLModel
    except ImportError as e:
        print(f"Warning: InfiniteVL provider unavailable: {e}")
    try:
        from .extras.spatial_ttt_models import SpatialTTTModel
        registry["spatial-ttt"] = SpatialTTTModel
    except ImportError as e:
        print(f"Warning: Spatial-TTT provider unavailable: {e}")
    try:
        from .extras.hermes_models import HermesModel
        registry["hermes"] = HermesModel
    except ImportError as e:
        print(f"Warning: HERMES provider unavailable: {e}")
    try:
        from .extras.infinipot_models import InfiniPotModel
        registry["infinipot"] = InfiniPotModel
    except ImportError as e:
        print(f"Warning: InfiniPot-V provider unavailable: {e}")
    try:
        from .extras.fluxmem_models import FluxMemModel
        registry["fluxmem"] = FluxMemModel
    except ImportError as e:
        print(f"Warning: FluxMem provider unavailable: {e}")
    try:
        from .extras.streamingtom_models import StreamingTOMModel
        registry["streamingtom"] = StreamingTOMModel
    except ImportError as e:
        print(f"Warning: StreamingTOM provider unavailable: {e}")
    try:
        from .extras.streamforest_models import StreamForestModel
        registry["streamforest"] = StreamForestModel
    except ImportError as e:
        print(f"Warning: StreamForest provider unavailable: {e}")
    try:
        from .extras.llava_next_video_models import LLaVANextVideoModel
        registry["llava-next-video"] = LLaVANextVideoModel
    except ImportError as e:
        print(f"Warning: LLaVA-NeXT-Video provider unavailable: {e}")
    try:
        from .llava_onevision_vllm_models import VLLMLlavaOnevisionModel
        registry["llava-onevision-vllm"] = VLLMLlavaOnevisionModel
    except ImportError as e:
        print(f"Warning: LLaVA-OneVision vllm provider unavailable: {e}")
    try:
        from .extras.flash_vstream_models import FlashVStreamQwenModel
        registry["flash-vstream"] = FlashVStreamQwenModel
    except ImportError as e:
        print(f"Warning: Flash-VStream provider unavailable: {e}")
    return registry


def create_vllm_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
    """Factory function to create offline model instances."""
    provider = config.get("provider", "qwen")

    # Check for SA sampling variant
    if provider == "spatial-mllm" and config.get("sa_sampling", False):
        provider = "spatial-mllm-sa"

    # Check for LFP variant
    if provider == "cambrian-mllm" and config.get("lfp", False):
        provider = "cambrian-mllm-lfp"

    registry = _get_offline_registry()
    model_class = registry.get(provider)
    if model_class is None:
        raise ValueError(
            f"Unknown offline provider: {provider}. "
            f"Available: {list(registry.keys())}"
        )
    return model_class(model_name, config)
