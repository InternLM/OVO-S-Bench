"""
API-based model implementations for OVO-S evaluation.
Supports OpenAI, Google Gemini, and Anthropic Claude.
"""

import os
import base64
import time
from typing import Dict, List, Any, Optional
from io import BytesIO
from PIL import Image

from .base import BaseModel

# API limits for different providers
API_IMAGE_LIMITS = {
    "openai": 50,     # GPT-4o / apiyi proxy hard cap
    "google": 50,     # Gemini Pro / apiyi proxy hard cap
    "anthropic": 20,
}


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OpenAIModel(BaseModel):
    """OpenAI GPT-4V model wrapper."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("CUSTOM_BASE_URL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY or CUSTOM_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """Run inference using OpenAI API."""
        # Enforce API image limit
        max_images = API_IMAGE_LIMITS.get("openai", 50)
        if len(frames) > max_images:
            frames = frames[-max_images:]  # Keep the most recent frames

        # Image-option path (task 4.3.x): append option PIL images so they
        # become extra trailing image_url blocks alongside the video frames.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=max_images
            )

        # Build message content with images
        content = []

        # Add images
        for frame in frames:
            base64_image = image_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Make API call with retry
        extra_body = self.config.get("extra_body") or {}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_body=extra_body,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e


class GeminiModel(BaseModel):
    """Google Gemini model wrapper."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._init_client()

    def _init_client(self):
        """Initialize Gemini client."""
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_id)

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """Run inference using Gemini API."""
        # Enforce API image limit
        max_images = API_IMAGE_LIMITS.get("google", 100)
        if len(frames) > max_images:
            frames = frames[-max_images:]

        # Image-option path (task 4.3.x): append option PIL images so the
        # `content` list naturally carries them as trailing image inputs.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=max_images
            )

        # Build content with images and prompt
        content = list(frames) + [prompt]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    content,
                    generation_config={
                        "max_output_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                )
                return response.text.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e


class ClaudeModel(BaseModel):
    """Anthropic Claude model wrapper."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client."""
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=api_key)

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """Run inference using Claude API."""
        # Enforce API image limit
        max_images = API_IMAGE_LIMITS.get("anthropic", 20)
        if len(frames) > max_images:
            frames = frames[-max_images:]

        # Image-option path (task 4.3.x): append option PIL images so they get
        # the same {"type": "image", "source": ...} block treatment below.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=max_images
            )

        # Build message content with images
        content = []

        # Add images
        for frame in frames:
            base64_image = image_to_base64(frame)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": content}]
                )
                return response.content[0].text.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e


# Model registry for easy instantiation
MODEL_REGISTRY = {
    "openai": OpenAIModel,
    "google": GeminiModel,
    "gemini-native": None,  # filled in below after class is defined
    "anthropic": ClaudeModel,
}


class GeminiNativeModel(BaseModel):
    """Gemini via native /v1beta REST API (works through OpenAI-compatible proxies
    like CUSTOM_BASE_URL that route /v1beta/* directly to Google).

    Avoids the google-generativeai SDK so any HTTP proxy / forwarder works.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._init_client()

    def _init_client(self):
        import requests  # local import to avoid hard dep at module import time

        self._requests = requests
        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("CUSTOM_API_KEY")
        )
        base_url = (
            os.getenv("GOOGLE_BASE_URL")
            or os.getenv("CUSTOM_BASE_URL")
            or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or CUSTOM_API_KEY not found in environment"
            )
        # Proxies often expose both /v1 and /v1beta; strip trailing /v1 if present.
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self._url = f"{base_url}/v1beta/models/{self.model_id}:generateContent"
        self._headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        # Same image budget as the google-genai path.
        max_images = API_IMAGE_LIMITS.get("google", 100)
        if len(frames) > max_images:
            frames = frames[-max_images:]

        # Image-option path (task 4.3.x): append option PIL images so the loop
        # below packs them as additional inline_data parts in label order.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=max_images
            )

        parts = []
        for f in frames:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_to_base64(f),
                }
            })
        parts.append({"text": prompt})

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        max_retries = 3
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self._requests.post(
                    self._url, headers=self._headers, json=body, timeout=180
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError, TypeError) as e:
                    raise RuntimeError(
                        f"unexpected response shape: {str(data)[:500]}"
                    ) from e
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise last_err


MODEL_REGISTRY["gemini-native"] = GeminiNativeModel


def create_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to create model instances.

    Args:
        model_name: Name of the model
        config: Model configuration

    Returns:
        Model instance
    """
    model_type = config.get("type", "api")
    provider = config.get("provider", "openai")

    # Handle native offline providers directly. Importing the full vLLM/native
    # registry pulls in unrelated CUDA stacks and can destabilize custom
    # runtimes such as Flash-VStream.
    if model_type == "offline" or model_type == "vllm":
        native_providers = {
            "flash-vstream": ("flash_vstream_models", "FlashVStreamQwenModel"),
            "streamforest": ("streamforest_models", "StreamForestModel"),
            "streaming-vlm": ("streaming_vlm_models", "StreamingVLMModel"),
            "minicpm-v": ("minicpmv_models", "MiniCPMVModel"),
            "streamingtom": ("streamingtom_models", "StreamingTOMModel"),
        }
        if provider in native_providers:
            import importlib

            module_name, class_name = native_providers[provider]
            module = importlib.import_module(f".{module_name}", package=__package__)
            return getattr(module, class_name)(model_name, config)

        from .vllm_models import create_vllm_model
        return create_vllm_model(model_name, config)

    # Handle API models
    model_class = MODEL_REGISTRY.get(provider)

    if model_class is None:
        raise ValueError(f"Unknown provider: {provider}")

    return model_class(model_name, config)
