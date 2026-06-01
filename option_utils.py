"""Helpers for multiple-choice options that may contain base64-encoded images.

Task 4.3.x ships options as `{"A": "data:image/png;base64,...", ...}`. The default
prompt builder would otherwise inline the entire base64 string. This module
detects those values, decodes them into PIL.Image objects, and substitutes a
short placeholder so the model receives a clean text prompt plus a separate
list of images aligned to option labels.
"""

from __future__ import annotations

import base64
import re
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Tuple

from PIL import Image


DATA_URI_RE = re.compile(r"^data:image/[A-Za-z0-9.+\-]+;base64,", re.IGNORECASE)

# Default placeholder text shown to the model in place of the base64 string.
PLACEHOLDER_TEMPLATE = "[See option image labeled {label}]"

# Instruction appended to the prompt when image options are present.
IMAGE_OPTIONS_INSTRUCTION = (
    "Note: The images that follow the video correspond to options "
    "{labels} in order. Choose the option whose image best matches the question."
)


def is_image_option_value(value: Any) -> bool:
    """Return True if *value* looks like a base64 image data URI."""
    if not isinstance(value, str):
        return False
    return bool(DATA_URI_RE.match(value))


def has_image_options(options: Mapping[str, Any]) -> bool:
    """Return True if any value in *options* is a base64 image data URI."""
    if not isinstance(options, Mapping):
        return False
    return any(is_image_option_value(v) for v in options.values())


def decode_option_image(value: str, frame_size: Optional[int] = None) -> Image.Image:
    """Decode a `data:image/...;base64,...` URI to a PIL.Image.

    If *frame_size* is given, scale the longest side down to that value (keeping
    aspect ratio) so the option image lands in the same token budget as a video
    frame.
    """
    if not is_image_option_value(value):
        raise ValueError("value is not a base64 image data URI")
    _, _, b64 = value.partition(",")
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes))
    # PNG screenshots arrive as RGBA; vLLM image preprocessors expect RGB.
    if img.mode != "RGB":
        img = img.convert("RGB")
    if frame_size and max(img.size) > frame_size:
        scale = frame_size / max(img.size)
        new_size = (int(round(img.size[0] * scale)), int(round(img.size[1] * scale)))
        img = img.resize(new_size, Image.BILINEAR)
    return img


def split_options(
    options: Mapping[str, Any],
    frame_size: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Tuple[str, Image.Image]]]:
    """Split *options* into text-only options and a list of (label, PIL.Image).

    Returns:
        text_options: same keys as *options*, with image values replaced by a
            short placeholder string.
        option_images: list of (label, PIL.Image) in original option-label order
            (sorted by label for determinism).
    """
    if not isinstance(options, Mapping):
        return dict(options) if options else {}, []

    text_options: Dict[str, Any] = {}
    option_images: List[Tuple[str, Image.Image]] = []
    for label in sorted(options.keys()):
        value = options[label]
        if is_image_option_value(value):
            text_options[label] = PLACEHOLDER_TEMPLATE.format(label=label)
            option_images.append((label, decode_option_image(value, frame_size)))
        else:
            text_options[label] = value
    return text_options, option_images


def build_image_options_instruction(option_images: List[Tuple[str, Image.Image]]) -> str:
    """Build the instruction line shown to the model when image options exist."""
    labels = ", ".join(label for label, _ in option_images)
    return IMAGE_OPTIONS_INSTRUCTION.format(labels=labels)


def append_option_images_to_frames(
    frames: Optional[List[Image.Image]],
    option_images: Optional[List[Tuple[str, Image.Image]]],
    max_n: Optional[int] = None,
) -> List[Image.Image]:
    """Append option PIL images to the end of *frames*, capped at *max_n* total.

    Used by model wrappers that do not separately distinguish video frames from
    image options — they let their existing image encoder handle the extra
    trailing images. The companion prompt (see BaseModel.prepare_prompt) already
    tells the model that the last N images correspond to options A, B, ... in
    order.

    When *frames* is non-empty, option images are resized to match
    ``frames[0].size`` so that downstream pipelines which stack all visual
    inputs into a single tensor (e.g. Spatial-MLLM's video processor) do not
    fail on shape mismatches. Aspect ratio of options is sacrificed if needed —
    for trajectory-shape questions in task 4.3.x this is acceptable.

    If max_n is set and there are more option images than allowed, oldest
    frames are dropped first to make room (options are never trimmed).
    """
    frames = list(frames or [])
    if not option_images:
        return frames
    opt_imgs = [img for _, img in option_images]
    if frames:
        target_size = frames[0].size  # (W, H) — assume frames share a shape
        opt_imgs = [
            img.resize(target_size, Image.BILINEAR) if img.size != target_size else img
            for img in opt_imgs
        ]
    if max_n is None:
        return frames + opt_imgs
    if len(opt_imgs) >= max_n:
        return opt_imgs[:max_n]
    budget = max_n - len(opt_imgs)
    if len(frames) > budget:
        frames = frames[-budget:]
    return frames + opt_imgs
