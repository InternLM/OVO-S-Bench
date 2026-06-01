"""Utility functions for OVO-S evaluation."""

from .frame_utils import (
    extract_frames_for_query,
    extract_frames_for_annotation,
    get_video_info,
)

__all__ = [
    "extract_frames_for_query",
    "extract_frames_for_annotation",
    "get_video_info",
]
