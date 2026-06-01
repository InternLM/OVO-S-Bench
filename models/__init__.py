"""Model wrapper package for OVO-S-Bench evaluation.

Top-level imports are kept light so optional / heavyweight model wrappers
(e.g. vLLM-backed open-source MLLMs in `models.extras`) do not have to be
imported unless they're actually being used.
"""

from .base import BaseModel
from .api_models import create_model

__all__ = ["BaseModel", "create_model"]
