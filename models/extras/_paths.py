"""Resolve upstream source directories for extras wrappers.

Each extras wrapper (HERMES, FluxMem, InfiniPot-V, StreamingTOM, etc.)
depends on an upstream research repo that the user must clone separately —
we don't vendor them because they bring their own (often conflicting)
dependency stacks.

Lookup order for an upstream named `HERMES`:
  1. `$OVO_S_HERMES_SRC`  (env var, exact name in upper-case)
  2. `$OVO_S_EXTRAS_SRC/HERMES`  (umbrella env var)
  3. `<repo_root>/extras_src/HERMES`  (default convention)

Returns the resolved path string if any candidate exists; raises
FileNotFoundError with an install hint otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

# Three levels up: models/extras/_paths.py → models/extras → models → repo_root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EXTRAS_DIR = _REPO_ROOT / "extras_src"


def find_upstream_src(name: str, *, strict: bool = True) -> str:
    """Resolve the path to upstream repo `name`. See module docstring.

    With `strict=True` (default), raises `FileNotFoundError` when none of the
    candidate paths exist. Use `strict=False` for module-level constants where
    the path is only consulted on first use — the caller can detect the
    missing directory and produce a more contextual error.
    """
    candidates = []
    env_specific = os.environ.get(f"OVO_S_{name.upper()}_SRC")
    if env_specific:
        candidates.append(Path(env_specific))
    umbrella = os.environ.get("OVO_S_EXTRAS_SRC")
    if umbrella:
        candidates.append(Path(umbrella) / name)
    candidates.append(_DEFAULT_EXTRAS_DIR / name)

    for cand in candidates:
        if cand.exists():
            return str(cand)

    if not strict:
        # Return the first preference (env-driven if set, otherwise default).
        return str(candidates[0])

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Upstream source for '{name}' not found. Tried:\n  {tried}\n"
        f"Clone the upstream repo to one of the above locations, or set "
        f"OVO_S_{name.upper()}_SRC. See models/extras/README.md."
    )
