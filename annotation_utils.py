"""Annotation loading helpers for JSON and Parquet inputs."""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


_LEVEL_ALIAS_RE = re.compile(r"^level[_-]?(\d+)$", re.IGNORECASE)


def _infer_requested_level(path: Path) -> Optional[int]:
    """Infer level filters from legacy names like level_1.json."""
    match = _LEVEL_ALIAS_RE.match(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_annotation_path(annotation_path: str) -> tuple[Path, Optional[int]]:
    """Resolve legacy level_N.json paths to the consolidated parquet file."""
    path = Path(annotation_path)
    requested_level = _infer_requested_level(path)
    if path.exists():
        return path, requested_level

    fallback = path.with_name("annotations.parquet")
    is_consolidated_alias = path.stem.lower() == "annotations"
    if fallback.exists() and (requested_level is not None or is_consolidated_alias):
        return fallback, requested_level

    return path, requested_level


def _load_json_annotations(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("annotations", [])


def _load_parquet_annotations(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading parquet annotations requires pyarrow. "
            "Install it with `python -m pip install pyarrow`."
        ) from exc

    return pq.read_table(path).to_pylist()


def _to_builtin(value: Any) -> Any:
    """Convert pyarrow/pandas/numpy containers into plain Python values."""
    if hasattr(value, "as_py"):
        value = value.as_py()

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _to_builtin(value.tolist())
        except TypeError:
            pass

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, Mapping):
        return {k: _to_builtin(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_builtin(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)

    return value


def _normalize_options(value: Any) -> Any:
    """Parquet map columns arrive as [(key, value), ...]; use dicts downstream."""
    value = _to_builtin(value)
    if isinstance(value, Mapping):
        return dict(value)

    if isinstance(value, list):
        options: dict[str, Any] = {}
        for item in value:
            if isinstance(item, Mapping) and {"key", "value"} <= set(item):
                key, option_value = item["key"], item["value"]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                key, option_value = item
            else:
                return value
            options[str(key)] = option_value
        return options

    return value


def _normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: _to_builtin(value) for key, value in annotation.items()}

    if "options" in normalized:
        normalized["options"] = _normalize_options(normalized["options"])

    reconstructed = normalized.get("re_constructed")
    if isinstance(reconstructed, Mapping) and "options" in reconstructed:
        reconstructed = dict(reconstructed)
        reconstructed["options"] = _normalize_options(reconstructed["options"])
        normalized["re_constructed"] = reconstructed

    return normalized


def _level_matches(annotation: dict[str, Any], requested_level: Optional[int]) -> bool:
    if requested_level is None:
        return True
    try:
        return int(annotation.get("level")) == requested_level
    except (TypeError, ValueError):
        return False


def load_annotations(annotation_path: str) -> list[dict[str, Any]]:
    """Load annotations from JSON or parquet, preserving the legacy list format."""
    path, requested_level = _resolve_annotation_path(annotation_path)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        annotations = _load_parquet_annotations(path)
    else:
        annotations = _load_json_annotations(path)

    normalized = [_normalize_annotation(ann) for ann in annotations]
    return [ann for ann in normalized if _level_matches(ann, requested_level)]
