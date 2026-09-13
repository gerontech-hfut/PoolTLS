from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_SECTIONS = (
    "paths",
    "embedding",
    "stage1",
    "clustering",
    "supervision",
    "cross_encoder",
    "fusion",
    "decoding",
)


def _resolve_paths(values: Mapping[str, Any], base: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            output[key] = None
        elif isinstance(value, str):
            path = Path(value).expanduser()
            output[key] = (path if path.is_absolute() else base / path).resolve()
        else:
            raise ValueError(f"paths.{key} must be a path string or null")
    return output


def load_config(path: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"configuration file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    config = dict(raw)
    dataset = config.get("dataset")
    if dataset not in {"crest", "wcep_ctg"}:
        raise ValueError("dataset must be 'crest' or 'wcep_ctg'")
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("configuration must contain a paths mapping")
    config["paths"] = _resolve_paths(paths, source.parent)
    config["config_path"] = source
    config["seed"] = int(config.get("seed", 42))
    if require_complete:
        missing = [name for name in REQUIRED_SECTIONS if not isinstance(config.get(name), Mapping)]
        if missing:
            raise ValueError(f"configuration is missing mappings: {', '.join(missing)}")
        for key in (
            "data_root",
            "base_model",
            "gte_model",
            "cross_encoder_model",
        ):
            if config["paths"].get(key) is None:
                raise ValueError(f"paths.{key} is required")
    return config


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return dict(value)
