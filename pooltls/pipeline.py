from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


STAGE_NAMES = (
    "prepare_stage1",
    "train_stage1",
    "generate_train",
    "generate_development",
    "generate_test",
    "cluster_all",
    "prepare_stage2",
    "train_stage2",
    "select_development",
    "score_test",
    "build_test_timelines",
    "evaluate_test",
)


def select_stages(
    from_stage: str | None = None,
    stop_after: str | None = None,
) -> tuple[str, ...]:
    start_name = from_stage or STAGE_NAMES[0]
    stop_name = stop_after or STAGE_NAMES[-1]
    if start_name not in STAGE_NAMES:
        raise ValueError(f"unknown start stage: {start_name}")
    if stop_name not in STAGE_NAMES:
        raise ValueError(f"unknown stop stage: {stop_name}")
    start = STAGE_NAMES.index(start_name)
    stop = STAGE_NAMES.index(stop_name)
    if start > stop:
        raise ValueError("from-stage occurs after stop-after")
    return STAGE_NAMES[start : stop + 1]


def artifact_paths(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir).expanduser().resolve()
    return {
        "root": root,
        "stage1_data": root / "stage1_data",
        "stage1_adapter": root / "models" / "stage1" / "final_adapter",
        "mentions": root / "mentions",
        "candidates": root / "candidates",
        "stage2_data": root / "stage2_data",
        "cross_models": root / "models" / "cross_encoder",
        "selection": root / "selection" / "selected_config.json",
        "scores": root / "scores",
        "timelines": root / "timelines",
        "evaluation": root / "evaluation",
        "logs": root / "logs",
    }


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    from .data import DatasetReader

    paths = config["paths"]
    required = ("data_root", "base_model", "gte_model", "cross_encoder_model")
    missing = [
        name
        for name in required
        if paths.get(name) is None or not Path(paths[name]).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "missing or non-directory configured paths: " + ", ".join(missing)
        )

    reader = DatasetReader(config)
    partitions: dict[str, Any] = {}
    for partition in ("train", "development", "test"):
        entity_ids = reader.entity_ids(partition)
        article_count = 0
        reference_count = 0
        for entity_id in entity_ids:
            constraints = reader.constraints(entity_id)
            if len(constraints) != 5:
                raise ValueError(f"{entity_id} has {len(constraints)} constraints, expected 5")
            article_count += len(reader.articles(entity_id))
            reference_count += sum(
                len(events) for events in reader.references(entity_id).values()
            )
        partitions[partition] = {
            "entities": len(entity_ids),
            "articles": article_count,
            "reference_events": reference_count,
        }
    return {
        "dataset": config["dataset"],
        "model_paths_checked": True,
        "model_weights_loaded": False,
        "partitions": partitions,
    }


def run_handlers(
    stages: Sequence[str],
    handlers: Mapping[str, Callable[[], None]],
) -> None:
    for stage in stages:
        if stage not in handlers:
            raise KeyError(f"no handler registered for stage {stage}")
        handlers[stage]()
