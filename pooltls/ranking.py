from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .schema import Candidate, Constraint
from .text import cosine_matrix


ScoreMap = Mapping[str, np.ndarray]


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return result


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank each constraint column independently; ties receive their mean rank."""

    values = _matrix(scores, "scores")
    row_count, column_count = values.shape
    result = np.empty_like(values, dtype=np.float32)
    if row_count == 0:
        return result
    for column in range(column_count):
        order = np.argsort(values[:, column], kind="mergesort")
        ordered_values = values[order, column]
        start = 0
        while start < row_count:
            stop = start + 1
            while stop < row_count and ordered_values[stop] == ordered_values[start]:
                stop += 1
            mean_one_based_rank = (start + 1 + stop) * 0.5
            result[order[start:stop], column] = mean_one_based_rank / row_count
            start = stop
    return result


def direct_semantic_scores(
    candidates: Sequence[Candidate],
    constraints: Sequence[Constraint],
    event_embeddings: np.ndarray,
    constraint_embeddings: np.ndarray,
) -> np.ndarray:
    events = _matrix(event_embeddings, "event_embeddings")
    queries = _matrix(constraint_embeddings, "constraint_embeddings")
    if events.shape[0] != len(candidates) or queries.shape[0] != len(constraints):
        raise ValueError("embedding rows do not match candidates or constraints")
    return cosine_matrix(events, queries).astype(np.float32, copy=False)


def fuse_scores(
    cross_scores: ScoreMap,
    direct_scores: ScoreMap,
    *,
    cross_weight: float,
) -> dict[str, np.ndarray]:
    if isinstance(cross_weight, bool) or not isinstance(cross_weight, Real):
        raise ValueError("cross_weight must be in [0, 1]")
    weight = float(cross_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("cross_weight must be in [0, 1]")
    if set(cross_scores) != set(direct_scores):
        raise ValueError("cross and direct score entity IDs must match")
    output: dict[str, np.ndarray] = {}
    for entity_id in cross_scores:
        cross = _matrix(cross_scores[entity_id], "cross scores")
        direct = _matrix(direct_scores[entity_id], "direct scores")
        if cross.shape != direct.shape:
            raise ValueError(f"score shape mismatch for {entity_id}")
        output[entity_id] = (
            weight * percentile_ranks(cross)
            + (1.0 - weight) * percentile_ranks(direct)
        ).astype(np.float32, copy=False)
    return output


def fusion_weight_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    values = (float(start), float(stop), float(step))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("fusion grid values must be finite")
    if step <= 0.0 or start < 0.0 or stop > 1.0 or stop < start:
        raise ValueError("invalid fusion weight grid")
    count = int(math.floor((stop - start) / step + 1.0e-9))
    grid = tuple(round(start + index * step, 10) for index in range(count + 1))
    if not grid or grid[-1] < stop - 1.0e-8:
        grid = (*grid, round(stop, 10))
    return grid


def development_key(
    metrics: Mapping[str, float], *, epoch: int, cross_weight: float
) -> tuple[float, float, float, float]:
    names = ("rouge1_f1", "rouge2_f1", "date_f1")
    values = tuple(float(metrics[name]) for name in names)
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("development F1 values must be finite and non-negative")
    geometric = math.prod(max(value, 1.0e-12) for value in values) ** (1.0 / 3.0)
    return geometric, sum(values) / len(values), -float(epoch), -float(cross_weight)


@dataclass(frozen=True, slots=True)
class FusionSelection:
    cross_weight: float
    direct_weight: float
    metrics: Mapping[str, float]
    selection_key: tuple[float, float, float, float]


def select_fusion_weight(
    cross_scores: ScoreMap,
    direct_scores: ScoreMap,
    *,
    weights: Sequence[float],
    epoch: int,
    evaluator: Callable[[Mapping[str, np.ndarray]], Mapping[str, float]],
) -> FusionSelection:
    best: FusionSelection | None = None
    for weight in weights:
        metrics = {name: float(value) for name, value in evaluator(
            fuse_scores(cross_scores, direct_scores, cross_weight=float(weight))
        ).items()}
        key = development_key(metrics, epoch=epoch, cross_weight=float(weight))
        trial = FusionSelection(float(weight), 1.0 - float(weight), metrics, key)
        if best is None or trial.selection_key > best.selection_key:
            best = trial
    if best is None:
        raise ValueError("fusion weight grid must not be empty")
    return best


__all__ = [
    "FusionSelection",
    "development_key",
    "direct_semantic_scores",
    "fuse_scores",
    "fusion_weight_grid",
    "percentile_ranks",
    "select_fusion_weight",
]
