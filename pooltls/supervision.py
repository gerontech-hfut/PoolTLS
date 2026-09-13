from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .encoders import TextEncoder
from .schema import Candidate, Constraint, PairExample, ReferenceEvent
from .text import cosine_matrix, stable_id, word_f1


@dataclass(frozen=True, slots=True)
class SupervisionResult:
    examples: tuple[PairExample, ...]
    candidate_positive_count: int
    reference_positive_count: int
    reliable_negative_count: int


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _validate_entity(
    candidates: Sequence[Candidate],
    constraints: Sequence[Constraint],
    references: Mapping[str, Sequence[ReferenceEvent]],
) -> str:
    entity_ids = {
        *(candidate.entity_id for candidate in candidates),
        *(constraint.entity_id for constraint in constraints),
        *(event.entity_id for events in references.values() for event in events),
    }
    if len(entity_ids) != 1:
        raise ValueError("candidates, constraints, and references must share one entity")
    constraint_ids = tuple(constraint.constraint_id for constraint in constraints)
    if len(set(constraint_ids)) != len(constraint_ids):
        raise ValueError("constraint IDs must be unique")
    if set(references) != set(constraint_ids):
        raise ValueError("references must contain exactly the configured constraints")
    return next(iter(entity_ids))


def _score_matrix(
    candidates: Sequence[Candidate],
    references: Sequence[ReferenceEvent],
    encoder: TextEncoder,
    semantic_weight: float,
) -> np.ndarray:
    if not candidates or not references:
        return np.empty((len(candidates), len(references)), dtype=np.float32)
    candidate_vectors = np.asarray(
        encoder.encode(tuple(candidate.summary for candidate in candidates)),
        dtype=np.float32,
    )
    reference_vectors = np.asarray(
        encoder.encode(tuple(event.summary for event in references)), dtype=np.float32
    )
    semantic = cosine_matrix(candidate_vectors, reference_vectors)
    lexical = np.asarray(
        [
            [word_f1(candidate.summary, event.summary) for event in references]
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    return np.clip(
        semantic_weight * semantic + (1.0 - semantic_weight) * lexical,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)


def build_supervision(
    candidates: Sequence[Candidate],
    constraints: Sequence[Constraint],
    references: Mapping[str, Sequence[ReferenceEvent]],
    encoder: TextEncoder,
    *,
    semantic_weight: float,
    positive_threshold: float,
    negative_threshold: float,
    include_reference_positives: bool = False,
) -> SupervisionResult:
    """Build Hungarian positives and globally screened reliable negatives."""

    coefficient = _unit_interval(semantic_weight, "semantic_weight")
    positive_cutoff = _unit_interval(positive_threshold, "positive_threshold")
    negative_cutoff = _unit_interval(negative_threshold, "negative_threshold")
    if negative_cutoff > positive_cutoff:
        raise ValueError("negative_threshold must not exceed positive_threshold")
    candidate_values = tuple(candidates)
    constraint_values = tuple(constraints)
    reference_values = {key: tuple(value) for key, value in references.items()}
    entity_id = _validate_entity(candidate_values, constraint_values, reference_values)

    candidate_by_date: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidate_values):
        candidate_by_date[candidate.event_date].append(index)

    positive_scores: dict[tuple[int, str], float] = {}
    global_same_date_max = np.zeros(len(candidate_values), dtype=np.float32)
    for constraint in constraint_values:
        constraint_references = reference_values[constraint.constraint_id]
        reference_by_date: dict[str, list[int]] = defaultdict(list)
        for index, event in enumerate(constraint_references):
            reference_by_date[event.event_date].append(index)
        for event_date in sorted(candidate_by_date.keys() & reference_by_date.keys()):
            candidate_indices = candidate_by_date[event_date]
            reference_indices = reference_by_date[event_date]
            date_candidates = tuple(candidate_values[index] for index in candidate_indices)
            date_references = tuple(
                constraint_references[index] for index in reference_indices
            )
            scores = _score_matrix(
                date_candidates, date_references, encoder, coefficient
            )
            maxima = scores.max(axis=1)
            for local_index, value in enumerate(maxima.tolist()):
                global_index = candidate_indices[local_index]
                global_same_date_max[global_index] = max(
                    global_same_date_max[global_index], float(value)
                )
            matched_candidates, matched_references = linear_sum_assignment(
                scores, maximize=True
            )
            for candidate_index, reference_index in zip(
                matched_candidates.tolist(), matched_references.tolist(), strict=True
            ):
                score = float(scores[candidate_index, reference_index])
                if score > positive_cutoff:
                    positive_scores[
                        (candidate_indices[candidate_index], constraint.constraint_id)
                    ] = score

    examples: list[PairExample] = []
    constraint_by_id = {
        constraint.constraint_id: constraint for constraint in constraint_values
    }
    for (candidate_index, constraint_id), score in sorted(positive_scores.items()):
        candidate = candidate_values[candidate_index]
        constraint = constraint_by_id[constraint_id]
        examples.append(
            PairExample(
                entity_id=entity_id,
                candidate_id=candidate.candidate_id,
                constraint_id=constraint_id,
                constraint_text=constraint.text,
                event_date=candidate.event_date,
                event_summary=candidate.summary,
                label=1,
                matching_score=score,
            )
        )

    reliable_count = 0
    for candidate_index, candidate in enumerate(candidate_values):
        if float(global_same_date_max[candidate_index]) > negative_cutoff:
            continue
        for constraint in constraint_values:
            examples.append(
                PairExample(
                    entity_id=entity_id,
                    candidate_id=candidate.candidate_id,
                    constraint_id=constraint.constraint_id,
                    constraint_text=constraint.text,
                    event_date=candidate.event_date,
                    event_summary=candidate.summary,
                    label=0,
                    matching_score=float(global_same_date_max[candidate_index]),
                )
            )
            reliable_count += 1

    reference_positive_count = 0
    if include_reference_positives:
        for constraint in constraint_values:
            for event in reference_values[constraint.constraint_id]:
                examples.append(
                    PairExample(
                        entity_id=entity_id,
                        candidate_id=stable_id(
                            "ref_", entity_id, constraint.constraint_id, event.event_id
                        ),
                        constraint_id=constraint.constraint_id,
                        constraint_text=constraint.text,
                        event_date=event.event_date,
                        event_summary=event.summary,
                        label=1,
                        matching_score=1.0,
                    )
                )
                reference_positive_count += 1

    examples.sort(
        key=lambda item: (
            item.entity_id,
            item.constraint_id,
            -item.label,
            item.event_date,
            item.event_summary.casefold(),
            item.candidate_id,
        )
    )
    return SupervisionResult(
        examples=tuple(examples),
        candidate_positive_count=len(positive_scores),
        reference_positive_count=reference_positive_count,
        reliable_negative_count=reliable_count,
    )


__all__ = ["SupervisionResult", "build_supervision"]
