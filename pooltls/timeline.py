from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .schema import Candidate, Constraint, TimelineEvent


def build_timelines(
    candidates_by_entity: Mapping[str, Sequence[Candidate]],
    constraints_by_entity: Mapping[str, Sequence[Constraint]],
    scores_by_entity: Mapping[str, np.ndarray],
    budgets: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], tuple[TimelineEvent, ...]]:
    """Independently take each constraint's feasible budget, then sort by date."""

    entity_ids = set(candidates_by_entity)
    if set(constraints_by_entity) != entity_ids or set(scores_by_entity) != entity_ids:
        raise ValueError("candidate, constraint, and score entity IDs must match")
    output: dict[tuple[str, str], tuple[TimelineEvent, ...]] = {}
    for entity_id in sorted(entity_ids):
        candidates = tuple(candidates_by_entity[entity_id])
        constraints = tuple(constraints_by_entity[entity_id])
        scores = np.asarray(scores_by_entity[entity_id], dtype=np.float32)
        if scores.shape != (len(candidates), len(constraints)):
            raise ValueError(f"invalid score shape for {entity_id}")
        if not np.isfinite(scores).all():
            raise ValueError(f"scores must be finite for {entity_id}")
        for column, constraint in enumerate(constraints):
            key = (entity_id, constraint.constraint_id)
            if key not in budgets or type(budgets[key]) is not int or budgets[key] < 0:
                raise ValueError(f"missing or invalid timeline budget for {key}")
            budget = min(budgets[key], len(candidates))
            ranked = sorted(
                range(len(candidates)),
                key=lambda row: (-float(scores[row, column]), candidates[row].candidate_id),
            )
            chosen = ranked[:budget]
            chosen.sort(
                key=lambda row: (
                    candidates[row].event_date,
                    -float(scores[row, column]),
                    candidates[row].candidate_id,
                )
            )
            output[key] = tuple(
                TimelineEvent(
                    entity_id=entity_id,
                    constraint_id=constraint.constraint_id,
                    candidate_id=candidates[row].candidate_id,
                    event_date=candidates[row].event_date,
                    summary=candidates[row].summary,
                    score=float(scores[row, column]),
                )
                for row in chosen
            )
    if set(output) != set(budgets):
        raise ValueError("budgets contain entities or constraints absent from inputs")
    return output


__all__ = ["build_timelines"]
