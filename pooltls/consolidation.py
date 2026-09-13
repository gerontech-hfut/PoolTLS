from __future__ import annotations

from collections import defaultdict
from numbers import Real
from typing import Sequence

import numpy as np

from .encoders import TextEncoder
from .schema import Candidate, Mention
from .text import cosine_matrix, word_f1


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _complete_link(compatible: np.ndarray, affinity: np.ndarray) -> list[list[int]]:
    """Greedily merge the most similar pair whose every cross-pair is compatible."""

    if compatible.ndim != 2 or compatible.shape[0] != compatible.shape[1]:
        raise ValueError("compatible must be a square matrix")
    if affinity.shape != compatible.shape:
        raise ValueError("affinity and compatibility shapes must match")
    clusters = [[index] for index in range(compatible.shape[0])]
    while True:
        best: tuple[float, int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                cross = np.ix_(clusters[left], clusters[right])
                if not bool(compatible[cross].all()):
                    continue
                # Complete-link uses the weakest cross-cluster edge.  Among
                # eligible pairs, merge the pair whose weakest edge is best.
                candidate = (float(affinity[cross].min()), -left, -right)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            return clusters
        left, right = -best[1], -best[2]
        clusters[left] = sorted((*clusters[left], *clusters[right]))
        del clusters[right]


def consolidate_mentions(
    mentions: Sequence[Mention],
    encoder: TextEncoder,
    *,
    semantic_threshold: float,
    word_f1_threshold: float,
    semantic_weight: float = 0.75,
) -> tuple[Candidate, ...]:
    """Create one shared candidate pool using same-date complete-link clustering."""

    semantic_cutoff = _unit_interval(semantic_threshold, "semantic_threshold")
    lexical_cutoff = _unit_interval(word_f1_threshold, "word_f1_threshold")
    semantic_coefficient = _unit_interval(semantic_weight, "semantic_weight")
    values = tuple(mentions)
    if not values:
        return ()
    entity_ids = {mention.entity_id for mention in values}
    if len(entity_ids) != 1:
        raise ValueError("mentions must belong to exactly one entity")
    mention_ids = tuple(mention.mention_id for mention in values)
    if len(set(mention_ids)) != len(mention_ids):
        raise ValueError("mention IDs must be unique within an entity")

    ordered = tuple(
        sorted(
            values,
            key=lambda item: (item.event_date, item.summary.casefold(), item.mention_id),
        )
    )
    embeddings = np.asarray(
        encoder.encode(tuple(mention.summary for mention in ordered)), dtype=np.float32
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ordered):
        raise ValueError("encoder returned an invalid mention embedding matrix")
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("encoder returned an invalid mention embedding matrix")

    by_date: dict[str, list[int]] = defaultdict(list)
    for index, mention in enumerate(ordered):
        by_date[mention.event_date].append(index)

    provisional: list[tuple[str, str, tuple[str, ...]]] = []
    for event_date in sorted(by_date):
        rows = tuple(ordered[index] for index in by_date[event_date])
        date_embeddings = embeddings[np.asarray(by_date[event_date], dtype=np.int64)]
        semantic = cosine_matrix(date_embeddings)
        lexical = np.asarray(
            [
                [word_f1(left.summary, right.summary) for right in rows]
                for left in rows
            ],
            dtype=np.float32,
        )
        compatible = (semantic >= semantic_cutoff) | (lexical >= lexical_cutoff)
        affinity = (
            semantic_coefficient * semantic
            + (1.0 - semantic_coefficient) * lexical
        )
        for cluster in _complete_link(compatible, affinity):
            medoid = max(
                cluster,
                key=lambda index: (
                    float(affinity[index, cluster].mean()),
                    len(rows[index].summary),
                    rows[index].mention_id,
                ),
            )
            provisional.append(
                (
                    event_date,
                    rows[medoid].summary,
                    tuple(sorted(rows[index].mention_id for index in cluster)),
                )
            )

    provisional.sort(key=lambda item: (item[0], item[1].casefold(), item[2]))
    entity_id = next(iter(entity_ids))
    return tuple(
        Candidate(
            entity_id=entity_id,
            candidate_id=f"c_{index:06d}",
            event_date=event_date,
            summary=summary,
            member_ids=member_ids,
        )
        for index, (event_date, summary, member_ids) in enumerate(provisional)
    )


__all__ = ["consolidate_mentions"]
