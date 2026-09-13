from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from math import ceil, exp, isclose, isfinite
from pathlib import Path
import json
import random
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .encoders import TextEncoder
from .io import write_json, write_jsonl
from .schema import Article, Constraint, Event, ReferenceEvent
from .support_verifier import SupportVerifier
from .text import cosine_matrix, normalize_text, stable_id, word_f1, word_tokens


if TYPE_CHECKING:
    from .data import DatasetReader


SYSTEM_PROMPT = (
    "Extract one shared assignment-free set of atomic events for all requested "
    "timelines. Write concise timeline-style events and return JSON only."
)


@dataclass(frozen=True, slots=True)
class GoldGroup:
    entity_id: str
    group_id: str
    event_date: str
    summary: str
    reference_ids: tuple[str, ...]


def _bounded_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number in [0, 1]")
    resolved = float(value)
    if not isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be a real number in [0, 1]")
    return resolved


def _constraints(values: Sequence[Constraint], entity_id: str) -> tuple[Constraint, ...]:
    resolved = tuple(values)
    if len(resolved) != 5:
        raise ValueError(f"entity {entity_id!r} must have exactly five constraints")
    if any(item.entity_id != entity_id for item in resolved):
        raise ValueError("constraint entity does not match its owner")
    if len({item.constraint_id for item in resolved}) != len(resolved):
        raise ValueError(f"entity {entity_id!r} has duplicate constraint IDs")
    return tuple(sorted(resolved, key=lambda item: item.constraint_id))


def _record_seed(seed: int, entity_id: str, article_id: str) -> int:
    raw = f"{int(seed)}\0{entity_id}\0{article_id}".encode("utf-8")
    return int.from_bytes(sha256(raw).digest()[:8], "big")


def joint_prompt(
    article: Article,
    constraints: Sequence[Constraint],
    *,
    seed: int,
    require_explicit_target_name: bool,
) -> str:
    """Render the shared full-document, five-constraint extraction prompt."""

    if type(require_explicit_target_name) is not bool:
        raise ValueError("require_explicit_target_name must be a boolean")
    values = list(_constraints(constraints, article.entity_id))
    random.Random(_record_seed(seed, article.entity_id, article.article_id)).shuffle(values)
    rendered_constraints = "\n".join(f"- {item.text}" for item in values)
    target_requirement = (
        ", explicitly name the target entity," if require_explicit_target_name else ""
    )
    return (
        "For this article, perform exactly one joint extraction pass over the complete "
        "document and the entire requested timeline set. Produce one shared, "
        "assignment-free event set; do not run separate extractions for individual "
        "requests.\n\n"
        "Return only a JSON array. Every array element must have exactly these two "
        'keys: {"date":"YYYY-MM-DD","event_summary":"..."}. Each event must be an '
        f"atomic fact written in concise timeline-style language{target_requirement} "
        "and contain no fact that the document does not state.\n\n"
        f"Target entity: {article.entity_id}\n\n"
        "Requested timelines (the complete set; order has no meaning):\n"
        f"{rendered_constraints}\n\n"
        f"Document date: {article.published_at or 'unknown'}\n"
        f"Title:\n{article.title}\n\n"
        f"Document:\n{article.text}"
    )


def assistant_target(events: Sequence[Event | GoldGroup]) -> str:
    ordered = sorted(
        events,
        key=lambda item: (
            item.event_date,
            item.summary.casefold(),
            item.summary,
            item.event_id if isinstance(item, Event) else item.group_id,
        ),
    )
    return json.dumps(
        [
            {"date": item.event_date, "event_summary": item.summary}
            for item in ordered
        ],
        ensure_ascii=False,
    )


def joint_messages(
    article: Article,
    constraints: Sequence[Constraint],
    *,
    seed: int,
    require_explicit_target_name: bool,
    targets: Sequence[Event | GoldGroup] | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": joint_prompt(
                article,
                constraints,
                seed=seed,
                require_explicit_target_name=require_explicit_target_name,
            ),
        },
    ]
    if targets is not None:
        messages.append({"role": "assistant", "content": assistant_target(targets)})
    return messages


def _complete_link(
    compatible: np.ndarray,
    affinity: np.ndarray,
) -> list[list[int]]:
    clusters = [[index] for index in range(int(compatible.shape[0]))]
    while True:
        selected: tuple[int, int] | None = None
        selected_score = -float("inf")
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                cross = np.ix_(clusters[left_index], clusters[right_index])
                if not bool(compatible[cross].all()):
                    continue
                # Complete-link selects the merge with the strongest weakest
                # cross-cluster edge.  The medoid below deliberately remains a
                # mean-affinity center.
                score = float(affinity[cross].min())
                if selected is None or score > selected_score:
                    selected = left_index, right_index
                    selected_score = score
        if selected is None:
            return clusters
        left_index, right_index = selected
        clusters[left_index] = sorted(clusters[left_index] + clusters[right_index])
        del clusters[right_index]


def deduplicate_references(
    entity_id: str,
    references: Sequence[ReferenceEvent],
    encoder: TextEncoder,
    *,
    semantic_threshold: float,
    word_f1_threshold: float,
    semantic_weight: float,
) -> tuple[GoldGroup, ...]:
    """Merge compatible same-day reference variants across all constraints."""

    semantic_cutoff = _bounded_probability(semantic_threshold, "semantic_threshold")
    lexical_cutoff = _bounded_probability(word_f1_threshold, "word_f1_threshold")
    weight = _bounded_probability(semantic_weight, "semantic_weight")
    ordered = sorted(
        references,
        key=lambda item: (
            item.event_date,
            item.summary.casefold(),
            item.summary,
            item.constraint_id,
            item.event_id,
        ),
    )
    if any(item.entity_id != entity_id for item in ordered):
        raise ValueError("reference entity does not match its owner")

    groups: list[GoldGroup] = []
    by_date: dict[str, list[ReferenceEvent]] = defaultdict(list)
    for item in ordered:
        by_date[item.event_date].append(item)
    for event_date in sorted(by_date):
        rows = by_date[event_date]
        embeddings = np.asarray(
            encoder.encode([item.summary for item in rows]), dtype=np.float32
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
            raise ValueError("encoder returned an invalid reference embedding matrix")
        semantic = cosine_matrix(embeddings)
        lexical = np.asarray(
            [
                [word_f1(left.summary, right.summary) for right in rows]
                for left in rows
            ],
            dtype=np.float32,
        )
        compatible = (semantic >= semantic_cutoff) | (lexical >= lexical_cutoff)
        affinity = weight * semantic + (1.0 - weight) * lexical
        for cluster in _complete_link(compatible, affinity):
            medoid = max(
                cluster,
                key=lambda index: (
                    float(affinity[index, cluster].mean()),
                    -len(word_tokens(rows[index].summary)),
                    rows[index].summary.casefold(),
                    rows[index].event_id,
                ),
            )
            members = tuple(rows[index] for index in cluster)
            reference_ids = tuple(
                sorted(
                    f"{item.constraint_id}:{item.event_id}" for item in members
                )
            )
            group_id = stable_id(
                "gold_",
                entity_id,
                event_date,
                reference_ids,
                tuple(sorted(item.summary for item in members)),
            )
            groups.append(
                GoldGroup(
                    entity_id=entity_id,
                    group_id=group_id,
                    event_date=event_date,
                    summary=rows[medoid].summary,
                    reference_ids=reference_ids,
                )
            )
    return tuple(
        sorted(
            groups,
            key=lambda item: (
                item.event_date,
                item.summary.casefold(),
                item.summary,
                item.group_id,
            ),
        )
    )


def retrieve_supporting_articles(
    groups: Sequence[GoldGroup],
    articles: Sequence[Article],
    encoder: TextEncoder,
    *,
    semantic_weight: float = 0.65,
    word_f1_weight: float = 0.25,
    temporal_weight: float = 0.10,
) -> tuple[dict[str, tuple[GoldGroup, ...]], list[dict[str, Any]]]:
    """Assign the five highest-scoring same-topic articles to each reference."""

    semantic_weight = _bounded_probability(semantic_weight, "retrieval_semantic_weight")
    word_f1_weight = _bounded_probability(word_f1_weight, "retrieval_word_f1_weight")
    temporal_weight = _bounded_probability(temporal_weight, "retrieval_temporal_weight")
    if not isclose(semantic_weight + word_f1_weight + temporal_weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("retrieval weights must sum to one")
    ordered_groups = tuple(groups)
    ordered_articles = tuple(sorted(articles, key=lambda item: item.article_id))
    if not ordered_groups or not ordered_articles:
        return {}, []
    entity_id = ordered_groups[0].entity_id
    if any(item.entity_id != entity_id for item in ordered_groups + ordered_articles):
        raise ValueError("retrieval accepts one entity at a time")
    article_texts = [normalize_text(f"{item.title} {item.text}") for item in ordered_articles]
    group_vectors = np.asarray(
        encoder.encode([item.summary for item in ordered_groups]), dtype=np.float32
    )
    if group_vectors.ndim != 2 or group_vectors.shape[0] != len(ordered_groups):
        raise ValueError("encoder returned an invalid event embedding matrix")
    article_vectors = np.asarray(encoder.encode(article_texts), dtype=np.float32)
    if article_vectors.ndim != 2 or article_vectors.shape[0] != len(ordered_articles):
        raise ValueError("encoder returned an invalid article embedding matrix")
    semantic = cosine_matrix(group_vectors, article_vectors)
    article_dates = [
        date.fromisoformat(item.published_at) if item.published_at else None
        for item in ordered_articles
    ]
    by_article: dict[str, dict[str, GoldGroup]] = defaultdict(dict)
    alignments: list[dict[str, Any]] = []
    for group_index, group in enumerate(ordered_groups):
        event_date = date.fromisoformat(group.event_date)
        ranked = []
        for article_index, article in enumerate(ordered_articles):
            semantic_score = float(semantic[group_index, article_index])
            lexical_score = word_f1(group.summary, article_texts[article_index])
            article_date = article_dates[article_index]
            temporal_score = (
                exp(-abs((event_date - article_date).days) / 7.0)
                if article_date is not None else 0.0
            )
            score = (
                semantic_weight * semantic_score
                + word_f1_weight * lexical_score
                + temporal_weight * temporal_score
            )
            ranked.append((score, article_index, semantic_score, lexical_score, temporal_score))
        ranked.sort(key=lambda item: (-item[0], ordered_articles[item[1]].article_id))
        for score, article_index, semantic_score, lexical_score, temporal_score in ranked[:5]:
            article = ordered_articles[article_index]
            by_article[article.article_id][group.group_id] = group
            alignments.append(
                {
                    "entity_id": entity_id,
                    "article_id": article.article_id,
                    "group_id": group.group_id,
                    "reference_ids": list(group.reference_ids),
                    "article_date": article.published_at,
                    "event_date": group.event_date,
                    "event_summary": group.summary,
                    "score": score,
                    "semantic_score": semantic_score,
                    "word_f1": lexical_score,
                    "temporal_score": temporal_score,
                }
            )
    resolved = {
        article_id: tuple(
            sorted(
                values.values(),
                key=lambda item: (
                    item.event_date,
                    item.summary.casefold(),
                    item.group_id,
                ),
            )
        )
        for article_id, values in by_article.items()
    }
    alignments.sort(
        key=lambda row: (str(row["entity_id"]), str(row["group_id"]), str(row["article_id"]))
    )
    return resolved, alignments


def original_reference_targets(
    entity_id: str, references: Sequence[ReferenceEvent]
) -> tuple[GoldGroup, ...]:
    """Retain original summaries; coalesce only identical date/text outputs.

    Identical events repeated under different constraints share an output but
    keep every source reference ID. No similarity clustering or medoid is used.
    """
    members: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in references:
        if item.entity_id != entity_id:
            raise ValueError("reference entity does not match its owner")
        members[(item.event_date, item.summary)].add(
            f"{item.constraint_id}:{item.event_id}"
        )
    return tuple(
        GoldGroup(
            entity_id=entity_id,
            group_id=stable_id("reference_", entity_id, event_date, summary),
            event_date=event_date,
            summary=summary,
            reference_ids=tuple(sorted(reference_ids)),
        )
        for (event_date, summary), reference_ids in sorted(members.items())
    )


def _cross_topic_empty_records(
    entity_values: Sequence[str],
    articles: Mapping[str, Sequence[Article]],
    constraints: Mapping[str, Sequence[Constraint]],
    groups_by_entity: Mapping[str, Sequence[GoldGroup]],
    by_article: Mapping[tuple[str, str], Sequence[GoldGroup]],
    *,
    requested: int,
    seed: int,
    require_name: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Sample other-topic articles, screening obvious topic overlap.

    This is a synthetic negative heuristic, not proof of semantic irrelevance.
    Reject target-name mentions, articles also present under the target topic,
    and source alignments sharing an exact reference event with that topic.
    """
    if requested == 0:
        return [], 0
    article_texts = {
        (entity_id, article.article_id): normalize_text(
            f"{article.title} {article.text}"
        ).casefold()
        for entity_id in entity_values
        for article in articles[entity_id]
    }
    article_tokens = {key: set(word_tokens(text)) for key, text in article_texts.items()}
    topic_documents = {
        entity_id: {article_texts[(entity_id, item.article_id)] for item in articles[entity_id]}
        for entity_id in entity_values
    }
    topic_events = {
        entity_id: {(item.event_date, item.summary) for item in groups_by_entity[entity_id]}
        for entity_id in entity_values
    }
    rng = random.Random(seed)
    sampled: list[tuple[str, Article]] = []
    candidate_count = 0
    for target_entity_id in entity_values:
        target_name = normalize_text(target_entity_id.replace("_", " ")).casefold()
        target_tokens = word_tokens(target_name)
        for source_entity_id in entity_values:
            if source_entity_id == target_entity_id:
                continue
            for source in sorted(articles[source_entity_id], key=lambda item: item.article_id):
                key = source_entity_id, source.article_id
                searchable = article_texts[key]
                if not target_tokens or target_name in searchable:
                    continue
                if len(target_tokens[-1]) >= 3 and target_tokens[-1] in article_tokens[key]:
                    continue
                if searchable in topic_documents[target_entity_id]:
                    continue
                if any(
                    (item.event_date, item.summary) in topic_events[target_entity_id]
                    for item in by_article.get(key, ())
                ):
                    continue
                candidate_count += 1
                # Uniform reservoir sampling avoids materializing all topic pairs.
                if len(sampled) < requested:
                    sampled.append((target_entity_id, source))
                else:
                    position = rng.randrange(candidate_count)
                    if position < requested:
                        sampled[position] = target_entity_id, source
    records = []
    for target_entity_id, source in sampled:
        synthetic_id = f"negative::{source.entity_id}::{source.article_id}"
        prompt_article = Article(
            target_entity_id, synthetic_id, source.published_at, source.title, source.text
        )
        records.append({
            "entity_id": target_entity_id,
            "article_id": synthetic_id,
            "article_date": source.published_at,
            "source_entity_id": source.entity_id,
            "source_article_id": source.article_id,
            "supervision": "cross_topic_empty",
            "event_ids": [],
            "reference_ids": [],
            "messages": joint_messages(
                prompt_article, constraints[target_entity_id], seed=seed,
                require_explicit_target_name=require_name, targets=(),
            ),
        })
    return records, candidate_count


def build_aligned_gold_records(
    reader: "DatasetReader",
    entity_ids: Sequence[str],
    encoder: TextEncoder,
    stage1: Mapping[str, Any],
    *,
    verifier: SupportVerifier,
    dataset_name: str,
    seed: int,
) -> dict[str, Any]:
    """Align original references to the highest-scoring same-topic articles."""

    if dataset_name not in {"crest", "wcep_ctg"}:
        raise ValueError("dataset_name must be crest or wcep_ctg")
    if not callable(getattr(verifier, "supports", None)):
        raise ValueError("Stage-1 preparation requires a factual-support verifier")
    retrieval_weights = {
        "semantic_weight": stage1.get("retrieval_semantic_weight", 0.65),
        "word_f1_weight": stage1.get("retrieval_word_f1_weight", 0.25),
        "temporal_weight": stage1.get("retrieval_temporal_weight", 0.10),
    }
    entity_values = tuple(sorted(entity_ids))
    constraints = reader.constraints_for(entity_values)
    articles = reader.articles_for(entity_values)
    references = reader.references_for(entity_values)
    require_name = stage1.get("require_explicit_target_name", True)
    if type(require_name) is not bool:
        raise ValueError("stage1.require_explicit_target_name must be a boolean")

    groups_by_entity: dict[str, tuple[GoldGroup, ...]] = {}
    original_references: list[ReferenceEvent] = []
    by_article: dict[tuple[str, str], tuple[GoldGroup, ...]] = {}
    retrieved_by_article: dict[tuple[str, str], tuple[GoldGroup, ...]] = {}
    alignment_rows: list[dict[str, Any]] = []
    support_checks: list[dict[str, Any]] = []
    for entity_id in entity_values:
        entity_references = tuple(
            item
            for constraint in _constraints(constraints[entity_id], entity_id)
            for item in references.get((entity_id, constraint.constraint_id), ())
        )
        original_references.extend(entity_references)
        groups = original_reference_targets(entity_id, entity_references)
        groups_by_entity[entity_id] = groups
        retrieved, rows = retrieve_supporting_articles(
            groups,
            articles[entity_id],
            encoder,
            **retrieval_weights,
        )
        retrieved_by_article.update(
            {(entity_id, article_id): values for article_id, values in retrieved.items()}
        )
        topic_groups = {group.group_id: group for group in groups}
        topic_articles = {article.article_id: article for article in articles[entity_id]}
        accepted: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            group = topic_groups[str(row["group_id"])]
            article = topic_articles[str(row["article_id"])]
            context = f"{article.title}\n\n{article.text}".strip()
            supported = verifier.supports(group.summary, context) is True
            support_checks.append({**row, "supported": supported})
            if supported:
                accepted[article.article_id].add(group.group_id)
                alignment_rows.append(row)
        by_article.update({
            (entity_id, article_id): tuple(
                group for group in retrieved[article_id] if group.group_id in group_ids
            )
            for article_id, group_ids in accepted.items()
        })

    article_lookup = {
        (entity_id, article.article_id): article
        for entity_id in entity_values
        for article in articles[entity_id]
    }
    records: list[dict[str, Any]] = []
    for key in sorted(by_article):
        entity_id, article_id = key
        target_groups = by_article[key]
        records.append(
            {
                "entity_id": entity_id,
                "article_id": article_id,
                "article_date": article_lookup[key].published_at,
                "supervision": "aligned_retrieved_gold",
                "event_ids": [item.group_id for item in target_groups],
                "reference_ids": sorted({
                    reference_id for item in target_groups
                    for reference_id in item.reference_ids
                }),
                "messages": joint_messages(
                    article_lookup[key],
                    constraints[entity_id],
                    seed=seed,
                    require_explicit_target_name=require_name,
                    targets=target_groups,
                ),
            }
        )

    positive_count = len(records)
    empty_ratio = _bounded_probability(
        stage1.get("empty_article_ratio", 0.05), "empty_article_ratio"
    )
    requested_empty = ceil(positive_count * empty_ratio)
    empty_records, empty_candidates = _cross_topic_empty_records(
        entity_values, articles, constraints, groups_by_entity, retrieved_by_article,
        requested=requested_empty, seed=seed, require_name=require_name,
    )
    records.extend(empty_records)
    records.sort(key=lambda row: (str(row["entity_id"]), str(row["article_id"])))
    articles_per_group = Counter(
        (row["entity_id"], row["group_id"]) for row in alignment_rows
    )
    all_groups = tuple(group for values in groups_by_entity.values() for group in values)
    unmatched = tuple(
        group for group in all_groups
        if not articles_per_group[(group.entity_id, group.group_id)]
    )
    article_counts = {
        entity_id: len(articles[entity_id])
        for entity_id in entity_values
    }

    def count_distribution(values: Sequence[int]) -> dict[str, int | float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    summary = {
        "dataset": dataset_name,
        "train_entities": list(entity_values),
        "training_records": len(records),
        "positive_records": positive_count,
        "empty_records": len(empty_records),
        "empty_sampling": {
            "ratio_to_positive": empty_ratio,
            "requested": requested_empty,
            "eligible_candidates": empty_candidates,
            "screening": "target_name_shared_document_and_shared_reference_exclusion",
            "semantic_irrelevance_verified": False,
        },
        "input_articles": len(article_lookup),
        "articles_without_date": sum(not item.published_at for item in article_lookup.values()),
        "reference_events": len(original_references),
        "unique_reference_targets": len(all_groups),
        "matched_reference_targets": len(all_groups) - len(unmatched),
        "unmatched_reference_targets": len(unmatched),
        "support_verification": {
            "context": "full_article",
            "checked_pairs": len(support_checks),
            "supported_pairs": len(alignment_rows),
            "rejected_pairs": len(support_checks) - len(alignment_rows),
            "resample_after_rejection": False,
        },
        "references_without_articles": sum(
            not article_counts[item.entity_id] for item in all_groups
        ),
        "candidate_pairs": sum(
            article_counts[item.entity_id] for item in all_groups
        ),
        "reference_preprocessing": "exact_date_text_duplicates_only",
        "semantic_reference_clustering": False,
        "gold_event_groups": len(all_groups),
        "matched_gold_event_groups": len(all_groups) - len(unmatched),
        "unmatched_gold_event_groups": len(unmatched),
        "unmatched_groups": [
            {
                "entity_id": group.entity_id,
                "group_id": group.group_id,
                "date": group.event_date,
                "summary": group.summary,
                "reference_ids": list(group.reference_ids),
                "reason": (
                    "no_verified_support" if article_counts[group.entity_id]
                    else "no_article"
                ),
            }
            for group in unmatched
        ],
        "alignment_records": len(alignment_rows),
        "articles_per_event": count_distribution([
            articles_per_group[(group.entity_id, group.group_id)] for group in all_groups
        ]),
        "events_per_article": count_distribution([
            len(row["event_ids"]) for row in records if row["supervision"] == "aligned_retrieved_gold"
        ]),
        "retrieval": {
            "same_topic_only": True,
            "same_date_only": False,
            "article_date_field": "Article.published_at",
            "score_threshold": None,
            **retrieval_weights,
            "temporal_decay_days": 7.0,
            "article_semantic_aggregation": "single_vector",
            "article_limit_per_event": 5,
        },
        "input_scope": "full_article",
        "all_constraints_joint": True,
        "seed": int(seed),
    }
    return {
        "records": records, "alignments": alignment_rows, "summary": summary,
        "support_checks": support_checks,
        "references": [item.to_dict() for item in original_references],
    }


def prepare_stage1_records(
    config: Mapping[str, Any],
    reader: "DatasetReader",
    *,
    verifier: SupportVerifier,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    if not callable(getattr(verifier, "supports", None)):
        raise ValueError("Stage-1 preparation requires a factual-support verifier")
    stage1 = config.get("stage1")
    if not isinstance(stage1, Mapping):
        raise ValueError("configuration must contain a stage1 mapping")
    entity_ids = reader.entity_ids("train")
    seed = int(config.get("seed", 42))
    dataset_name = config.get("dataset")
    if dataset_name not in {"crest", "wcep_ctg"}:
        raise ValueError("dataset must be 'crest' or 'wcep_ctg'")
    if encoder is None:
        raise ValueError("Stage-1 preparation requires a text encoder")
    return build_aligned_gold_records(
        reader,
        entity_ids,
        encoder,
        stage1,
        verifier=verifier,
        dataset_name=str(dataset_name),
        seed=seed,
    )


def write_stage1_artifacts(output_dir: str | Path, artifacts: Mapping[str, Any]) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", artifacts["records"])
    write_jsonl(output / "alignments.jsonl", artifacts.get("alignments", ()))
    write_jsonl(output / "support_checks.jsonl", artifacts.get("support_checks", ()))
    write_jsonl(output / "references.jsonl", artifacts.get("references", ()))
    write_json(output / "summary.json", artifacts["summary"])


__all__ = [
    "GoldGroup",
    "SYSTEM_PROMPT",
    "assistant_target",
    "build_aligned_gold_records",
    "deduplicate_references",
    "joint_messages",
    "joint_prompt",
    "original_reference_targets",
    "prepare_stage1_records",
    "retrieve_supporting_articles",
    "write_stage1_artifacts",
]
