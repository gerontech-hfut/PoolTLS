from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from .data import DatasetReader
from .io import iter_jsonl, write_jsonl
from .schema import ReferenceEvent, TimelineEvent
from .text import normalize_text


PredictionMap = Mapping[tuple[str, str], Sequence[TimelineEvent]]
ReferenceMap = Mapping[tuple[str, str], Sequence[ReferenceEvent]]


def _event_dict(event: TimelineEvent) -> dict[str, object]:
    if not isinstance(event, TimelineEvent):
        raise ValueError("prediction values must contain TimelineEvent records")
    return {
        "candidate_id": event.candidate_id,
        "date": event.event_date,
        "summary": event.summary,
        "score": float(event.score),
    }


def write_predictions(path: str | Path, predictions: PredictionMap) -> None:
    rows = []
    for entity_id, constraint_id in sorted(predictions):
        events = sorted(
            predictions[(entity_id, constraint_id)],
            key=lambda event: (event.event_date, -float(event.score), event.candidate_id),
        )
        rows.append(
            {
                "entity_id": entity_id,
                "constraint_id": constraint_id,
                "events": [_event_dict(event) for event in events],
            }
        )
    write_jsonl(path, rows)


def read_predictions(path: str | Path) -> dict[tuple[str, str], tuple[TimelineEvent, ...]]:
    output: dict[tuple[str, str], tuple[TimelineEvent, ...]] = {}
    for row in iter_jsonl(path):
        entity_id = normalize_text(row.get("entity_id"))
        constraint_id = normalize_text(row.get("constraint_id"))
        raw_events = row.get("events")
        if not entity_id or not constraint_id or not isinstance(raw_events, list):
            raise ValueError("prediction row requires entity_id, constraint_id, and events")
        key = (entity_id, constraint_id)
        if key in output:
            raise ValueError(f"duplicate prediction row for {entity_id}/{constraint_id}")
        events: list[TimelineEvent] = []
        for raw in raw_events:
            if not isinstance(raw, Mapping):
                raise ValueError(f"prediction event must be an object for {entity_id}/{constraint_id}")
            events.append(
                TimelineEvent(
                    entity_id=entity_id,
                    constraint_id=constraint_id,
                    candidate_id=normalize_text(raw.get("candidate_id")),
                    event_date=normalize_text(raw.get("date", raw.get("event_date"))),
                    summary=normalize_text(raw.get("summary")),
                    score=float(raw.get("score", 0.0)),
                )
            )
        output[key] = tuple(
            sorted(
                events,
                key=lambda event: (
                    event.event_date,
                    -float(event.score),
                    event.candidate_id,
                ),
            )
        )
    return output


def write_crest_timelines(root: str | Path, predictions: PredictionMap) -> None:
    """Also export predictions in the dataset's one-file-per-timeline shape."""

    destination = Path(root)
    for (entity_id, constraint_id), events in sorted(predictions.items()):
        by_date: dict[str, list[str]] = defaultdict(list)
        for event in sorted(
            events,
            key=lambda item: (item.event_date, -float(item.score), item.candidate_id),
        ):
            by_date[event.event_date].append(event.summary)
        timeline = [
            [f"{event_date} 00:00:00", summaries]
            for event_date, summaries in sorted(by_date.items())
        ]
        path = destination / entity_id / constraint_id / "timelines.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(timeline, ensure_ascii=False) + "\n", encoding="utf-8")


def _harmonic(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _as_date(value: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError as error:
        raise ValueError(f"invalid event date: {value!r}") from error


def _tilse_timeline(events: Sequence[ReferenceEvent | TimelineEvent], timeline_type: Any):
    by_date: dict[date, list[str]] = defaultdict(list)
    for event in events:
        summary = normalize_text(event.summary)
        if summary:
            by_date[_as_date(event.event_date)].append(summary)
    return timeline_type(dict(by_date))


def evaluate_predictions(
    predictions: PredictionMap,
    references: ReferenceMap,
) -> dict[str, float | int]:
    """Compute the paper's macro TILSE content and unique-date scores."""

    expected = set(references)
    actual = set(predictions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"prediction keys mismatch: missing={missing}; extra={extra}")
    try:
        from tilse.data.timelines import GroundTruth, Timeline
        from tilse.evaluation.rouge import TimelineRougeEvaluator
    except ImportError as error:
        raise RuntimeError(
            "TILSE is required for evaluation; install the dependency from requirements.txt"
        ) from error

    evaluator = TimelineRougeEvaluator(
        measures=["rouge_1", "rouge_2"], rouge_computation="reimpl"
    )
    totals = {
        "rouge1_precision": 0.0,
        "rouge1_recall": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "date_precision": 0.0,
        "date_recall": 0.0,
    }
    evaluated = 0
    for key in sorted(expected):
        gold_events = tuple(references[key])
        if not gold_events:
            continue
        selected = tuple(predictions[key])
        evaluated += 1
        prediction = _tilse_timeline(selected, Timeline)
        reference = _tilse_timeline(gold_events, Timeline)
        ground_truth = GroundTruth([reference])
        if selected:
            rouge = evaluator.evaluate_align_date_content_costs_many_to_one(
                prediction, ground_truth
            )
            predicted_dates = prediction.get_dates()
            reference_dates = ground_truth.get_dates()
            shared = len(predicted_dates.intersection(reference_dates))
            date_precision = shared / len(predicted_dates) if predicted_dates else 0.0
            date_recall = shared / len(reference_dates) if reference_dates else 0.0
        else:
            rouge = {
                "rouge_1": {"precision": 0.0, "recall": 0.0},
                "rouge_2": {"precision": 0.0, "recall": 0.0},
            }
            date_precision = 0.0
            date_recall = 0.0
        totals["rouge1_precision"] += float(rouge["rouge_1"]["precision"])
        totals["rouge1_recall"] += float(rouge["rouge_1"]["recall"])
        totals["rouge2_precision"] += float(rouge["rouge_2"]["precision"])
        totals["rouge2_recall"] += float(rouge["rouge_2"]["recall"])
        totals["date_precision"] += date_precision
        totals["date_recall"] += date_recall
    if evaluated == 0:
        raise ValueError("no non-empty reference timelines")

    means = {name: value / evaluated for name, value in totals.items()}
    return {
        "timeline_count": evaluated,
        "rouge1_precision": means["rouge1_precision"],
        "rouge1_recall": means["rouge1_recall"],
        "rouge1_f1": _harmonic(means["rouge1_precision"], means["rouge1_recall"]),
        "rouge2_precision": means["rouge2_precision"],
        "rouge2_recall": means["rouge2_recall"],
        "rouge2_f1": _harmonic(means["rouge2_precision"], means["rouge2_recall"]),
        "date_precision": means["date_precision"],
        "date_recall": means["date_recall"],
        "date_f1": _harmonic(means["date_precision"], means["date_recall"]),
    }


def evaluate_dataset(
    predictions: PredictionMap,
    reader: DatasetReader,
    partition: str = "development",
) -> dict[str, float | int]:
    entity_ids = reader.entity_ids(partition)
    return evaluate_predictions(predictions, reader.references_for(entity_ids))


__all__ = [
    "evaluate_dataset",
    "evaluate_predictions",
    "read_predictions",
    "write_crest_timelines",
    "write_predictions",
]
