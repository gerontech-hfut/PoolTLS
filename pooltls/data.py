from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from .schema import Article, Constraint, ReferenceEvent
from .text import normalize_text


_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_CONSTRAINT_IDS = tuple(str(index) for index in range(5))
_DATE_KEYS = ("date", "event_date", "time")
_SUMMARY_KEYS = ("summaries", "summary", "event_summary", "events")
_PARTITION_ALIASES = {
    "train": "train",
    "training": "train",
    "development": "development",
    "dev": "development",
    "validation": "development",
    "test": "test",
}
_PARTITION_DIRECTORIES = {
    "train": "train",
    "development": "validation",
    "test": "test",
}


def _parse_date(value: object, *, name: str = "date") -> str:
    text = normalize_text(value)
    match = _ISO_DATE.match(text)
    if match is None:
        raise ValueError(f"invalid {name}: {value!r}")
    resolved = match.group(0)
    try:
        date.fromisoformat(resolved)
    except ValueError as error:
        raise ValueError(f"invalid {name}: {value!r}") from error
    return resolved


def _identifier(value: object) -> str:
    """Normalize an identifier without treating the integer zero as empty."""

    return "" if value is None else normalize_text(str(value))


def _json_lines(path: Path) -> Iterator[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def _first(raw: Mapping[object, object], names: Sequence[str]) -> object:
    for name in names:
        if name in raw:
            return raw[name]
    return None


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _timeline_items(raw: object) -> Iterator[tuple[object, object]]:
    """Yield date and summary values from CREST and common equivalent shapes."""

    if isinstance(raw, Mapping):
        raw_date = _first(raw, _DATE_KEYS)
        if raw_date is not None:
            yield raw_date, _first(raw, _SUMMARY_KEYS)
            return
        for wrapper in ("timeline", "events"):
            if wrapper in raw:
                yield from _timeline_items(raw[wrapper])
                return
        dated = [
            (key, value)
            for key, value in raw.items()
            if _ISO_DATE.search(normalize_text(key)) is not None
        ]
        if dated:
            yield from dated
            return
        raise ValueError("timeline record has an invalid object shape")

    if not _is_sequence(raw):
        raise ValueError("timeline record must be an array or object")
    if len(raw) >= 2 and not isinstance(raw[0], Mapping) and not _is_sequence(raw[0]):
        yield raw[0], raw[1]
        return
    for item in raw:
        yield from _timeline_items(item)


def _summaries(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: Sequence[object] = (value,)
    elif _is_sequence(value):
        raw_values = value
    else:
        raise ValueError("timeline summaries must be a string or an array of strings")
    if any(not isinstance(item, str) for item in raw_values):
        raise ValueError("timeline summaries must contain only strings")
    return tuple(text for item in raw_values if (text := normalize_text(item)))


class DatasetReader:
    """Read either configured dataset through one small CREST-compatible API."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")
        dataset = config.get("dataset")
        if dataset not in {"crest", "wcep_ctg"}:
            raise ValueError("dataset must be 'crest' or 'wcep_ctg'")
        paths = config.get("paths")
        folds = config.get("folds", {})
        if not isinstance(paths, Mapping):
            raise ValueError("config must contain a paths mapping")
        if not isinstance(folds, Mapping):
            raise ValueError("config.folds must be a mapping when provided")
        self.config = config
        self.dataset = str(dataset)
        self.paths = dict(paths)
        self.fold_config = dict(folds)
        self._partitioned_layout: bool | None = None
        self._entities_by_partition: dict[str, tuple[str, ...]] | None = None
        self._partition_by_entity: dict[str, str] | None = None
        self._fold_map: dict[str, int] | None = None
        self._constraint_map: dict[str, Mapping[str, object]] | None = None
        self._article_cache: dict[str, tuple[Article, ...]] = {}
        self._loaded_article_partitions: set[str] = set()
        self._reference_cache: dict[
            str, dict[str, tuple[ReferenceEvent, ...]]
        ] = {}
        self._loaded_reference_partitions: set[str] = set()

    def _path(self, name: str, *, required: bool = True) -> Path | None:
        value = self.paths.get(name)
        if value is None:
            if required:
                raise ValueError(f"paths.{name} is required")
            return None
        path = Path(value).expanduser().resolve()
        if required and not path.exists():
            raise FileNotFoundError(f"paths.{name} does not exist: {path}")
        return path

    def _uses_partitioned_layout(self) -> bool:
        if self._partitioned_layout is not None:
            return self._partitioned_layout
        root = self._path("data_root")
        assert root is not None
        if self.dataset == "wcep_ctg":
            markers = tuple(
                root / directory / "topics.jsonl"
                for directory in _PARTITION_DIRECTORIES.values()
            )
            present = tuple(path.is_file() for path in markers)
        else:
            markers = tuple(
                root / directory for directory in _PARTITION_DIRECTORIES.values()
            )
            present = tuple(path.is_dir() for path in markers)
        if any(present) and not all(present):
            missing = ", ".join(
                str(path) for path, exists in zip(markers, present, strict=True) if not exists
            )
            raise FileNotFoundError(f"partitioned dataset layout is incomplete: {missing}")
        self._partitioned_layout = all(present)
        return self._partitioned_layout

    def _partition_entities(self) -> dict[str, tuple[str, ...]]:
        if self._entities_by_partition is not None:
            return dict(self._entities_by_partition)
        if not self._uses_partitioned_layout():
            raise ValueError("the configured data root does not use split directories")
        root = self._path("data_root")
        assert root is not None
        output: dict[str, tuple[str, ...]] = {}
        owners: dict[str, str] = {}
        for partition, directory in _PARTITION_DIRECTORIES.items():
            split_root = root / directory
            if self.dataset == "crest":
                entity_ids = tuple(
                    sorted(path.name for path in split_root.iterdir() if path.is_dir())
                )
            else:
                source = split_root / "topics.jsonl"
                values: list[str] = []
                for row_number, raw in enumerate(_json_lines(source), 1):
                    if not isinstance(raw, Mapping):
                        raise ValueError(f"topic row {row_number} must be an object: {source}")
                    entity_id = _identifier(raw.get("topic_id"))
                    if not entity_id:
                        raise ValueError(f"topic row {row_number} has no topic_id: {source}")
                    declared = raw.get("split")
                    if declared is None or self._partition(str(declared)) != partition:
                        raise ValueError(
                            f"topic {entity_id!r} has the wrong split in {source}"
                        )
                    values.append(entity_id)
                if len(set(values)) != len(values):
                    raise ValueError(f"duplicate topic_id in {source}")
                entity_ids = tuple(sorted(values))
            if not entity_ids:
                raise ValueError(f"partition {directory!r} contains no entities")
            for entity_id in entity_ids:
                previous = owners.get(entity_id)
                if previous is not None:
                    raise ValueError(
                        f"entity {entity_id!r} appears in both {previous} and {partition}"
                    )
                owners[entity_id] = partition
            output[partition] = entity_ids
        self._entities_by_partition = output
        self._partition_by_entity = owners
        return dict(output)

    def _entity_partition(self, entity_id: str) -> str:
        self._partition_entities()
        assert self._partition_by_entity is not None
        try:
            return self._partition_by_entity[entity_id]
        except KeyError as error:
            raise KeyError(f"unknown entity {entity_id!r}") from error

    def _entity_root(self, entity_id: str) -> Path:
        root = self._path("data_root")
        assert root is not None
        if not self._uses_partitioned_layout():
            return root / entity_id
        partition = self._entity_partition(entity_id)
        return root / _PARTITION_DIRECTORIES[partition] / entity_id

    @staticmethod
    def _partition(value: str) -> str:
        try:
            return _PARTITION_ALIASES[value.casefold()]
        except (AttributeError, KeyError) as error:
            raise ValueError(
                "partition must be train, development/dev, or test"
            ) from error

    def _folds(self) -> dict[str, int]:
        if self._fold_map is None:
            source = self._path("folds", required=False)
            if source is None:
                raise ValueError(
                    "paths.folds is required for a dataset without split directories"
                )
            if not source.is_file():
                raise FileNotFoundError(f"fold file not found: {source}")
            raw = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("fold file must contain an object")
            output: dict[str, int] = {}
            for raw_entity_id, raw_fold in raw.items():
                entity_id = normalize_text(raw_entity_id)
                if not entity_id or isinstance(raw_fold, bool):
                    raise ValueError("fold entries must map entity IDs to integers")
                try:
                    fold = int(raw_fold)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid fold for {entity_id!r}") from error
                if fold != raw_fold or fold < 0:
                    raise ValueError(f"invalid fold for {entity_id!r}")
                output[entity_id] = fold
            self._fold_map = output
        return self._fold_map

    def entity_ids(self, partition: str) -> tuple[str, ...]:
        name = self._partition(partition)
        if self._uses_partitioned_layout():
            return self._partition_entities()[name]
        raw_folds = self.fold_config.get(name)
        if isinstance(raw_folds, int) and not isinstance(raw_folds, bool):
            folds = (raw_folds,)
        elif _is_sequence(raw_folds) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw_folds
        ):
            folds = tuple(raw_folds)
        else:
            raise ValueError(f"folds.{name} must be an integer or integer array")
        fold_map = self._folds()
        return tuple(
            entity_id
            for fold in folds
            for entity_id in sorted(
                entity for entity, assigned in fold_map.items() if assigned == fold
            )
        )

    def _constraints(self) -> dict[str, Mapping[str, object]]:
        if self._constraint_map is None:
            output: dict[str, Mapping[str, object]] = {}
            if self.dataset == "wcep_ctg" and self._uses_partitioned_layout():
                root = self._path("data_root")
                assert root is not None
                for partition, entity_ids in self._partition_entities().items():
                    source = (
                        root
                        / _PARTITION_DIRECTORIES[partition]
                        / "constraints.jsonl"
                    )
                    if not source.is_file():
                        raise FileNotFoundError(f"constraint file not found: {source}")
                    expected = set(entity_ids)
                    grouped: dict[str, dict[str, str]] = {
                        entity_id: {} for entity_id in entity_ids
                    }
                    for row_number, raw in enumerate(_json_lines(source), 1):
                        if not isinstance(raw, Mapping):
                            raise ValueError(
                                f"constraint row {row_number} must be an object: {source}"
                            )
                        entity_id = _identifier(raw.get("topic_id"))
                        constraint_id = _identifier(raw.get("constraint_id"))
                        constraint_text = normalize_text(
                            raw.get("constraint", raw.get("text"))
                        )
                        if entity_id not in expected:
                            raise ValueError(
                                f"constraint row has unknown topic {entity_id!r}: {source}"
                            )
                        if constraint_id not in _CONSTRAINT_IDS or not constraint_text:
                            raise ValueError(
                                f"invalid constraint row {row_number}: {source}"
                            )
                        if constraint_id in grouped[entity_id]:
                            raise ValueError(
                                f"duplicate constraint {entity_id}/{constraint_id}: {source}"
                            )
                        grouped[entity_id][constraint_id] = constraint_text
                    for entity_id, values in grouped.items():
                        if set(values) != set(_CONSTRAINT_IDS):
                            raise ValueError(
                                f"entity {entity_id!r} must have constraints 0 through 4"
                            )
                        output[entity_id] = values
            else:
                source = self._path("constraints", required=False)
                if source is None:
                    root = self._path("data_root")
                    assert root is not None
                    source = root / "constraint_dict.json"
                if not source.is_file():
                    raise FileNotFoundError(f"constraint file not found: {source}")
                raw = json.loads(source.read_text(encoding="utf-8"))
                if not isinstance(raw, Mapping):
                    raise ValueError("constraint file must contain an object")
                for raw_entity_id, raw_values in raw.items():
                    entity_id = _identifier(raw_entity_id)
                    if not entity_id or not isinstance(raw_values, Mapping):
                        raise ValueError("invalid constraint entry")
                    values = {
                        _identifier(raw_id): normalize_text(text)
                        for raw_id, text in raw_values.items()
                    }
                    if set(values) != set(_CONSTRAINT_IDS) or not all(values.values()):
                        raise ValueError(
                            f"entity {entity_id!r} must have constraints 0 through 4"
                        )
                    output[entity_id] = values
            self._constraint_map = output
        return self._constraint_map

    def constraints(self, entity_id: str) -> tuple[Constraint, ...]:
        entity_id = normalize_text(entity_id)
        try:
            values = self._constraints()[entity_id]
        except KeyError as error:
            raise KeyError(f"constraints missing for entity {entity_id!r}") from error
        return tuple(
            Constraint(entity_id, constraint_id, normalize_text(values[constraint_id]))
            for constraint_id in _CONSTRAINT_IDS
        )

    def constraints_for(
        self, entity_ids: Iterable[str]
    ) -> dict[str, tuple[Constraint, ...]]:
        return {entity_id: self.constraints(entity_id) for entity_id in entity_ids}

    @staticmethod
    def _sentence_text(raw: Mapping[object, object]) -> str:
        sentences = raw.get("sentences", ())
        if not _is_sequence(sentences):
            raise ValueError("article sentences must be an array")
        return normalize_text(
            " ".join(
                normalize_text(sentence.get("raw"))
                for sentence in sentences
                if isinstance(sentence, Mapping) and normalize_text(sentence.get("raw"))
            )
        )

    def articles(self, entity_id: str) -> tuple[Article, ...]:
        entity_id = normalize_text(entity_id)
        if entity_id in self._article_cache:
            return self._article_cache[entity_id]
        if self.dataset == "wcep_ctg" and self._uses_partitioned_layout():
            partition = self._entity_partition(entity_id)
            self._load_wcep_articles(partition)
            return self._article_cache[entity_id]
        source = self._entity_root(entity_id) / "articles.preprocessed.jsonl.gz"
        if not source.is_file():
            raise FileNotFoundError(f"article file not found: {source}")
        output: list[Article] = []
        for index, raw in enumerate(_json_lines(source)):
            if not isinstance(raw, Mapping):
                raise ValueError(f"article row {index + 1} must be an object: {source}")
            article_id = _identifier(raw.get("id")) or f"{entity_id}_{index}"
            text = normalize_text(raw.get("text")) or self._sentence_text(raw)
            raw_time = normalize_text(raw.get("time"))
            output.append(
                Article(
                    entity_id=entity_id,
                    article_id=article_id,
                    published_at=(
                        _parse_date(raw_time, name="article time") if raw_time else ""
                    ),
                    title=normalize_text(raw.get("title")),
                    text=text,
                )
            )
        result = tuple(output)
        self._article_cache[entity_id] = result
        return result

    def _load_wcep_articles(self, partition: str) -> None:
        if partition in self._loaded_article_partitions:
            return
        root = self._path("data_root")
        assert root is not None
        source = root / _PARTITION_DIRECTORIES[partition] / "documents.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"document file not found: {source}")
        entity_ids = self._partition_entities()[partition]
        grouped: dict[str, list[Article]] = {entity_id: [] for entity_id in entity_ids}
        seen: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}
        for row_number, raw in enumerate(_json_lines(source), 1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"document row {row_number} must be an object: {source}")
            entity_id = _identifier(raw.get("topic_id"))
            if entity_id not in grouped:
                raise ValueError(
                    f"document row has unknown topic {entity_id!r}: {source}"
                )
            article_id = _identifier(_first(raw, ("document_id", "id")))
            if not article_id:
                raise ValueError(f"document row {row_number} has no document_id: {source}")
            if article_id in seen[entity_id]:
                raise ValueError(
                    f"duplicate document ID {entity_id}/{article_id}: {source}"
                )
            seen[entity_id].add(article_id)
            raw_time = normalize_text(
                _first(raw, ("event_date", "published_at", "time", "date"))
            )
            text = normalize_text(raw.get("text")) or self._sentence_text(raw)
            grouped[entity_id].append(
                Article(
                    entity_id=entity_id,
                    article_id=article_id,
                    published_at=(
                        _parse_date(raw_time, name="document event_date")
                        if raw_time
                        else ""
                    ),
                    title=normalize_text(raw.get("title")),
                    text=text,
                )
            )
        for entity_id, articles in grouped.items():
            self._article_cache[entity_id] = tuple(articles)
        self._loaded_article_partitions.add(partition)

    def articles_for(
        self, entity_ids: Iterable[str]
    ) -> dict[str, tuple[Article, ...]]:
        return {entity_id: self.articles(entity_id) for entity_id in entity_ids}

    def references(self, entity_id: str) -> dict[str, tuple[ReferenceEvent, ...]]:
        entity_id = normalize_text(entity_id)
        if entity_id in self._reference_cache:
            return dict(self._reference_cache[entity_id])
        if self.dataset == "wcep_ctg" and self._uses_partitioned_layout():
            partition = self._entity_partition(entity_id)
            self._load_wcep_references(partition)
            return dict(self._reference_cache[entity_id])
        output: dict[str, tuple[ReferenceEvent, ...]] = {}
        for constraint in self.constraints(entity_id):
            source = (
                self._entity_root(entity_id)
                / constraint.constraint_id
                / "timelines.jsonl"
            )
            if not source.is_file():
                raise FileNotFoundError(f"reference timeline not found: {source}")
            events: list[ReferenceEvent] = []
            for raw in _json_lines(source):
                for raw_date, raw_summaries in _timeline_items(raw):
                    event_date = _parse_date(raw_date, name="reference date")
                    for summary in _summaries(raw_summaries):
                        events.append(
                            ReferenceEvent(
                                entity_id=entity_id,
                                constraint_id=constraint.constraint_id,
                                event_id=f"r_{len(events):06d}",
                                event_date=event_date,
                                summary=summary,
                            )
                        )
            output[constraint.constraint_id] = tuple(events)
        self._reference_cache[entity_id] = output
        return dict(output)

    def _load_wcep_references(self, partition: str) -> None:
        if partition in self._loaded_reference_partitions:
            return
        root = self._path("data_root")
        assert root is not None
        source = root / _PARTITION_DIRECTORIES[partition] / "gold_timelines.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"gold timeline file not found: {source}")
        entity_ids = self._partition_entities()[partition]
        expected = {
            (entity_id, constraint.constraint_id)
            for entity_id in entity_ids
            for constraint in self.constraints(entity_id)
        }
        grouped: dict[tuple[str, str], tuple[ReferenceEvent, ...]] = {}
        for row_number, raw in enumerate(_json_lines(source), 1):
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"gold timeline row {row_number} must be an object: {source}"
                )
            entity_id = _identifier(raw.get("topic_id"))
            constraint_id = _identifier(raw.get("constraint_id"))
            key = (entity_id, constraint_id)
            if key not in expected:
                raise ValueError(f"gold timeline has unknown key {key}: {source}")
            if key in grouped:
                raise ValueError(f"duplicate gold timeline {key}: {source}")
            events: list[ReferenceEvent] = []
            for raw_date, raw_summaries in _timeline_items(raw):
                event_date = _parse_date(raw_date, name="reference date")
                for summary in _summaries(raw_summaries):
                    events.append(
                        ReferenceEvent(
                            entity_id=entity_id,
                            constraint_id=constraint_id,
                            event_id=f"r_{len(events):06d}",
                            event_date=event_date,
                            summary=summary,
                        )
                    )
            grouped[key] = tuple(events)
        missing = expected - set(grouped)
        if missing:
            raise ValueError(f"gold timelines are missing keys: {sorted(missing)}")
        for entity_id in entity_ids:
            self._reference_cache[entity_id] = {
                constraint_id: grouped[(entity_id, constraint_id)]
                for constraint_id in _CONSTRAINT_IDS
            }
        self._loaded_reference_partitions.add(partition)

    def references_for(
        self, entity_ids: Iterable[str]
    ) -> dict[tuple[str, str], tuple[ReferenceEvent, ...]]:
        output: dict[tuple[str, str], tuple[ReferenceEvent, ...]] = {}
        for entity_id in entity_ids:
            for constraint_id, events in self.references(entity_id).items():
                output[(entity_id, constraint_id)] = events
        return output

__all__ = ["DatasetReader"]
