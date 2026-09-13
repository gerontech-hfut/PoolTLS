from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _date(value: object) -> str:
    value = _text(value, "event_date")[:10]
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid ISO event date: {value!r}") from error
    return value


class JsonRecord:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class Article(JsonRecord):
    entity_id: str
    article_id: str
    published_at: str
    title: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "article_id", _text(self.article_id, "article_id"))
        published_at = str(self.published_at or "").strip()
        object.__setattr__(
            self,
            "published_at",
            _date(published_at) if published_at else "",
        )
        title = str(self.title or "").strip()
        text = str(self.text or "").strip()
        if not title and not text:
            raise ValueError("an article must contain a title or body text")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True, slots=True)
class Constraint(JsonRecord):
    entity_id: str
    constraint_id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "constraint_id", _text(self.constraint_id, "constraint_id"))
        object.__setattr__(self, "text", _text(self.text, "constraint text"))


@dataclass(frozen=True, slots=True)
class Event(JsonRecord):
    entity_id: str
    event_id: str
    event_date: str
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "event_date", _date(self.event_date))
        object.__setattr__(self, "summary", _text(self.summary, "summary"))


@dataclass(frozen=True, slots=True)
class ReferenceEvent(JsonRecord):
    entity_id: str
    constraint_id: str
    event_id: str
    event_date: str
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "constraint_id", _text(self.constraint_id, "constraint_id"))
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "event_date", _date(self.event_date))
        object.__setattr__(self, "summary", _text(self.summary, "summary"))


@dataclass(frozen=True, slots=True)
class Mention(JsonRecord):
    entity_id: str
    mention_id: str
    article_id: str
    event_date: str
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "mention_id", _text(self.mention_id, "mention_id"))
        object.__setattr__(self, "article_id", _text(self.article_id, "article_id"))
        object.__setattr__(self, "event_date", _date(self.event_date))
        object.__setattr__(self, "summary", _text(self.summary, "summary"))


@dataclass(frozen=True, slots=True)
class Candidate(JsonRecord):
    entity_id: str
    candidate_id: str
    event_date: str
    summary: str
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "event_date", _date(self.event_date))
        object.__setattr__(self, "summary", _text(self.summary, "summary"))
        object.__setattr__(self, "member_ids", tuple(str(item) for item in self.member_ids))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        data = dict(value)
        data["member_ids"] = tuple(data.get("member_ids", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PairExample(JsonRecord):
    entity_id: str
    candidate_id: str
    constraint_id: str
    constraint_text: str
    event_date: str
    event_summary: str
    label: int
    matching_score: float

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("pair label must be 0 or 1")
        if not 0.0 <= float(self.matching_score) <= 1.0:
            raise ValueError("matching_score must be in [0, 1]")
        object.__setattr__(self, "event_date", _date(self.event_date))


@dataclass(frozen=True, slots=True)
class TimelineEvent(JsonRecord):
    entity_id: str
    constraint_id: str
    candidate_id: str
    event_date: str
    summary: str
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_date", _date(self.event_date))
