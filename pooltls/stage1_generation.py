from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING

from .io import write_json, write_jsonl
from .schema import Mention
from .stage1_data import joint_messages
from .text import batched, extract_json_array, normalize_text, stable_id


if TYPE_CHECKING:
    from .data import DatasetReader


DATE_KEYS = ("date", "eventdate", "time", "datetime")
SUMMARY_KEYS = ("eventsummary", "summary", "event", "description")


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_input_tokens: int = 24576
    max_new_tokens: int = 4096
    temperature: float = 0.0
    batch_size: int = 2
    require_explicit_target_name: bool = True

    def __post_init__(self) -> None:
        if type(self.max_input_tokens) is not int or self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be a positive integer")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(self.temperature, bool)
            or not isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be non-negative")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(self.require_explicit_target_name) is not bool:
            raise ValueError("require_explicit_target_name must be a boolean")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GenerationSettings":
        return cls(
            max_input_tokens=int(values.get("max_input_tokens", 24576)),
            max_new_tokens=int(values.get("max_new_tokens", 4096)),
            temperature=float(values.get("temperature", 0.0)),
            batch_size=int(values.get("generation_batch_size", 2)),
            require_explicit_target_name=values.get(
                "require_explicit_target_name", True
            ),
        )


@dataclass(frozen=True, slots=True)
class ParsedEvent:
    event_date: str
    summary: str


@dataclass(frozen=True, slots=True)
class ParseFailure:
    reason: str
    item_index: int | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ParseResult:
    events: tuple[ParsedEvent, ...]
    failures: tuple[ParseFailure, ...]
    status: str
    recovered: bool = False


class Generator(Protocol):
    def generate_batch(
        self,
        batch_messages: Sequence[Sequence[Mapping[str, str]]],
        *,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
    ) -> Sequence[str]: ...


class InputTooLongError(ValueError):
    def __init__(self, lengths: Sequence[int], maximum: int) -> None:
        self.lengths = tuple(int(item) for item in lengths)
        self.maximum = int(maximum)
        super().__init__(
            f"input token lengths {self.lengths} exceed max_input_tokens={self.maximum}"
        )


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) if isinstance(value, str) else ""


def _aliased_value(
    item: Mapping[object, object], aliases: Sequence[str]
) -> tuple[object | None, bool]:
    normalized = [(_normalized_key(key), value) for key, value in item.items()]
    for alias_index, alias in enumerate(aliases):
        for key, value in normalized:
            if key == alias:
                return value, alias_index != 0
    return None, False


def _iso_date(value: object) -> str:
    text = normalize_text(value)
    if len(text) < 10:
        raise ValueError(f"invalid event date: {value!r}")
    resolved = text[:10]
    date.fromisoformat(resolved)
    return resolved


def parse_generation_response(response: object) -> ParseResult:
    """Parse valid event objects while retaining precise per-item failures."""

    if not isinstance(response, str):
        return ParseResult((), (ParseFailure("invalid_response_type"),), "invalid_response_type")
    if not response.strip():
        return ParseResult((), (ParseFailure("empty_response"),), "empty_response")
    payload, recovered = extract_json_array(response)
    if payload is None:
        return ParseResult((), (ParseFailure("invalid_json_array"),), "invalid_json_array")
    if not payload:
        return ParseResult((), (), "empty", recovered)

    events: list[ParsedEvent] = []
    failures: list[ParseFailure] = []
    aliases_recovered = False
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            failures.append(ParseFailure("invalid_item", index))
            continue
        raw_date, date_alias = _aliased_value(item, DATE_KEYS)
        raw_summary, summary_alias = _aliased_value(item, SUMMARY_KEYS)
        aliases_recovered = aliases_recovered or date_alias or summary_alias
        if raw_date is None or not normalize_text(raw_date):
            failures.append(ParseFailure("missing_date", index))
            continue
        try:
            event_date = _iso_date(raw_date)
        except (TypeError, ValueError) as error:
            failures.append(ParseFailure("invalid_date", index, str(error)))
            continue
        if not isinstance(raw_summary, str):
            failures.append(
                ParseFailure(
                    "missing_event_summary"
                    if raw_summary is None
                    else "invalid_event_summary",
                    index,
                )
            )
            continue
        summary = normalize_text(raw_summary)
        if not summary:
            failures.append(ParseFailure("empty_event_summary", index))
            continue
        identity = event_date, summary.casefold()
        if identity in seen:
            failures.append(ParseFailure("duplicate_event", index))
            continue
        seen.add(identity)
        events.append(ParsedEvent(event_date, summary))

    was_recovered = recovered or aliases_recovered
    status = (
        "partial"
        if failures and events
        else "invalid_items"
        if failures
        else "recovered"
        if was_recovered
        else "ok"
    )
    return ParseResult(tuple(events), tuple(failures), status, was_recovered)


class LocalAdapterGenerator:
    """Local BF16 base model with a PEFT adapter."""

    def __init__(
        self,
        base_model: str | Path,
        adapter: str | Path,
        *,
        device: str = "cuda:0",
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(base_model), local_files_only=True, use_fast=True
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        device_map: Any = "auto" if device == "auto" else {"": device}
        base = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            local_files_only=True,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model = PeftModel.from_pretrained(
            base, str(adapter), local_files_only=True
        )
        self.model.eval()
        self.model.requires_grad_(False)
        raw_context = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(raw_context, int) or raw_context <= 0:
            raw_context = getattr(self.tokenizer, "model_max_length", None)
        self.model_context_length = (
            int(raw_context)
            if isinstance(raw_context, int) and 0 < raw_context < 10_000_000
            else None
        )

    def generate_batch(
        self,
        batch_messages: Sequence[Sequence[Mapping[str, str]]],
        *,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
    ) -> Sequence[str]:
        rendered = [
            self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]
        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        input_lengths = [int(item) for item in attention_mask.sum(dim=-1).tolist()]
        if any(length > max_input_tokens for length in input_lengths):
            raise InputTooLongError(input_lengths, max_input_tokens)
        if self.model_context_length is not None and any(
            length + max_new_tokens > self.model_context_length
            for length in input_lengths
        ):
            offenders = tuple(
                length
                for length in input_lengths
                if length + max_new_tokens > self.model_context_length
            )
            raise ValueError(
                "input plus configured output budget exceeds model context "
                f"{self.model_context_length}: input_lengths={offenders}, "
                f"max_new_tokens={max_new_tokens}"
            )
        input_width = int(input_ids.shape[-1])
        try:
            target_device = self.model.get_input_embeddings().weight.device
        except AttributeError:
            target_device = next(self.model.parameters()).device
        prepared = {
            key: value.to(target_device)
            if isinstance(value, self._torch.Tensor)
            else value
            for key, value in encoded.items()
        }
        arguments: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            arguments["temperature"] = temperature
        with self._torch.inference_mode():
            generated = self.model.generate(**prepared, **arguments)

        outputs: list[str] = []
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        for index in range(len(input_lengths)):
            raw_ids = [int(item) for item in generated[index, input_width:].tolist()]
            content_ids: list[int] = []
            for token_id in raw_ids:
                if eos_id is not None and token_id == eos_id:
                    break
                if pad_id is not None and token_id == pad_id:
                    break
                content_ids.append(token_id)
            outputs.append(
                self.tokenizer.decode(
                    content_ids, skip_special_tokens=True
                ).strip()
            )
        return tuple(outputs)


def _invoke_batch(
    generator: Any,
    messages: Sequence[Sequence[Mapping[str, str]]],
    settings: GenerationSettings,
) -> tuple[str, ...]:
    arguments = {
        "max_input_tokens": settings.max_input_tokens,
        "max_new_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
    }
    if hasattr(generator, "generate_batch"):
        values = generator.generate_batch(messages, **arguments)
    elif hasattr(generator, "generate"):
        values = [generator.generate(item, **arguments) for item in messages]
    else:
        raise TypeError("generator must expose generate_batch or generate")
    resolved = tuple(values)
    if len(resolved) != len(messages):
        raise ValueError(
            f"generator returned {len(resolved)} responses for {len(messages)} prompts"
        )
    if any(not isinstance(value, str) for value in resolved):
        raise TypeError("generator outputs must be strings")
    return resolved


def generate_partition(
    reader: "DatasetReader",
    partition: str,
    generator: Generator,
    output_dir: str | Path,
    stage1: Mapping[str, Any],
    *,
    seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Generate mentions from complete articles and write entity JSONL files."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    settings = GenerationSettings.from_mapping(stage1)
    entity_ids = reader.entity_ids(partition)
    constraints = reader.constraints_for(entity_ids)
    raw_articles = reader.articles_for(entity_ids)
    articles = {
        entity_id: tuple(sorted(
            raw_articles[entity_id],
            key=lambda item: (item.published_at, item.article_id),
        ))
        for entity_id in entity_ids
    }
    all_mentions: dict[str, list[Mention]] = {entity_id: [] for entity_id in entity_ids}
    failure_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    seen_mentions: set[str] = set()
    raw_total = sum(len(raw_articles[entity_id]) for entity_id in entity_ids)
    total = sum(len(articles[entity_id]) for entity_id in entity_ids)
    completed = 0

    for entity_id in entity_ids:
        entity_constraints = constraints[entity_id]
        for article_batch in batched(articles[entity_id], settings.batch_size):
            batch_messages = tuple(
                joint_messages(
                    article,
                    entity_constraints,
                    seed=seed,
                    require_explicit_target_name=settings.require_explicit_target_name,
                )
                for article in article_batch
            )
            try:
                responses = _invoke_batch(generator, batch_messages, settings)
            except InputTooLongError as error:
                details = ", ".join(
                    f"{article.entity_id}/{article.article_id}={length}"
                    for article, length in zip(article_batch, error.lengths, strict=True)
                    if length > error.maximum
                )
                raise ValueError(
                    f"complete article prompts exceed max_input_tokens={error.maximum}: "
                    f"{details}"
                ) from error
            for article, response in zip(article_batch, responses, strict=True):
                parsed = parse_generation_response(response)
                status_counts[parsed.status] += 1
                for event in parsed.events:
                    mention_id = stable_id(
                        "mention_",
                        article.entity_id,
                        article.article_id,
                        event.event_date,
                        event.summary.casefold(),
                    )
                    if mention_id in seen_mentions:
                        raise RuntimeError(f"duplicate mention ID: {mention_id}")
                    seen_mentions.add(mention_id)
                    all_mentions[entity_id].append(
                        Mention(
                            entity_id=entity_id,
                            mention_id=mention_id,
                            article_id=article.article_id,
                            event_date=event.event_date,
                            summary=event.summary,
                        )
                    )
                for failure in parsed.failures:
                    row: dict[str, Any] = {
                        "entity_id": entity_id,
                        "article_id": article.article_id,
                        "reason": failure.reason,
                    }
                    if failure.item_index is not None:
                        row["item_index"] = failure.item_index
                    if failure.detail is not None:
                        row["detail"] = failure.detail
                    failure_rows.append(row)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

    output.mkdir(parents=True)
    metadata = output / "_meta"
    metadata.mkdir()
    for entity_id in entity_ids:
        mentions = sorted(
            all_mentions[entity_id],
            key=lambda item: (
                item.event_date,
                item.summary.casefold(),
                item.summary,
                item.mention_id,
            ),
        )
        write_jsonl(output / f"{entity_id}.jsonl", (item.to_dict() for item in mentions))
    write_jsonl(metadata / "parse_failures.jsonl", failure_rows)
    summary = {
        "partition": partition,
        "entities": list(entity_ids),
        "raw_articles": raw_total,
        "articles": total,
        "mentions": sum(len(values) for values in all_mentions.values()),
        "parse_failures": len(failure_rows),
        "parse_statuses": dict(sorted(status_counts.items())),
        "settings": {
            "max_input_tokens": settings.max_input_tokens,
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "batch_size": settings.batch_size,
            "allow_input_truncation": False,
            "require_explicit_target_name": settings.require_explicit_target_name,
            "deduplicate_articles": False,
        },
    }
    write_json(metadata / "summary.json", summary)
    return summary


__all__ = [
    "GenerationSettings",
    "Generator",
    "InputTooLongError",
    "LocalAdapterGenerator",
    "ParseFailure",
    "ParseResult",
    "ParsedEvent",
    "generate_partition",
    "parse_generation_response",
]
