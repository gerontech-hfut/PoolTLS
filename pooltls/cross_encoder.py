from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Candidate, Constraint, PairExample


@dataclass(frozen=True, slots=True)
class LoadedCrossEncoder:
    tokenizer: Any
    model: Any
    checkpoint: Mapping[str, Any]


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return result


def balanced_binary_cross_entropy_with_logits(
    logits: Any,
    targets: Any,
    *,
    positive_count: int,
    negative_count: int,
) -> Any:
    """Return the positive-class mean plus the negative-class mean.

    Fixed inverse-population weights make shuffled minibatches an unbiased
    stochastic estimate of the two equally weighted terms in the paper loss.
    """

    import torch
    import torch.nn.functional as functional

    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must be torch tensors")
    if logits.shape != targets.shape or logits.numel() == 0:
        raise ValueError("logits and targets must have the same non-empty shape")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
    if logits.device != targets.device or logits.dtype != targets.dtype:
        raise ValueError("logits and targets must share device and dtype")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("logits and targets must be finite")
    if not bool(((targets == 0.0) | (targets == 1.0)).all()):
        raise ValueError("targets must be binary")
    positive = _positive_integer(positive_count, "positive_count")
    negative = _positive_integer(negative_count, "negative_count")
    population = float(positive + negative)
    weights = torch.where(
        targets == 1.0,
        torch.as_tensor(population / positive, dtype=logits.dtype, device=logits.device),
        torch.as_tensor(population / negative, dtype=logits.dtype, device=logits.device),
    )
    return functional.binary_cross_entropy_with_logits(logits, targets, weight=weights)


def _load_pretrained(model_path: str | Path) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = str(Path(model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=1,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(checkpoint), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _selection_key(
    result: Mapping[str, Any] | None, *, loss: float, epoch: int
) -> tuple[float, ...]:
    if result is None:
        return (-float(loss), -float(epoch))
    raw = result.get("selection_key")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("development scorer must return a non-empty selection_key")
    key = tuple(float(value) for value in raw)
    if not all(np.isfinite(value) for value in key):
        raise ValueError("development selection key must be finite")
    return key


def train_cross_encoder(
    examples: Sequence[PairExample],
    *,
    model_path: str | Path,
    output_dir: str | Path,
    device: str,
    settings: Mapping[str, Any],
    dev_scorer: Callable[[Any, Any, int], Mapping[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune constraint-first MiniLM and retain the best dev epoch."""

    import torch

    values = tuple(examples)
    if not values:
        raise ValueError("cross-encoder training examples must not be empty")
    positive_count = sum(example.label == 1 for example in values)
    negative_count = sum(example.label == 0 for example in values)
    if not positive_count or not negative_count:
        raise ValueError("balanced BCE requires positive and negative examples")
    if settings.get("pair_order", "constraint_event") != "constraint_event":
        raise ValueError("the minimal method requires constraint-first pair order")

    max_epochs = _positive_integer(settings.get("max_epochs", 3), "max_epochs")
    batch_size = _positive_integer(
        settings.get("train_batch_size", settings.get("batch_size", 32)),
        "train_batch_size",
    )
    max_length = _positive_integer(settings.get("max_length", 192), "max_length")
    learning_rate = _finite_number(
        settings.get("learning_rate", 2.0e-5), "learning_rate", positive=True
    )
    weight_decay = _finite_number(
        settings.get("weight_decay", 0.0), "weight_decay", positive=False
    )
    gradient_clip = _finite_number(
        settings.get("gradient_clip", 1.0), "gradient_clip", positive=True
    )
    normalized_settings = {
        **dict(settings),
        "pair_order": "constraint_event",
        "max_epochs": max_epochs,
        "train_batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "loss": "balanced_binary_cross_entropy",
    }

    checkpoint_path = Path(output_dir).expanduser().resolve() / "checkpoint.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    target_device = torch.device(device)
    tokenizer, model = _load_pretrained(model_path)
    model.to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    indices = list(range(len(values)))
    for epoch in range(1, max_epochs + 1):
        model.train()
        rng.shuffle(indices)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(indices), batch_size):
            batch_examples = [values[index] for index in indices[start : start + batch_size]]
            encoded = tokenizer(
                [example.constraint_text for example in batch_examples],
                [example.event_summary for example in batch_examples],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            model_inputs = {
                name: value.to(target_device) if isinstance(value, torch.Tensor) else value
                for name, value in encoded.items()
            }
            targets = torch.as_tensor(
                [example.label for example in batch_examples],
                dtype=torch.float32,
                device=target_device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(**model_inputs).logits.reshape(-1)
            if logits.shape != targets.shape:
                raise ValueError("cross encoder must return one logit per pair")
            loss = balanced_binary_cross_entropy_with_logits(
                logits,
                targets,
                positive_count=positive_count,
                negative_count=negative_count,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_examples)
            seen += len(batch_examples)

        mean_loss = loss_sum / seen
        model.eval()
        dev_result = (
            dict(dev_scorer(model, tokenizer, epoch))
            if dev_scorer is not None
            else None
        )
        key = _selection_key(dev_result, loss=mean_loss, epoch=epoch)
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "dev_result": dev_result,
                "selection_key": list(key),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            best_result = dev_result

    assert best_key is not None and best_state is not None
    checkpoint: dict[str, Any] = {
        "kind": "pooltls_cross_encoder",
        "model_path": str(Path(model_path).expanduser().resolve()),
        "epoch": best_epoch,
        "seed": int(seed),
        "settings": normalized_settings,
        "class_counts": {"positive": positive_count, "negative": negative_count},
        "dev_result": best_result,
        "selection_key": list(best_key),
        "history": history,
        "state_dict": best_state,
    }
    _save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def load_cross_encoder_checkpoint(
    path: str | Path, *, device: str, model_path: str | Path | None = None
) -> LoadedCrossEncoder:
    import torch

    checkpoint = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping) or checkpoint.get("kind") != "pooltls_cross_encoder":
        raise ValueError("checkpoint is not a PoolTLS cross encoder")
    source = model_path if model_path is not None else checkpoint.get("model_path")
    if not isinstance(source, (str, Path)):
        raise ValueError("checkpoint does not identify its pretrained model")
    tokenizer, model = _load_pretrained(source)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint is missing its state dictionary")
    model.load_state_dict(dict(state_dict))
    model.to(torch.device(device))
    model.eval()
    return LoadedCrossEncoder(tokenizer, model, checkpoint)


def score_cross_encoder(
    model: Any,
    tokenizer: Any,
    candidates_by_entity: Mapping[str, Sequence[Candidate]],
    constraints_by_entity: Mapping[str, Sequence[Constraint]],
    *,
    device: str,
    batch_size: int = 256,
    max_length: int = 192,
) -> dict[str, np.ndarray]:
    """Return raw constraint-first logits for every candidate--constraint pair."""

    import torch

    size = _positive_integer(batch_size, "batch_size")
    length = _positive_integer(max_length, "max_length")
    if set(candidates_by_entity) != set(constraints_by_entity):
        raise ValueError("candidate and constraint entity IDs must match")
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    output: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for entity_id in sorted(candidates_by_entity):
            candidates = tuple(candidates_by_entity[entity_id])
            constraints = tuple(constraints_by_entity[entity_id])
            if not constraints:
                raise ValueError(f"entity {entity_id} has no constraints")
            first = [
                constraint.text
                for candidate in candidates
                for constraint in constraints
            ]
            second = [
                candidate.summary
                for candidate in candidates
                for constraint in constraints
            ]
            chunks: list[Any] = []
            for start in range(0, len(first), size):
                encoded = tokenizer(
                    first[start : start + size],
                    second[start : start + size],
                    padding=True,
                    truncation=True,
                    max_length=length,
                    return_tensors="pt",
                )
                batch = {
                    name: value.to(target_device) if isinstance(value, torch.Tensor) else value
                    for name, value in encoded.items()
                }
                logits = model(**batch).logits.reshape(-1)
                if logits.shape[0] != len(first[start : start + size]):
                    raise ValueError("cross encoder returned an inconsistent batch")
                chunks.append(logits.detach().cpu())
            if chunks:
                flat = torch.cat(chunks).numpy().astype(np.float32, copy=False)
                scores = flat.reshape(len(candidates), len(constraints))
            else:
                scores = np.empty((0, len(constraints)), dtype=np.float32)
            output[entity_id] = scores
    return output


def score_loaded_cross_encoder(
    loaded: LoadedCrossEncoder,
    candidates_by_entity: Mapping[str, Sequence[Candidate]],
    constraints_by_entity: Mapping[str, Sequence[Constraint]],
    *,
    device: str,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    if not isinstance(loaded, LoadedCrossEncoder):
        raise ValueError("loaded must be a LoadedCrossEncoder")
    settings = loaded.checkpoint.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("checkpoint settings are invalid")
    return score_cross_encoder(
        loaded.model,
        loaded.tokenizer,
        candidates_by_entity,
        constraints_by_entity,
        device=device,
        batch_size=(
            int(settings.get("evaluation_batch_size", 256))
            if batch_size is None
            else batch_size
        ),
        max_length=int(settings.get("max_length", 192)),
    )


__all__ = [
    "LoadedCrossEncoder",
    "balanced_binary_cross_entropy_with_logits",
    "load_cross_encoder_checkpoint",
    "score_cross_encoder",
    "score_loaded_cross_encoder",
    "train_cross_encoder",
]
