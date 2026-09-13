from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import iter_jsonl, write_json


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids", ())
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("expected one tokenized sequence")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("tokenizer must return a sequence of token IDs")
    return [int(item) for item in value]


def _find_last(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return None


def _render_and_split(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[int], int]:
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=False
        )
    )
    prefix_ids = _token_ids(
        tokenizer.apply_chat_template(
            list(messages[:-1]), tokenize=True, add_generation_prompt=True
        )
    )
    if prefix_ids and full_ids[: len(prefix_ids)] == prefix_ids:
        if len(prefix_ids) >= len(full_ids):
            raise ValueError("assistant turn contains no supervised tokens")
        return full_ids, len(prefix_ids)

    assistant = str(messages[-1].get("content", ""))
    locations: list[int] = []
    for candidate in (assistant, "\n" + assistant, " " + assistant):
        candidate_ids = _token_ids(tokenizer(candidate, add_special_tokens=False))
        location = _find_last(full_ids, candidate_ids)
        if location is not None:
            locations.append(location)
    if locations:
        return full_ids, max(locations)

    common = 0
    while common < min(len(full_ids), len(prefix_ids)):
        if full_ids[common] != prefix_ids[common]:
            break
        common += 1
    if 0 < common < len(full_ids):
        return full_ids, common
    raise ValueError("could not locate assistant content in the chat template")


class FullDocumentSFTDataset:
    """Tokenize complete chats and mask every token before the assistant answer."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        import torch

        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        self.examples: list[dict[str, Any]] = []
        maximum = 0
        for index, record in enumerate(records):
            messages = record.get("messages")
            if not isinstance(messages, Sequence) or isinstance(
                messages, (str, bytes, bytearray)
            ):
                raise ValueError(f"SFT record {index} must contain a messages array")
            if len(messages) != 3 or any(not isinstance(item, Mapping) for item in messages):
                raise ValueError(
                    f"SFT record {index} must contain system, user, assistant messages"
                )
            if [item.get("role") for item in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValueError(
                    f"SFT record {index} roles must be system, user, assistant"
                )
            full_ids, assistant_start = _render_and_split(tokenizer, messages)
            observed = len(full_ids)
            maximum = max(maximum, observed)
            if observed > max_length:
                raise ValueError(
                    f"full SFT record {index} has {observed} tokens and exceeds "
                    f"max_length={max_length}; complete documents are never truncated"
                )
            labels = torch.tensor(full_ids, dtype=torch.long)
            labels[:assistant_start] = -100
            if not bool(labels.ne(-100).any()):
                raise ValueError(f"SFT record {index} has no assistant tokens")
            self.examples.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(observed, dtype=torch.long),
                    "labels": labels,
                }
            )
        if not self.examples:
            raise ValueError("Stage-1 SFT file contains no records")
        self.stats = {
            "records": len(self.examples),
            "maximum_observed_length": maximum,
            "max_length": max_length,
            "truncated_records": 0,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


@dataclass(slots=True)
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not examples:
            raise ValueError("cannot collate an empty batch")
        lengths = [int(torch.as_tensor(item["input_ids"]).numel()) for item in examples]
        maximum = max(lengths)
        shape = len(examples), maximum
        input_ids = torch.full(shape, int(self.pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.long)
        labels = torch.full(shape, -100, dtype=torch.long)
        for row, (example, length) in enumerate(zip(examples, lengths, strict=True)):
            ids = torch.as_tensor(example["input_ids"], dtype=torch.long).reshape(-1)
            item_labels = torch.as_tensor(example["labels"], dtype=torch.long).reshape(-1)
            item_mask = torch.as_tensor(
                example.get("attention_mask", torch.ones(length)), dtype=torch.long
            ).reshape(-1)
            if len(ids) != length or len(item_labels) != length or len(item_mask) != length:
                raise ValueError("SFT tensor fields must have equal lengths")
            input_ids[row, :length] = ids
            attention_mask[row, :length] = item_mask
            labels[row, :length] = item_labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def read_sft_records(path: str | Path) -> list[Mapping[str, Any]]:
    records = list(iter_jsonl(path))
    if not records:
        raise ValueError(f"SFT training file is empty: {Path(path)}")
    return records


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive real number")
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be a positive real number")
    return resolved


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def train_stage1_qlora(
    train_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    stage1: Mapping[str, Any],
    *,
    seed: int,
    device: str = "cuda:0",
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train the configured local NF4 QLoRA adapter without truncating records."""

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    output = Path(output_dir)
    if resume_from_checkpoint is None:
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
    elif not output.is_dir():
        raise FileNotFoundError(f"resume output directory does not exist: {output}")
    final_adapter = output / "final_adapter"
    if final_adapter.exists():
        raise FileExistsError(f"final adapter already exists: {final_adapter}")

    max_length = _positive_int(
        int(stage1.get("max_sequence_length", 32768)), "max_sequence_length"
    )
    epochs = _positive_float(stage1.get("epochs", 3.0), "epochs")
    learning_rate = _positive_float(
        stage1.get("learning_rate", 1e-4), "learning_rate"
    )
    train_batch_size = _positive_int(
        int(stage1.get("train_batch_size", 1)), "train_batch_size"
    )
    accumulation = _positive_int(
        int(stage1.get("gradient_accumulation_steps", 8)),
        "gradient_accumulation_steps",
    )
    rank = _positive_int(int(stage1.get("lora_rank", 16)), "lora_rank")
    alpha = _positive_int(int(stage1.get("lora_alpha", 32)), "lora_alpha")
    dropout = float(stage1.get("lora_dropout", 0.05))
    if not isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")

    records = read_sft_records(train_file)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = FullDocumentSFTDataset(records, tokenizer, max_length=max_length)

    if device == "cpu":
        raise ValueError("4-bit QLoRA training requires a CUDA device or device='auto'")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    attention = str(stage1.get("attention", "flash_attention_2"))
    fallback = stage1.get("attention_fallback")
    if fallback is not None and not isinstance(fallback, str):
        raise ValueError("attention_fallback must be a string or null")
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "quantization_config": quantization,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto" if device == "auto" else {"": device},
        "attn_implementation": attention,
    }
    selected_attention = attention
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    except (ImportError, RuntimeError, ValueError):
        if not fallback or fallback == attention:
            raise
        selected_attention = fallback
        model_kwargs["attn_implementation"] = fallback
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=list(LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False

    output.mkdir(parents=True, exist_ok=resume_from_checkpoint is not None)
    checkpoints = output / "checkpoints"
    arguments = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=learning_rate,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=5,
        remove_unused_columns=False,
        report_to=[],
        optim="paged_adamw_8bit",
        seed=int(seed),
        data_seed=int(seed),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=CausalLMCollator(int(tokenizer.pad_token_id)),
    )
    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    checkpoint_paths = sorted(
        (item for item in checkpoints.glob("checkpoint-*") if item.is_dir()),
        key=lambda item: int(item.name.rsplit("-", 1)[-1]),
    )
    summary = {
        "base_model": str(model_path),
        "train_file": str(train_file),
        "train_examples": len(dataset),
        "max_length": max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": accumulation,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "train_loss": float(getattr(result, "training_loss", float("nan"))),
        "global_step": int(getattr(result, "global_step", 0)),
        "attention_implementation": selected_attention,
        "resumed_from_checkpoint": (
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        ),
        "checkpoints": [str(item) for item in checkpoint_paths],
        "final_adapter": str(final_adapter),
        "dataset": dict(dataset.stats),
    }
    write_json(output / "training_summary.json", summary)
    return summary


__all__ = [
    "CausalLMCollator",
    "FullDocumentSFTDataset",
    "LORA_TARGET_MODULES",
    "read_sft_records",
    "train_stage1_qlora",
]
