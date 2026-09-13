#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pooltls.config import load_config, section
from pooltls.data import DatasetReader
from pooltls.encoders import LocalTextEncoder
from pooltls.io import iter_jsonl, write_json, write_jsonl
from pooltls.schema import Candidate
from pooltls.supervision import build_supervision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Hungarian positives and reliable negatives for Stage II."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--partition", default="train", choices=("train", "development")
    )
    parser.add_argument("--candidates-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _candidates(path: Path, entity_id: str) -> tuple[Candidate, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate file not found: {path}")
    values = tuple(Candidate.from_dict(row) for row in iter_jsonl(path))
    if any(candidate.entity_id != entity_id for candidate in values):
        raise ValueError(f"candidate entity mismatch in {path}")
    identifiers = [candidate.candidate_id for candidate in values]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate candidate IDs in {path}")
    return values


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    embedding = section(config, "embedding")
    supervision = section(config, "supervision")
    if supervision.get("negative_screening") != "global_cross_constraint":
        raise ValueError(
            "the minimal method requires supervision.negative_screening="
            "global_cross_constraint"
        )
    include_reference_positives = supervision.get(
        "include_reference_positives", False
    )
    if type(include_reference_positives) is not bool:
        raise ValueError("supervision.include_reference_positives must be boolean")

    reader = DatasetReader(config)
    entity_ids = reader.entity_ids(args.partition)
    encoder = LocalTextEncoder(
        config["paths"]["gte_model"],
        device=args.device,
        batch_size=int(embedding["batch_size"]),
        max_length=int(embedding["max_length"]),
        chunk_overlap=int(embedding.get("chunk_overlap", 64)),
    )
    candidate_dir = args.candidates_dir.expanduser().resolve()
    examples = []
    per_entity: dict[str, dict[str, int]] = {}
    for entity_id in entity_ids:
        candidates = _candidates(candidate_dir / f"{entity_id}.jsonl", entity_id)
        result = build_supervision(
            candidates,
            reader.constraints(entity_id),
            reader.references(entity_id),
            encoder,
            semantic_weight=float(supervision["semantic_weight"]),
            positive_threshold=float(supervision["positive_threshold"]),
            negative_threshold=float(supervision["negative_threshold"]),
            include_reference_positives=include_reference_positives,
        )
        examples.extend(result.examples)
        per_entity[entity_id] = {
            "candidates": len(candidates),
            "candidate_positives": result.candidate_positive_count,
            "reference_positives": result.reference_positive_count,
            "reliable_negatives": result.reliable_negative_count,
        }

    output = args.output.expanduser().resolve()
    write_jsonl(output, (example.to_dict() for example in examples))
    summary_output = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else output.with_name(f"{output.stem}_summary.json")
    )
    write_json(
        summary_output,
        {
            "dataset": config["dataset"],
            "partition": args.partition,
            "entity_count": len(entity_ids),
            "example_count": len(examples),
            "positive_count": sum(example.label == 1 for example in examples),
            "negative_count": sum(example.label == 0 for example in examples),
            "supervision": supervision,
            "per_entity": per_entity,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
