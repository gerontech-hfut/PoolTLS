#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pooltls.config import load_config, section
from pooltls.consolidation import consolidate_mentions
from pooltls.data import DatasetReader
from pooltls.encoders import LocalTextEncoder
from pooltls.io import iter_jsonl, write_json, write_jsonl
from pooltls.schema import Mention


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate generated mentions into shared event candidates."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--partition", required=True, choices=("train", "development", "test")
    )
    parser.add_argument("--mentions-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _mentions(path: Path, entity_id: str) -> tuple[Mention, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"mention file not found: {path}")
    values = tuple(Mention.from_dict(row) for row in iter_jsonl(path))
    if any(mention.entity_id != entity_id for mention in values):
        raise ValueError(f"mention entity mismatch in {path}")
    return values


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    embedding = section(config, "embedding")
    clustering = section(config, "clustering")
    reader = DatasetReader(config)
    entity_ids = reader.entity_ids(args.partition)
    encoder = LocalTextEncoder(
        config["paths"]["gte_model"],
        device=args.device,
        batch_size=int(embedding["batch_size"]),
        max_length=int(embedding["max_length"]),
        chunk_overlap=int(embedding.get("chunk_overlap", 64)),
    )

    output_dir = args.output_dir.expanduser().resolve()
    mention_dir = args.mentions_dir.expanduser().resolve()
    counts: dict[str, dict[str, int]] = {}
    for entity_id in entity_ids:
        mentions = _mentions(mention_dir / f"{entity_id}.jsonl", entity_id)
        candidates = consolidate_mentions(
            mentions,
            encoder,
            semantic_threshold=float(clustering["semantic_threshold"]),
            word_f1_threshold=float(clustering["word_f1_threshold"]),
            semantic_weight=float(clustering["semantic_weight"]),
        )
        write_jsonl(
            output_dir / f"{entity_id}.jsonl",
            (candidate.to_dict() for candidate in candidates),
        )
        counts[entity_id] = {
            "mentions": len(mentions),
            "candidates": len(candidates),
        }
    write_json(
        output_dir / "_meta" / "summary.json",
        {
            "dataset": config["dataset"],
            "partition": args.partition,
            "entity_count": len(entity_ids),
            "mention_count": sum(value["mentions"] for value in counts.values()),
            "candidate_count": sum(value["candidates"] for value in counts.values()),
            "per_entity": counts,
            "clustering": clustering,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
