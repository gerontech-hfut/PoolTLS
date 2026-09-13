#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pooltls.config import load_config, section
from pooltls.cross_encoder import (
    load_cross_encoder_checkpoint,
    score_loaded_cross_encoder,
)
from pooltls.data import DatasetReader
from pooltls.encoders import LocalTextEncoder
from pooltls.evaluation import evaluate_predictions, write_crest_timelines, write_predictions
from pooltls.io import iter_jsonl, read_json, write_json
from pooltls.ranking import direct_semantic_scores, fuse_scores
from pooltls.schema import Candidate
from pooltls.timeline import build_timelines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score candidates and build exact-budget chronological timelines."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--partition", default="test", choices=("development", "test")
    )
    parser.add_argument("--candidates-dir", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scores-output-dir", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--crest-output-dir", type=Path)
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


def _direct_scores(
    candidates_by_entity: dict[str, tuple[Candidate, ...]],
    constraints_by_entity: dict[str, tuple[Any, ...]],
    encoder: LocalTextEncoder,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for entity_id in sorted(candidates_by_entity):
        candidates = candidates_by_entity[entity_id]
        constraints = constraints_by_entity[entity_id]
        output[entity_id] = direct_semantic_scores(
            candidates,
            constraints,
            encoder.encode(tuple(candidate.summary for candidate in candidates)),
            encoder.encode(tuple(constraint.text for constraint in constraints)),
        )
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    embedding = section(config, "embedding")
    cross_config = section(config, "cross_encoder")
    fusion_config = section(config, "fusion")
    decoding = section(config, "decoding")
    if fusion_config.get("normalize") != "percentile_rank":
        raise ValueError("the minimal method requires percentile-rank fusion")
    if (
        decoding.get("budget_source") != "reference_event_count"
        or decoding.get("allow_event_sharing") is not True
        or decoding.get("chronological_sort") is not True
    ):
        raise ValueError(
            "the minimal decoder requires reference-event budgets, sharing, "
            "and chronological sorting"
        )
    selection = read_json(args.selection.expanduser().resolve())
    if not isinstance(selection, dict) or selection.get("kind") != "pooltls_stage2_selection":
        raise ValueError("selection file is not a PoolTLS Stage-II selection")
    cross_weight = float(selection["cross_weight"])
    direct_weight = float(selection["direct_weight"])
    if not np.isclose(cross_weight + direct_weight, 1.0, atol=1.0e-8):
        raise ValueError("selected fusion weights must sum to one")

    reader = DatasetReader(config)
    entity_ids = reader.entity_ids(args.partition)
    candidate_dir = args.candidates_dir.expanduser().resolve()
    candidates_by_entity = {
        entity_id: _candidates(candidate_dir / f"{entity_id}.jsonl", entity_id)
        for entity_id in entity_ids
    }
    constraints_by_entity = reader.constraints_for(entity_ids)
    references = reader.references_for(entity_ids)
    budgets = {key: len(events) for key, events in references.items()}

    direct_encoder = LocalTextEncoder(
        config["paths"]["gte_model"],
        device=args.device,
        batch_size=int(embedding["batch_size"]),
        max_length=int(embedding["max_length"]),
        chunk_overlap=int(embedding.get("chunk_overlap", 64)),
    )
    direct_scores = _direct_scores(
        candidates_by_entity, constraints_by_entity, direct_encoder
    )
    del direct_encoder

    loaded = load_cross_encoder_checkpoint(
        selection["checkpoint"], device=args.device
    )
    cross_scores = score_loaded_cross_encoder(
        loaded,
        candidates_by_entity,
        constraints_by_entity,
        device=args.device,
        batch_size=int(cross_config["evaluation_batch_size"]),
    )
    fused = fuse_scores(
        cross_scores, direct_scores, cross_weight=cross_weight
    )
    if args.scores_output_dir is not None:
        score_dir = args.scores_output_dir.expanduser().resolve()
        score_dir.mkdir(parents=True, exist_ok=True)
        for entity_id in entity_ids:
            np.savez_compressed(
                score_dir / f"{entity_id}.npz",
                cross=cross_scores[entity_id],
                direct=direct_scores[entity_id],
                scores=fused[entity_id],
                candidate_ids=np.asarray(
                    [
                        candidate.candidate_id
                        for candidate in candidates_by_entity[entity_id]
                    ]
                ),
                constraint_ids=np.asarray(
                    [
                        constraint.constraint_id
                        for constraint in constraints_by_entity[entity_id]
                    ]
                ),
            )
    predictions = build_timelines(
        candidates_by_entity,
        constraints_by_entity,
        fused,
        budgets,
    )
    output = args.output.expanduser().resolve()
    write_predictions(output, predictions)
    if args.crest_output_dir is not None:
        write_crest_timelines(args.crest_output_dir.expanduser().resolve(), predictions)
    if args.metrics_output is not None:
        metrics = evaluate_predictions(predictions, references)
        write_json(args.metrics_output.expanduser().resolve(), metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
