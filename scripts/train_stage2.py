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
    score_cross_encoder,
    train_cross_encoder,
)
from pooltls.data import DatasetReader
from pooltls.encoders import LocalTextEncoder
from pooltls.evaluation import evaluate_predictions
from pooltls.io import iter_jsonl, write_json
from pooltls.ranking import (
    direct_semantic_scores,
    fusion_weight_grid,
    select_fusion_weight,
)
from pooltls.schema import Candidate, PairExample
from pooltls.timeline import build_timelines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Stage II trials and jointly select the checkpoint and "
            "cross/direct fusion weight on development data."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--development-candidates-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection-output", type=Path)
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
        event_embeddings = encoder.encode(
            tuple(candidate.summary for candidate in candidates)
        )
        constraint_embeddings = encoder.encode(
            tuple(constraint.text for constraint in constraints)
        )
        output[entity_id] = direct_semantic_scores(
            candidates,
            constraints,
            event_embeddings,
            constraint_embeddings,
        )
    return output


def _grid(values: object, name: str, conversion: Any) -> tuple[Any, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"cross_encoder.{name} must be a non-empty list")
    output = tuple(conversion(value) for value in values)
    if len(set(output)) != len(output):
        raise ValueError(f"cross_encoder.{name} contains duplicates")
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    embedding = section(config, "embedding")
    cross_config = section(config, "cross_encoder")
    fusion_config = section(config, "fusion")
    if cross_config.get("pair_order") != "constraint_event":
        raise ValueError("the minimal method requires constraint-first text pairs")
    if fusion_config.get("normalize") != "percentile_rank":
        raise ValueError("the minimal method requires percentile-rank fusion")
    if fusion_config.get("selection_metric") != "geometric_mean_f1":
        raise ValueError("the minimal method selects fusion by geometric mean F1")

    train_pair_path = args.train_pairs.expanduser().resolve()
    examples = tuple(
        PairExample.from_dict(row) for row in iter_jsonl(train_pair_path)
    )
    if not examples:
        raise ValueError(f"training pair file is empty: {train_pair_path}")

    reader = DatasetReader(config)
    development_ids = reader.entity_ids("development")
    development_candidate_dir = args.development_candidates_dir.expanduser().resolve()
    candidates_by_entity = {
        entity_id: _candidates(
            development_candidate_dir / f"{entity_id}.jsonl", entity_id
        )
        for entity_id in development_ids
    }
    constraints_by_entity = reader.constraints_for(development_ids)
    references = reader.references_for(development_ids)
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

    weights = fusion_weight_grid(
        float(fusion_config["cross_weight_start"]),
        float(fusion_config["cross_weight_stop"]),
        float(fusion_config["cross_weight_step"]),
    )
    evaluation_batch_size = int(cross_config["evaluation_batch_size"])
    max_length = int(cross_config["max_length"])

    def development_scorer(model: Any, tokenizer: Any, epoch: int) -> dict[str, Any]:
        cross_scores = score_cross_encoder(
            model,
            tokenizer,
            candidates_by_entity,
            constraints_by_entity,
            device=args.device,
            batch_size=evaluation_batch_size,
            max_length=max_length,
        )

        def evaluator(fused: dict[str, np.ndarray]) -> dict[str, float | int]:
            predictions = build_timelines(
                candidates_by_entity,
                constraints_by_entity,
                fused,
                budgets,
            )
            return evaluate_predictions(predictions, references)

        selected = select_fusion_weight(
            cross_scores,
            direct_scores,
            weights=weights,
            epoch=epoch,
            evaluator=evaluator,
        )
        return {
            "cross_weight": selected.cross_weight,
            "direct_weight": selected.direct_weight,
            "metrics": dict(selected.metrics),
            "selection_key": list(selected.selection_key),
        }

    learning_rates = _grid(
        cross_config["learning_rates"], "learning_rates", float
    )
    seeds = _grid(cross_config["seeds"], "seeds", int)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    trial_index = 0
    for learning_rate in learning_rates:
        for seed in seeds:
            trial_index += 1
            trial_dir = output_dir / f"trial_{trial_index:03d}"
            settings = {
                **cross_config,
                "learning_rate": learning_rate,
            }
            checkpoint = train_cross_encoder(
                examples,
                model_path=config["paths"]["cross_encoder_model"],
                output_dir=trial_dir,
                device=args.device,
                settings=settings,
                dev_scorer=development_scorer,
                seed=seed,
            )
            dev_result = checkpoint.get("dev_result")
            if not isinstance(dev_result, dict):
                raise ValueError("trained checkpoint is missing development results")
            summary = {
                "trial": trial_index,
                "checkpoint": str((trial_dir / "checkpoint.pt").resolve()),
                "learning_rate": learning_rate,
                "seed": seed,
                "selected_epoch": int(checkpoint["epoch"]),
                "training_example_count": len(examples),
                "class_counts": checkpoint["class_counts"],
                "development": dev_result,
                "selection_key": list(checkpoint["selection_key"]),
            }
            trial_summaries.append(summary)
            if best is None or tuple(summary["selection_key"]) > tuple(
                best["selection_key"]
            ):
                best = summary

    assert best is not None
    selected_development = best["development"]
    selection = {
        "kind": "pooltls_stage2_selection",
        "dataset": config["dataset"],
        "checkpoint": best["checkpoint"],
        "model_path": str(config["paths"]["cross_encoder_model"]),
        "selected_epoch": best["selected_epoch"],
        "learning_rate": best["learning_rate"],
        "seed": best["seed"],
        "cross_weight": selected_development["cross_weight"],
        "direct_weight": selected_development["direct_weight"],
        "development_metrics": selected_development["metrics"],
        "selection_key": best["selection_key"],
        "trial_count": len(trial_summaries),
        "trials": trial_summaries,
    }
    selection_output = (
        args.selection_output.expanduser().resolve()
        if args.selection_output is not None
        else output_dir / "selected_config.json"
    )
    write_json(selection_output, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
