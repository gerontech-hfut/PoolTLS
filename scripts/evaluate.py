#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pooltls.config import load_config
from pooltls.data import DatasetReader
from pooltls.evaluation import evaluate_dataset, read_predictions
from pooltls.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated PoolTLS timelines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--partition", choices=("development", "test"), default="test"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    reader = DatasetReader(config)
    predictions = read_predictions(args.predictions)
    metrics = evaluate_dataset(predictions, reader, args.partition)
    write_json(args.output, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
