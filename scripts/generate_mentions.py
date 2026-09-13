#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pooltls.config import load_config, section
from pooltls.data import DatasetReader
from pooltls.stage1_generation import (
    LocalAdapterGenerator,
    generate_partition,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one assignment-free event response per complete article"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition", choices=("train", "development", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _progress(completed: int, total: int) -> None:
    print(
        json.dumps(
            {"event": "stage1_generation_progress", "completed": completed, "total": total}
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    base_model = args.base_model or config["paths"]["base_model"]
    generator = LocalAdapterGenerator(base_model, args.adapter, device=args.device)
    summary = generate_partition(
        DatasetReader(config),
        args.partition,
        generator,
        args.output_dir,
        section(config, "stage1"),
        seed=int(config["seed"]),
        progress_callback=_progress,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
