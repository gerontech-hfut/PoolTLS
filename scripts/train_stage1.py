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
from pooltls.stage1_model import train_stage1_qlora


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the local full-document Stage-1 QLoRA adapter"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    summary = train_stage1_qlora(
        args.train_file,
        config["paths"]["base_model"],
        args.output_dir,
        section(config, "stage1"),
        seed=int(config["seed"]),
        device=args.device,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
