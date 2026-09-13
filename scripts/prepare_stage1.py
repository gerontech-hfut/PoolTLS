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
from pooltls.encoders import LocalTextEncoder
from pooltls.stage1_data import prepare_stage1_records, write_stage1_artifacts
from pooltls.support_verifier import LocalBooleanSupportVerifier


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build assignment-free full-document Stage-1 SFT records"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    reader = DatasetReader(config)
    embedding = section(config, "embedding")
    encoder = LocalTextEncoder(
        config["paths"]["gte_model"],
        device=args.device,
        batch_size=int(embedding.get("batch_size", 32)),
        max_length=int(embedding.get("max_length", 512)),
        chunk_overlap=int(embedding.get("chunk_overlap", 64)),
    )
    stage1 = section(config, "stage1")
    verifier = LocalBooleanSupportVerifier(
        config["paths"]["base_model"],
        device=args.device,
        max_new_tokens=stage1.get("support_verifier_max_new_tokens", 16),
        max_input_tokens=stage1.get("max_input_tokens", 24576),
    )
    try:
        artifacts = prepare_stage1_records(config, reader, encoder=encoder, verifier=verifier)
    finally:
        verifier.close()
    write_stage1_artifacts(args.output_dir, artifacts)
    print(json.dumps(artifacts["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
