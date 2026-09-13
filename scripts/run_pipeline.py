#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pooltls.config import load_config
from pooltls.evaluation import (
    read_predictions,
    write_crest_timelines,
    write_predictions,
)
from pooltls.io import read_json, write_json
from pooltls.pipeline import STAGE_NAMES, artifact_paths, preflight, select_stages


RUN_MANIFEST = {
    "schema_version": 1,
    "workflow": "article_generation",
    "method_name": "PoolTLS",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PoolTLS (Shared-Pool Timeline Summarization) experiment from raw data to timelines"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--from-stage", choices=STAGE_NAMES)
    parser.add_argument("--stop-after", choices=STAGE_NAMES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def _command(script: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / script),
        *(str(argument) for argument in arguments),
    ]


def _run_command(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        rendered = " ".join(command)
        print(f"RUN {rendered}", flush=True)
        log.write(f"RUN {rendered}\n")
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, list(command))


def _all_exist(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def _prepare_run_directory(run_dir: Path, *, resume: bool) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"run directory is not empty; use --resume to continue: {run_dir}"
            )
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Cannot resume a run without a valid article-generation manifest. "
                "Start the complete pipeline in a new --run-dir."
            ) from exc
        if manifest != RUN_MANIFEST:
            raise ValueError(
                "Cannot resume a run with an incompatible article-generation manifest. "
                "Start the complete pipeline in a new --run-dir."
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, RUN_MANIFEST)


def _handlers(
    config_path: Path,
    run_dir: Path,
    device: str,
    *,
    resume: bool,
) -> tuple[dict[str, Callable[[], None]], dict[str, Callable[[], bool]]]:
    paths = artifact_paths(run_dir)
    logs = paths["logs"]

    def command_stage(stage: str, command: Sequence[str]) -> None:
        _run_command(command, logs / f"{stage}.log")

    def prepare_stage1() -> None:
        command_stage(
            "prepare_stage1",
            _command(
                "prepare_stage1.py",
                "--config",
                config_path,
                "--output-dir",
                paths["stage1_data"],
                "--device",
                device,
            ),
        )

    def train_stage1() -> None:
        command_stage(
            "train_stage1",
            _command(
                "train_stage1.py",
                "--config",
                config_path,
                "--train-file",
                paths["stage1_data"] / "train.jsonl",
                "--output-dir",
                paths["root"] / "models" / "stage1",
                "--device",
                device,
            ),
        )

    def generate(partition: str) -> None:
        stage = f"generate_{partition}"
        command_stage(
            stage,
            _command(
                "generate_mentions.py",
                "--config",
                config_path,
                "--adapter",
                paths["stage1_adapter"],
                "--output-dir",
                paths["mentions"] / partition,
                "--partition",
                partition,
                "--device",
                device,
            ),
        )

    def cluster_all() -> None:
        for partition in ("train", "development", "test"):
            summary = paths["candidates"] / partition / "_meta" / "summary.json"
            if resume and summary.is_file():
                print(f"SKIP cluster_{partition}: {summary}", flush=True)
                continue
            command_stage(
                f"cluster_{partition}",
                _command(
                    "build_candidates.py",
                    "--config",
                    config_path,
                    "--partition",
                    partition,
                    "--mentions-dir",
                    paths["mentions"] / partition,
                    "--output-dir",
                    paths["candidates"] / partition,
                    "--device",
                    device,
                ),
            )

    def prepare_stage2() -> None:
        paths["stage2_data"].mkdir(parents=True, exist_ok=True)
        output = paths["stage2_data"] / "train.jsonl"
        command_stage(
            "prepare_stage2",
            _command(
                "prepare_stage2.py",
                "--config",
                config_path,
                "--partition",
                "train",
                "--candidates-dir",
                paths["candidates"] / "train",
                "--output",
                output,
                "--device",
                device,
            ),
        )

    def train_stage2() -> None:
        command_stage(
            "train_stage2",
            _command(
                "train_stage2.py",
                "--config",
                config_path,
                "--train-pairs",
                paths["stage2_data"] / "train.jsonl",
                "--development-candidates-dir",
                paths["candidates"] / "development",
                "--output-dir",
                paths["cross_models"],
                "--device",
                device,
            ),
        )

    def select_development() -> None:
        source = paths["cross_models"] / "selected_config.json"
        if not source.is_file():
            raise FileNotFoundError(f"Stage-II training did not write {source}")
        write_json(paths["selection"], read_json(source))

    def score_test() -> None:
        score_dir = paths["scores"] / "test"
        command_stage(
            "score_test",
            _command(
                "build_timelines.py",
                "--config",
                config_path,
                "--partition",
                "test",
                "--candidates-dir",
                paths["candidates"] / "test",
                "--selection",
                paths["selection"],
                "--output",
                score_dir / "predictions.jsonl",
                "--scores-output-dir",
                score_dir,
                "--device",
                device,
            ),
        )

    def build_test_timelines() -> None:
        predictions = read_predictions(paths["scores"] / "test" / "predictions.jsonl")
        write_predictions(paths["timelines"] / "test_predictions.jsonl", predictions)
        write_crest_timelines(paths["timelines"] / "crest", predictions)

    def evaluate_test() -> None:
        command_stage(
            "evaluate_test",
            _command(
                "evaluate.py",
                "--config",
                config_path,
                "--partition",
                "test",
                "--predictions",
                paths["timelines"] / "test_predictions.jsonl",
                "--output",
                paths["evaluation"] / "test_metrics.json",
            ),
        )

    handlers = {
        "prepare_stage1": prepare_stage1,
        "train_stage1": train_stage1,
        "generate_train": lambda: generate("train"),
        "generate_development": lambda: generate("development"),
        "generate_test": lambda: generate("test"),
        "cluster_all": cluster_all,
        "prepare_stage2": prepare_stage2,
        "train_stage2": train_stage2,
        "select_development": select_development,
        "score_test": score_test,
        "build_test_timelines": build_test_timelines,
        "evaluate_test": evaluate_test,
    }
    complete = {
        "prepare_stage1": lambda: (paths["stage1_data"] / "train.jsonl").is_file(),
        "train_stage1": lambda: (paths["stage1_adapter"] / "adapter_config.json").is_file(),
        "generate_train": lambda: (paths["mentions"] / "train" / "_meta" / "summary.json").is_file(),
        "generate_development": lambda: (paths["mentions"] / "development" / "_meta" / "summary.json").is_file(),
        "generate_test": lambda: (paths["mentions"] / "test" / "_meta" / "summary.json").is_file(),
        "cluster_all": lambda: _all_exist(
            *(paths["candidates"] / part / "_meta" / "summary.json" for part in ("train", "development", "test"))
        ),
        "prepare_stage2": lambda: (paths["stage2_data"] / "train.jsonl").is_file(),
        "train_stage2": lambda: (paths["cross_models"] / "selected_config.json").is_file(),
        "select_development": lambda: paths["selection"].is_file(),
        "score_test": lambda: (paths["scores"] / "test" / "predictions.jsonl").is_file(),
        "build_test_timelines": lambda: (paths["timelines"] / "test_predictions.jsonl").is_file(),
        "evaluate_test": lambda: (paths["evaluation"] / "test_metrics.json").is_file(),
    }
    return handlers, complete


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if args.check_only:
        print(json.dumps(preflight(config), ensure_ascii=False, indent=2), flush=True)
        return 0
    if not args.run_dir:
        raise ValueError("--run-dir is required unless --check-only is used")
    if (args.from_stage or args.stop_after) and not args.resume and args.from_stage:
        raise ValueError("--from-stage requires --resume")

    stages = select_stages(args.from_stage, args.stop_after)
    run_dir = Path(args.run_dir).expanduser().resolve()
    _prepare_run_directory(run_dir, resume=args.resume)
    frozen_config = run_dir / "config.yaml"
    if not frozen_config.exists():
        shutil.copyfile(config_path, frozen_config)
    handlers, complete = _handlers(
        config_path,
        run_dir,
        args.device,
        resume=args.resume,
    )
    for stage in stages:
        if args.resume and complete[stage]():
            print(f"SKIP {stage}: output already exists", flush=True)
            continue
        print(f"START {stage}", flush=True)
        handlers[stage]()
        if not complete[stage]():
            raise RuntimeError(f"stage {stage} returned without its expected output")
        print(f"DONE {stage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
