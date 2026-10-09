#!/usr/bin/env python3
"""Extract fixed-window throughput and quality metrics from TensorBoard logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean
from typing import Any

PERF_START = 50
PERF_STOP = 500
QUALITY_COUNT = 20
RAW_TAGS = (
    "Perf/total_fps",
    "Perf/iteration_time",
    "Perf/collection_time",
    "Perf/learning_time",
    "Perf/collector_env_step_ms",
    "Perf/dp_gradient_sync_calls_per_rank",
    "Train/mean_reward",
    "Train/mean_episode_length",
)


def scalar_values(accumulator: Any, tag: str) -> list[float]:
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [event.value for event in accumulator.Scalars(tag)]


def mean_window(values: list[float], start: int, stop: int) -> float | None:
    window = values[start:stop]
    return fmean(window) if window else None


def latest_event_file(run_dir: Path) -> Path:
    files = list(run_dir.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file below {run_dir}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def extract_run(root: Path, spec: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = root / spec["run_name"]
    event_file = latest_event_file(run_dir)
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    values = {tag: scalar_values(accumulator, tag) for tag in RAW_TAGS}
    fps = values["Perf/total_fps"]
    if len(fps) != PERF_STOP:
        raise ValueError(f"{spec['run_name']}: expected 500 FPS samples, got {len(fps)}")

    iteration_s = mean_window(values["Perf/iteration_time"], PERF_START, PERF_STOP)
    collection_s = mean_window(values["Perf/collection_time"], PERF_START, PERF_STOP)
    learning_s = mean_window(values["Perf/learning_time"], PERF_START, PERF_STOP)
    reward_tail = values["Train/mean_reward"][-QUALITY_COUNT:]
    episode_tail = values["Train/mean_episode_length"][-QUALITY_COUNT:]

    batch_size = int(spec["batch_size"]) if spec["batch_size"] else None
    updates = int(spec["updates_per_step"]) if spec["updates_per_step"] else None
    world_size = int(spec["world_size"])
    learner_rows_s = None
    if iteration_s and batch_size and updates:
        learner_rows_s = world_size * batch_size * updates / iteration_s

    summary = {
        **spec,
        "event_file": str(event_file.relative_to(root)),
        "fps_samples": len(fps),
        "mean_global_env_steps_s": format_optional(mean_window(fps, PERF_START, PERF_STOP)),
        "mean_iteration_ms": format_optional(iteration_s * 1000 if iteration_s else None),
        "mean_collection_ms": format_optional(collection_s * 1000 if collection_s else None),
        "mean_learning_ms": format_optional(learning_s * 1000 if learning_s else None),
        "mean_learner_rows_s": format_optional(learner_rows_s),
        "tail20_mean_reward": format_optional(fmean(reward_tail) if reward_tail else None),
        "tail20_mean_episode_length": format_optional(
            fmean(episode_tail) if episode_tail else None
        ),
    }

    raw_rows: list[dict[str, Any]] = []
    for index in range(len(fps)):
        row: dict[str, Any] = {"run_name": spec["run_name"], "index": index}
        for tag, tag_values in values.items():
            row[tag] = tag_values[index] if index < len(tag_values) else ""
        raw_rows.append(row)
    return summary, raw_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip manifest entries whose logs are stored on another host.",
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for spec in load_manifest(args.manifest):
        try:
            summary, run_rows = extract_run(args.root, spec)
        except FileNotFoundError:
            if args.skip_missing:
                continue
            raise
        summaries.append(summary)
        raw_rows.extend(run_rows)

    write_csv(args.output, summaries)
    if args.raw_output:
        write_csv(args.raw_output, raw_rows)
    print(f"wrote {len(summaries)} run summaries to {args.output}")


if __name__ == "__main__":
    main()
