#!/usr/bin/env python3
"""Print and validate the pinned runtime's final run_summary.json (stdlib only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def display(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def main() -> int:
    path = Path(sys.argv[1])
    expected_iterations, world_size, envs = map(int, sys.argv[2:5])
    algo = sys.argv[5]
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("expected a JSON object")
    except (OSError, ValueError) as exc:
        print(f"Cannot read final summary {path}: {exc}", file=sys.stderr)
        return 1

    # PPO stores the final zero-based index; SAC/FlashSAC store a count.
    raw_iterations = summary.get("completed_iterations")
    completed = raw_iterations + (1 if algo == "ppo" else 0) if isinstance(raw_iterations, int) else None
    rows = [
        ("Status", summary.get("status")),
        ("Algorithm / task", f"{summary.get('algo', algo)} / {summary.get('task', 'N/A')}"),
        ("Completed iterations", f"{display(completed)} / {expected_iterations:,}"),
        ("World size", world_size),
        ("Environments per rank / global", f"{envs:,} / {envs * world_size:,}"),
        ("Training wall time (s)", summary.get("training_wall_time_sec")),
        ("Runtime wall time incl. setup (s)", summary.get("wall_time_sec")),
    ]
    if algo == "ppo":
        rows += [
            ("Global environment steps (this run)", summary.get("run_env_steps")),
            ("Global throughput, full training (steps/s)", summary.get("training_throughput_env_steps_per_sec")),
        ]
    else:
        rows += [
            ("Environment steps (runtime counter)", summary.get("total_env_steps")),
            ("Global throughput, final iteration (steps/s)", summary.get("final_env_steps_per_sec")),
            ("Global learner throughput, final iteration (rows/s)", summary.get("final_learner_replay_rows_per_sec")),
            ("Final iteration time (ms)", summary.get("final_cycle_wall_ms")),
        ]
    rows += [
        ("Final mean reward", summary.get("final_mean_reward")),
        ("Mean episode length", summary.get("mean_episode_length")),
        ("Last checkpoint", summary.get("last_checkpoint")),
        ("Summary JSON", path),
    ]
    print("\n=== Final training statistics (rank0) ===")
    for label, value in rows:
        print(f"{label}: {display(value)}")
    print("For report comparisons, use extract_metrics.py: iterations 50-499, quality tail 20.")
    if summary.get("status") != "completed" or completed != expected_iterations:
        print("Final summary does not confirm all requested iterations completed.", file=sys.stderr)
        return 1
    if summary.get("algo") != algo:
        print("Final summary algorithm does not match the launched run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
