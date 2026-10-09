#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_one=$script_dir/run_one.sh

# Values are per-rank env counts; keep them explicit across carrier A/B runs.
ppo_tasks=("g1_walk_flat:2048" "g1_flip_tracking:1024" "go2_joystick_flat:2048" "allegro_inhand:16384")
offpolicy_tasks=("g1_walk_flat:2048" "g1_motion_tracking:2048")
port=29700

for env_device in cpu cuda; do
  for spec in "${ppo_tasks[@]}"; do
    task=${spec%%:*}; envs=${spec##*:}
    "$run_one" single ppo "$task" "$envs" "ppo_single_${env_device}_${task}_500" 0 "$env_device"
    "$run_one" dual ppo "$task" "$envs" "ppo_dual_${env_device}_${task}_500" "$port" "$env_device"
    port=$((port + 1))
  done
  for algo in sac flashsac; do
    for spec in "${offpolicy_tasks[@]}"; do
      task=${spec%%:*}; envs=${spec##*:}
      "$run_one" single "$algo" "$task" "$envs" "${algo}_single_${env_device}_${task}_500" 0 "$env_device"
      "$run_one" dual "$algo" "$task" "$envs" "${algo}_dual_${env_device}_${task}_500" "$port" "$env_device"
      port=$((port + 1))
    done
  done
done
