#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_one=$script_dir/run_one.sh

# Single-node baselines.
"$run_one" single ppo g1_walk_flat 2048 ppo_single_g1_walk_500 0
"$run_one" single ppo go2_joystick_flat 2048 ppo_single_go2_500 0
"$run_one" single sac g1_walk_flat 2048 sac_single_g1_walk_500 0
"$run_one" single sac g1_motion_tracking 2048 sac_single_g1_motion_500 0
"$run_one" single flashsac g1_walk_flat 2048 flashsac_single_g1_walk_500 0
"$run_one" single flashsac g1_motion_tracking 2048 flashsac_single_g1_motion_500 0

# Dual-node runs. Each workload uses a dedicated rendezvous port.
"$run_one" dual sac g1_walk_flat 2048 sac_dual_g1_walk_500 29700
"$run_one" dual sac g1_motion_tracking 2048 sac_dual_g1_motion_500 29701
"$run_one" dual flashsac g1_walk_flat 2048 flashsac_dual_g1_walk_500 29702
"$run_one" dual flashsac g1_motion_tracking 2048 flashsac_dual_g1_motion_500 29703
"$run_one" dual ppo g1_walk_flat 2048 ppo_dual_g1_walk_500 29704
"$run_one" dual ppo go2_joystick_flat 2048 ppo_dual_go2_500 29705
