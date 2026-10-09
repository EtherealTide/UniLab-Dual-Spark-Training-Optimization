# UniLab Dual-Spark Training Optimization

Multi-node UniLab training on two NVIDIA DGX Spark systems (one GB10 GPU per
node, 200 Gb/s QSFP direct link), with a complete 500-iteration experiment matrix
and a pinned reproduction bundle. Workloads include PPO, SAC, FlashSAC, G1
locomotion/motion tracking, Go2 joystick, and Allegro in-hand MuJoCo tasks.

## Results: absolute throughput

All results are from real training. Iterations 0-49 are warmup; reported
throughput is the mean over iterations 50-499. Dual-node throughput is the
global total across both nodes, not throughput per rank.

### PPO: global environment steps/s

| Task | Environments per node | Single-node steps/s | Dual-node global steps/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `g1_walk_flat` | 2,048 | 49,160.65 | **83,567.30** | 1.70x |
| `g1_flip_tracking` | 1,024 | 22,861.84 | **43,038.46** | 1.88x |
| `go2_joystick_flat` | 2,048 | 66,808.68 | **133,751.39** | 2.00x |
| `allegro_inhand` | 16,384 | 42,643.60 | **91,439.72** | 2.14x* |

\* Dual-node Allegro reached a later policy stage within 500 iterations. The
changed state distribution reduced collection time. The 2.14x result is real
end-to-end training throughput, not evidence of hardware-only superlinear scaling.

### Off-policy: environment and learner throughput

Each rank uses 2,048 environments, batch size 8,192, eight critic updates, and
two actor updates per iteration.

| Workload | Single-node env steps/s | Dual-node global env steps/s | Single-node learner rows/s | Dual-node global learner rows/s |
| --- | ---: | ---: | ---: | ---: |
| SAC `g1_walk_flat` | 38,496.57 | **64,022.10** | 1,231,354 | **2,041,779** |
| SAC `g1_motion_tracking` | 28,471.35 | **51,320.66** | 910,271 | **1,638,196** |
| FlashSAC `g1_walk_flat` | 8,742.86 | **16,474.44** | 279,748 | **526,023** |
| FlashSAC `g1_motion_tracking` | 8,355.22 | **15,759.63** | 267,349 | **503,298** |

Timing breakdowns, quality metrics, baseline comparisons, and interpretation:

- [`reports/dual_spark_validation_optimized_2026-10.md`](reports/dual_spark_validation_optimized_2026-10.md)
- [`reports/dual_spark_optimization_2026-10.md`](reports/dual_spark_optimization_2026-10.md)
- [`data/throughput_500.csv`](data/throughput_500.csv)
- [`data/quality_500.csv`](data/quality_500.csv)
- [`data/metrics_500_extracted.csv`](data/metrics_500_extracted.csv): full-precision extractor output.
- [`data/metrics_500_per_iteration.csv`](data/metrics_500_per_iteration.csv): 8,000 per-iteration records.

### What the speedup means

The dual/single ratio measures **weak scaling within the optimized runtime**:
every rank retains the single-node workload, so two nodes process twice the
global workload per iteration. It is not the speedup of the optimization patch
over the September code, or a measurement of time to a target reward.

Compared with the September report, PPO **dual-node absolute throughput** changes
by +2.39% (G1 walk), +4.07% (G1 flip), +0.93% (Go2), and -1.33% (Allegro).
These one-run differences do not establish a statistically reliable improvement.
Historical off-policy comparisons also change metric units, some workload
settings, and the integrated runtime. The reported 1.9-3.8x historical iteration
time ratios cannot be attributed entirely to RoCE or the DP optimization patch.
See the [historical comparison audit](reports/historical_comparison_2026-10.md).

## Optimizations

- Replace NCCL TCP with 200 Gb/s RoCE; retain P2P and SHM.
- Capture NCCL collectives inside FlashSAC whole-cycle CUDA Graphs.
- Use persistent flat gradient bucket views to remove repeated packing/unpacking.
- Combine SAC critic + alpha synchronization and FlashSAC actor + temperature synchronization.
- Reduce default collective counts from 18/12 to 10 per iteration.
- Use a cross-rank finite-loss sentinel to keep optimizer gates consistent for NaN/Inf.

## Exact version pins

[`versions.env`](versions.env) is the machine-readable source of version pins.

| Component | Reproduction version |
| --- | --- |
| UniLab | `feat/dual-spark` pin `d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3` + bundled patches; optimized experiment tree corresponds to commit `139524ac893efecf76feaf827e7236cb77307b33`; package `1.3.2` |
| UniSim | tag `v1.7.4`, commit `b48e91bbc62603299580a951c142a13c33bedae9`; package `unisim-core==1.7.4` |
| unilab-rl | `feat/dual-spark` pin `a3ed997d5c25ff708d674778782bc1be08a53e15` + tested main-runtime integration patch + optimization patch; experiment tree corresponds to commit `385a69f6d74bcfd9453bd2ed0c2d0687ae6c5556` |
| PyTorch | `2.9.0+cu130` |
| MuJoCo | `3.11.0` |

The measured experiment tree is preserved separately from
`UNILAB_RUNTIME_TREE` in `versions.env`. The current runtime additionally applies
UniLab patch `0003`: native launcher lifecycle/logging fixes and preservation
of the existing loggers' terminal output. These fixes were validated with short functional runs, not a new
500-iteration performance matrix. Exported benchmark data remains unchanged.

The public branch pins and actual experiment trees are recorded separately.
UniLab patches also include required off-policy logging fixes. The first
unilab-rl patch integrates the public branch with the tested main runtime; the
second adds DP/CUDA Graph optimizations. Patch provenance and final tree hashes
are documented in [`patches/README.md`](patches/README.md).

## Repository layout

```text
.
├── README.md
├── SHA256SUMS                    # Patch, report, and exported-data checksums
├── versions.env                 # Exact version pins
├── config/cluster.env.example   # SSH hosts, rendezvous IP, network devices
├── data/                        # Summary and per-iteration metrics
├── patches/                     # UniLab / unilab-rl patches
├── reports/                     # Validation and optimization reports
├── tests/                       # Launcher and summary regression tests
└── scripts/
    ├── bootstrap.sh             # Clone, patch, install, verify
    ├── run_one.sh               # Compact single-/dual-node training launcher
    ├── filter_console.awk       # Compact progress / optional full output
    ├── print_run_summary.py     # Final statistics and completion verification
    ├── run_matrix.sh            # Full 16-run matrix
    └── extract_metrics.py       # Fixed-window TensorBoard metrics
```

## Prerequisites

Both Spark nodes need Ubuntu aarch64, Python 3.12, `uv`, Git, Bash, the pinned
PyTorch runtime, matching training-code paths and asset caches, and an active
200 Gb/s interface/RoCE HCA (for example `enp1s0f1np1` / `rocep1s0f1`).

The **coordinator** is whichever machine runs `scripts/run_one.sh` or
`scripts/run_matrix.sh`. It can be `HOST0` (the master/rank0 Spark), `HOST1`, or a
separate Linux machine. It needs passwordless SSH to every participating node.
Even `single` mode uses SSH to `HOST0`, including when the coordinator is `HOST0`.

Verify the direct link on the Spark nodes:

```bash
ip -br addr show enp1s0f1np1
ibv_devinfo -d rocep1s0f1
iperf3 -c <peer-200g-ip> -P 4
```

Run one workload at a time when comparing throughput. Concurrent training,
inference servers, or stalled distributed jobs can affect CPU/GPU availability.

## 1. Install the pinned runtime on both Spark nodes

Run on **each Spark**:

```bash
git clone https://github.com/EtherealTide/UniLab-Dual-Spark-Training-Optimization.git
cd UniLab-Dual-Spark-Training-Optimization
bash scripts/bootstrap.sh /home/nvidia/unilab-dual-spark-repro
```

The bootstrap script clones the three upstream repositories, checks out the
base pins, applies bundled patches, verifies source trees, creates the frozen
UniLab `.venv`, installs the pinned UniSim/unilab-rl sources, and checks package
versions. It refuses to overwrite dirty repositories.

The bundle directory (for example `~/Desktop/UniLab-Dual-Spark-Training-Optimization`)
contains orchestration scripts and patches. The training runtime lives at
`/home/nvidia/unilab-dual-spark-repro/UniLab`.

## 2. Configure the coordinator

In the bundle checkout on the **machine that will launch the experiment**:

```bash
cp config/cluster.env.example config/cluster.env
```

Edit these settings for your cluster:

```bash
HOST0=nvidia@192.168.110.48
HOST1=nvidia@192.168.110.40
MASTER_ADDR=10.77.0.1
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_HCA=rocep1s0f1
REMOTE_ROOT=/home/nvidia/unilab-dual-spark-repro
MAX_ITERATIONS=500
```

`HOST0`/`HOST1` are SSH destinations; `MASTER_ADDR` is the master node's IP on
the direct training link. `REMOTE_ROOT` is an absolute path to the bootstrapped
runtime, identical on both Spark nodes. Workers do not need `cluster.env` unless
you also launch the orchestration script from them. Use `CLUSTER_FILE=/path/to/cluster.env`
to select a different coordinator configuration.

### Configure passwordless SSH from the coordinator

SSH access from your laptop to a Spark does not configure SSH from that Spark to
itself or its peer. Run the following **on the coordinator**, adjusting hosts:

```bash
test -f ~/.ssh/id_ed25519 || \
  ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
ssh-copy-id -i ~/.ssh/id_ed25519.pub nvidia@192.168.110.48
ssh-copy-id -i ~/.ssh/id_ed25519.pub nvidia@192.168.110.40

ssh -o BatchMode=yes -o ConnectTimeout=10 nvidia@192.168.110.48 hostname
ssh -o BatchMode=yes -o ConnectTimeout=10 nvidia@192.168.110.40 hostname
```

Both checks must return the expected hostname without a password prompt.
The launcher respects `~/.ssh/config`, so SSH aliases such as `HOST0=dgx1` and
`HOST1=dgx2`, custom ports, and `IdentityFile` settings work when configured on
the coordinator. `BatchMode=yes` makes authentication failures explicit;
`ConnectTimeout=10` bounds connection setup, not training duration.

## 3. Launch an experiment

Argument order: `mode algo task envs run_name port`.

```bash
# Single-node PPO, 500 iterations (port is unused in single mode).
bash scripts/run_one.sh single ppo go2_joystick_flat 2048 \
  ppo_single_go2_500 0

# Dual-node PPO: launch once from the coordinator, not once per Spark.
bash scripts/run_one.sh dual ppo go2_joystick_flat 2048 \
  ppo_dual_go2_500 29705

# Dual-node SAC, fixed batch/update settings.
bash scripts/run_one.sh dual sac g1_walk_flat 2048 \
  sac_dual_g1_walk_500 29700

# Dual-node FlashSAC.
bash scripts/run_one.sh dual flashsac g1_motion_tracking 2048 \
  flashsac_dual_g1_motion_500 29703
```

Use a **new `run_name` for every attempt**, for example
`ppo_single_go2_500_trial2`. Preflight checks SSH, the runtime Python executable,
and the absence of an existing run directory on all participating nodes before
launching training. This prevents stale summaries and mixed TensorBoard runs.

For a short functional smoke test:

```bash
MAX_ITERATIONS=2 bash scripts/run_one.sh single ppo go2_joystick_flat 64 \
  ppo_single_go2_smoke_trial1 0
```

An environment `MAX_ITERATIONS` override takes precedence over the config file.
Smoke-test throughput is not a benchmark.

### Console output and saved logs

The default `CONSOLE_MODE=compact` prints startup stages, full-log viewing
commands, a short progress line at most once every 30 seconds per rank, rank exit
markers, and final statistics. The first and final iteration are always shown
when the runtime emits a recognized progress line. Model definitions, individual
loss/reward fields, compiler messages, and warnings stay in the full log. On a
failure, the launcher also prints the last 20 log lines for diagnosis.

Use `PROGRESS_INTERVAL=60` to reduce progress updates, or opt into every log line:

```bash
CONSOLE_MODE=full bash scripts/run_one.sh dual ppo go2_joystick_flat 2048 \
  ppo_dual_go2_500_verbose_trial1 29705
```

Python still runs unbuffered, with `training.log_interval=1`, so saved logs remain
complete and live in both modes. Console lines carry `[single]`, `[rank0]`, or
`[rank1]` prefixes. NCCL initialization and compilation may take time before the
first iteration.

For **PPO**, rank0 owns iteration metrics, TensorBoard events, and checkpoints.
Rank1 can successfully complete an entire run without printing any iteration
metrics: its last runtime message may remain `Synchronizing parameters for rank 1...`.
That message records the start of synchronization, not the worker's current state.
The launcher appends `[launcher] rank1 exited (code=0)` to its log when it exits
successfully. Check rank0's advancing iterations and final summary to establish
training health; rank1 silence alone cannot distinguish training from a stalled
collective. A missing marker on an older log is expected because older launchers
did not write one. Nonzero exit markers indicate failure.

The same output is saved using `tee` on the training nodes:

| Mode | Node | Console log |
| --- | --- | --- |
| single | `HOST0` | `/tmp/<run_name>_single.log` |
| dual | `HOST0` | `/tmp/<run_name>_rank0.log` |
| dual | `HOST1` | `/tmp/<run_name>_rank1.log` |

At startup the launcher prints a copyable command for each full log, for example:

```bash
# Run from another terminal on the coordinator.
ssh nvidia@192.168.110.48 'tail -n 40 -F /tmp/ppo_dual_go2_500_rank0.log'
ssh nvidia@192.168.110.40 'tail -n 40 -F /tmp/ppo_dual_go2_500_rank1.log'
```

TensorBoard events, checkpoints, `run_config.json`, and `run_summary.json` live
under `$REMOTE_ROOT/UniLab/logs/<run_name>` on `HOST0`. The formatter is sent over
SSH stdin, so workers do not need an updated bundle checkout just to print stats.

To view a saved log from another terminal on the corresponding Spark:

```bash
tail -n 40 -F /tmp/ppo_single_go2_500_single.log
```

This prints the last 40 lines and follows new lines. `Ctrl+C` in the `tail`
terminal only stops the log viewer. `/tmp` logs may be removed on reboot.

### Final statistics

After every rank exits successfully, the launcher reads rank0's summary and prints:

- Completion status and completed iterations as a count (PPO's stored index is zero-based).
- World size, environments per rank, and global environment count.
- Training wall time and runtime wall time including setup.
- PPO global environment steps and full-training throughput.
- SAC/FlashSAC final-iteration global environment and learner throughput, plus cycle time.
- Final mean reward, mean episode length, checkpoint path, and summary path.

Missing optional fields are shown as `N/A`. `DONE:<run_name>:0` is printed only
after successful training and a matching completed summary. Training failures
retain their nonzero exit status through the logging pipeline and print `FAILED`.
If one distributed rank fails, its peer may remain in a collective until the
runtime timeout; inspect both logs and stop the matching launchers on both nodes
before retrying. Closing an SSH terminal is not a reliable cleanup method.

The printed PPO full-training average and off-policy final-iteration rates use
different windows from the tables above. Use the fixed-window extractor below
for report comparisons.

### Full matrix

```bash
bash scripts/run_matrix.sh
```

The matrix runs eight single-node and eight dual-node workloads sequentially,
with compact progress and final statistics for each. It uses fixed run names and is
intended for a fresh logs directory; existing runs trigger the same preflight
protection. Rank0 is the only TensorBoard/checkpoint writer.

## 3A. Train directly from the UniLab checkout

The commands below use the **bootstrapped UniLab runtime**, without this
bundle's `run_one.sh`. Run them from `HOST0` (`spark-0a2a`), where rank0 runs
locally. The UniLab launcher starts rank1 through SSH when `--num-nodes 2` is
selected; run it once, not once per node. Both nodes still need the pinned,
patched runtime and matching paths from step 1. These RoCE flags are provided by
the bundled UniLab patch; a checkout of the unpatched branch pin is insufficient.

```bash
cd /home/nvidia/unilab-dual-spark-repro/UniLab
export PATH="$HOME/.local/bin:$PATH"
uv run --no-sync scripts/launch_distributed.py --help
```

`--no-sync` preserves the runtime already verified by `bootstrap.sh`, including
the editable UniSim/unilab-rl installs. Running dependency synchronization again
can replace them with the versions in UniLab's original lockfile.

### Single-node PPO: native UniLab launcher

```bash
run_name="ppo_single_go2_500_native_$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 uv run --no-sync scripts/launch_distributed.py \
  --algo ppo --task go2_joystick_flat --sim mujoco \
  --log-dir "logs/$run_name" \
  algo.num_envs=2048 algo.max_iterations=500 ++training.log_interval=1
```

Single-node mode runs locally and does not need self-SSH or `cluster.env`.

### Dual-node PPO: native UniLab launcher

```bash
run_name="ppo_dual_go2_500_native_$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 uv run --no-sync scripts/launch_distributed.py \
  --algo ppo --task go2_joystick_flat --sim mujoco \
  --num-nodes 2 --peer nvidia@192.168.110.40 \
  --remote-dir /home/nvidia/unilab-dual-spark-repro/UniLab \
  --master-ip 10.77.0.1 --ifname enp1s0f1np1 --port 29705 \
  --nccl-transport ib --nccl-ib-hca rocep1s0f1 \
  --log-dir "logs/$run_name" \
  algo.num_envs=2048 algo.max_iterations=500 ++training.log_interval=1
```

`--peer` is the worker's SSH destination; `--master-ip` is rank0's direct-link IP.
Use a free rendezvous port. The launcher's default NCCL transport is TCP, so
`--nccl-transport ib` and `--nccl-ib-hca` are required for this RoCE experiment.

### Dual-node SAC and FlashSAC: native UniLab launcher

```bash
# SAC: use --dp-port for the off-policy rendezvous.
run_name="sac_dual_g1_walk_500_native_$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 uv run --no-sync scripts/launch_distributed.py \
  --algo sac --task g1_walk_flat --sim mujoco \
  --num-nodes 2 --peer nvidia@192.168.110.40 \
  --remote-dir /home/nvidia/unilab-dual-spark-repro/UniLab \
  --master-ip 10.77.0.1 --ifname enp1s0f1np1 --dp-port 29700 \
  --nccl-transport ib --nccl-ib-hca rocep1s0f1 \
  --log-dir "logs/$run_name" \
  algo.num_envs=2048 algo.max_iterations=500 \
  algo.batch_size=8192 algo.updates_per_step=8 algo.policy_frequency=4 \
  ++training.log_interval=1

# FlashSAC: a separate run name and rendezvous port.
run_name="flashsac_dual_g1_motion_500_native_$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 uv run --no-sync scripts/launch_distributed.py \
  --algo flashsac --task g1_motion_tracking --sim mujoco \
  --num-nodes 2 --peer nvidia@192.168.110.40 \
  --remote-dir /home/nvidia/unilab-dual-spark-repro/UniLab \
  --master-ip 10.77.0.1 --ifname enp1s0f1np1 --dp-port 29703 \
  --nccl-transport ib --nccl-ib-hca rocep1s0f1 \
  --log-dir "logs/$run_name" \
  algo.num_envs=2048 algo.max_iterations=500 \
  algo.batch_size=8192 algo.updates_per_step=8 algo.policy_frequency=4 \
  ++training.log_interval=1
```

For single-node SAC/FlashSAC, omit the cluster options (`--num-nodes`, `--peer`,
`--remote-dir`, `--master-ip`, `--ifname`, ports, and NCCL options). Keep the same
algorithm-specific batch/update overrides.

### Call the training entrypoint directly

For a local single-node run, you can bypass the UniLab launcher as well:

```bash
run_name="ppo_single_go2_500_entry_$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 uv run --no-sync src/unilab/scripts/train_rsl_rl.py \
  task=go2_joystick_flat/mujoco training.log_dir="logs/$run_name" \
  training.no_play=true ++training.log_interval=1 \
  algo.num_envs=2048 algo.max_iterations=500
```

Use `src/unilab/scripts/train_sac.py` or `train_flashsac.py` for off-policy
training, with the matching task and batch/update overrides above.

### Dual-node SAC: direct Hydra entrypoints

This form calls the existing UniLab training script directly, like the
[single-host multi-card example](https://github.com/EtherealTide/UniLab-Multi-Card-Training-Optimization).
For two Spark hosts, run it **once on each host**. Both GPUs are locally
`cuda:0`; `training.devices=[0,1]` means two GPUs on one machine. External DP
instead reads the world size, rank and rendezvous URL below.

On **both nodes**, use the same fresh `run_name` and identical overrides. Set
`rank=0` on HOST0 (`spark-0a2a`), and `rank=1` on HOST1 (`spark-7e93`). The
example name must be changed for a subsequent attempt. Start both sessions
within the rendezvous timeout.

```bash
cd /home/nvidia/unilab-dual-spark-repro/UniLab
export PATH="$HOME/.local/bin:$PATH"
run_name="sac_dual_g1_walk_500_direct_trial1"
rank=0  # HOST0; use rank=1 on HOST1

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=rocep1s0f1 \
NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 \
UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE=2 UNILAB_DP_RANK="$rank" \
UNILAB_DP_RENDEZVOUS_URL=tcp://10.77.0.1:29700 \
UNILAB_DP_LOG_DIR="logs/$run_name" \
uvx uv@0.12.5 run --no-sync \
  src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco training.devices=null \
  algo.num_envs=2048 algo.batch_size=8192 \
  algo.updates_per_step=8 algo.policy_frequency=4 algo.max_iterations=500 \
  training.no_play=true training.trace_enabled=false ++training.log_interval=1
```

Port 29700 must be free on HOST0. `training.devices=null` leaves local device
selection to the external-DP path; `CUDA_VISIBLE_DEVICES=0` exposes the one
local GPU. Environment count and batch size remain **per rank**. `uvx` selects
the uv executable version; `--no-sync` preserves the pinned training dependencies.
Plain `uv run --no-sync` also works with the installed uv.

For FlashSAC, change the entrypoint to `train_flashsac.py`, use a fresh run
name/port, and keep both ranks' remaining configuration identical.
PPO multi-node launch uses torchrun and its RSL-RL logger; this external-DP SAC
command is not a PPO launch recipe.

The interactive SAC/FlashSAC display is the existing
`uni_rl.logging.OffPolicyLogger`: **UniLab Off-Policy Training**, Losses &
Metrics, Rewards, Learner, Collector, System, and Training Summary. It is not a
MuJoCo-specific logging backend. Rank0 renders/writes metrics and checkpoints;
rank1 may remain quiet during successful training.

Manual direct commands do not launch the peer or provide the native launcher's
all-rank supervision, SSH lifetime lease, rank-log capture or peer cleanup.
On failure/interruption, check and stop both matching sessions. To get automatic
peer launch and saved logs with the same native Rich display, use the UniLab
launcher examples above. Piping a direct command through `tee` removes its TTY
and can disable live/color rendering; the native launcher preserves it through
an isolated output PTY.

### Native-launcher output and statistics

With UniLab patch `0003` applied **on both nodes**, these native commands print
rank0 output and keep rank1 details in saved logs. SAC/FlashSAC use their existing
Rich Live panel and Training Summary in an interactive terminal. PPO retains
the original RSL-RL logger. The launcher preserves rank0 TTY detection and width
using an isolated output PTY, without replacing logger metrics/layouts.
Redirected output uses the original logger's non-terminal behavior.
The bundle launcher still defaults to compact output.

The native launcher automatically saves full stdout/stderr, including SSH errors:

```bash
# Both files are on HOST0; rank1 also saves a local copy on HOST1.
tail -n 40 -F "logs/$run_name/launcher/rank0.log"
tail -n 40 -F "logs/$run_name/launcher/rank1.log"
# For a saved colored log, use less -R.
```

SSH uses `-T` and an isolated stdin pipe, preserving the coordinator's TTY.
EOF on that pipe stops the remote rank process group. The launcher monitors
every rank: a nonzero exit prints the error-log tail, stops the remaining ranks,
and returns failure. Ctrl+C also stops the groups owned by this launch.
After the first successful rank exit, remaining ranks have 60 seconds to finish;
use `--completion-timeout 120` when checkpoint shutdown needs longer.
Successful completion requires every rank to exit successfully and a completed,
matching `run_summary.json`. Final statistics print automatically: the original
off-policy Rich Training Summary remains intact, followed by all-rank completion
confirmation; PPO prints its full-training statistics.
These summary windows differ from the report's iterations 50-499.

Use a fresh timestamped run name. A nonempty run directory is rejected.
Updating the bundle alone does not update an existing runtime: re-run step 1
with `UV_BIN=/home/nvidia/.local/bin/uv` after committing or preserving any local
runtime changes. Bootstrap refuses dirty repositories and verifies the new
`UNILAB_RUNTIME_TREE`. Do not apply `0003` repeatedly to a patched checkout.

For additional expected-iteration validation, the bundle formatter remains available:

```bash
# PPO dual-node example: expected iterations, world size, envs per rank, algo.
uv run --no-sync \
  /home/nvidia/Desktop/UniLab-Dual-Spark-Training-Optimization/scripts/print_run_summary.py \
  "logs/$run_name/run_summary.json" 500 2 2048 ppo
```

Use world size `1` for single-node runs; select `sac` or `flashsac` for the other
algorithms. For fixed-window throughput, use the extractor in step 4 with a
manifest containing the new run name.

## 4. Extract comparable absolute throughput

Run on `HOST0`, in the bundle checkout. Set `REMOTE_ROOT` to the path used for
bootstrap; this does not require copying the coordinator's SSH config to `HOST0`:

```bash
REMOTE_ROOT=/home/nvidia/unilab-dual-spark-repro
"$REMOTE_ROOT/UniLab/.venv/bin/python" scripts/extract_metrics.py \
  --root "$REMOTE_ROOT/UniLab/logs" \
  --manifest data/run_manifest.csv \
  --output metrics.csv
```

The manifest expects the standard 500-iteration run names. For custom names,
copy and edit the manifest. If logs are spread across hosts, run extraction on
each host using `--skip-missing` and merge the CSV outputs.

Extraction uses iterations 50-499 for performance and the last 20 records for
quality. Environment throughput uses `Perf/total_fps`; off-policy learner
throughput is `world_size * batch_size * updates_per_step / iteration_time`.
PPO `completed_iterations=499` denotes the final zero-based index. The event
file must contain 500 `Perf/total_fps` samples; the extractor validates this.

## Troubleshooting

- **Missing `cluster.env`:** create it in the bundle checkout on the machine
  executing the launcher, even if that machine is also the master.
- **SSH authentication/host-key failure:** complete the coordinator SSH checks
  before retrying. Single-node mode also requires SSH to `HOST0` itself.
- **Existing run directory:** choose a new run name; do not mix repeated attempts.
- **Rendezvous timeout:** verify both ranks launched, `MASTER_ADDR`, the selected
  port, and direct-link reachability. A `single` run cannot pair with a dual rank1.
- **Rank1 log ends at parameter synchronization:** for PPO this is normal if
  rank0 progresses or the summary confirms completion with `world_size=2`. Use
  the exit marker for new runs. If rank0 also stops advancing, inspect both logs
  and processes rather than assuming silence means success.
- **Low throughput:** check `nvitop`/`nvidia-smi` for concurrent workloads. A live
  process stuck in parameter synchronization is not a zombie; zombie status is
  `Z` in `ps`. Stop the identified job's `torchrun` launchers with `SIGTERM` first;
  use `SIGKILL` only for confirmed leftovers. Preserve unrelated jobs and desktop services.
- **Old completed summary during a new run:** summaries are finalized at exit.
  Follow the current console log; fresh run names prevent this ambiguity.

## Result boundaries

- 500 iterations validate throughput and training health, not final convergence
  across multiple seeds and 5,000-10,000 iterations.
- Allegro and G1 walk reach different policy stages in single-/dual-node runs;
  throughput includes state-distribution feedback.
- NCCL uses `NET/IB`, but the experiments still use `GDR 0`; no driver/kernel
  module changes were made.
- GB10 compute capability 12.1 can trigger Triton/PTXAS `sm_121a` autotune fallback
  messages. Training can complete after cache warmup.

## Manual patch application

If you do not use `bootstrap.sh`:

```bash
git clone https://github.com/Motphys/UniLab.git
git -C UniLab checkout d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3
git -C UniLab apply ../patches/UniLab/*.patch

git clone https://github.com/unilabsim/unilab_rl.git
git -C unilab_rl checkout a3ed997d5c25ff708d674778782bc1be08a53e15
git -C unilab_rl apply ../patches/unilab_rl/*.patch
```

See [`patches/README.md`](patches/README.md) for provenance and tree checks.

## Launcher regression tests

On Linux, without SSH access or a GPU:

```bash
python3 -m unittest discover -s tests -v
```

These tests exercise compact/full output, progress throttling, rank exit markers,
live streaming, exit-code propagation, SSH preflight, stale-run protection, and
PPO/SAC/FlashSAC summary counting.
