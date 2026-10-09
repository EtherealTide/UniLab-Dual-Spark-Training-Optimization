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
    ├── run_one.sh               # Live single-/dual-node training launcher
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

## 3. Launch an experiment with live output

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

Python runs unbuffered, with `training.log_interval=1`. Output appears live with
`[single]`, `[rank0]`, or `[rank1]` prefixes. NCCL/distributed initialization and
compilation may take time before the first iteration; rank1 generally prints
initialization messages while rank0 owns iteration metrics and checkpoints.

The same output is saved using `tee` on the training nodes:

| Mode | Node | Console log |
| --- | --- | --- |
| single | `HOST0` | `/tmp/<run_name>_single.log` |
| dual | `HOST0` | `/tmp/<run_name>_rank0.log` |
| dual | `HOST1` | `/tmp/<run_name>_rank1.log` |

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
with live output and final statistics for each. It uses fixed run names and is
intended for a fresh logs directory; existing runs trigger the same preflight
protection. Rank0 is the only TensorBoard/checkpoint writer.

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

These tests exercise live streaming, both rank labels, exit-code propagation,
SSH preflight, stale-run protection, and PPO/SAC/FlashSAC summary counting.
