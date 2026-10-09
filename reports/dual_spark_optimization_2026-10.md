# Dual-DGX-Spark Training: Optimization and Measurements

Experiment date: 2026-10-08. Interpretation and reproduction instructions updated: 2026-10-09.

This report covers PPO, SAC, and FlashSAC training with MuJoCo on two NVIDIA
DGX Spark hosts, each with one GPU. The workloads include G1 locomotion and
motion tracking, Go2 joystick control, and Allegro in-hand manipulation.

The tables measure weak scaling within the October runtime. Historical
off-policy comparisons include changes to the runtime and workload, so their
iteration-time ratios do not isolate the effect of the optimization patch.
PPO dual-node throughput differs from September by +2.39%, +4.07%, +0.93%, and
-1.33% across the four tasks. See the
[historical comparison](historical_comparison_2026-10.md) for the baseline
differences. The exported measurements have not been changed.

The public source pins are:

- UniLab: `feat/dual-spark@d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3`.
- unilab-rl: `feat/dual-spark@a3ed997d5c25ff708d674778782bc1be08a53e15`.
- UniSim: `v1.7.4@b48e91bbc62603299580a951c142a13c33bedae9`.

The reproduction bundle applies its ordered patches to these pins to rebuild
the experiment code.

## 1. Workloads and measurement method

The test matrix covers the tasks in the September report:

- PPO: `g1_walk_flat`, `g1_flip_tracking`, `go2_joystick_flat`, and `allegro_inhand`.
- SAC: `g1_walk_flat` and `g1_motion_tracking`.
- FlashSAC: `g1_walk_flat` and `g1_motion_tracking`.

Each workload has one single-node run and one dual-node run of 500 iterations:
16 runs in total. Throughput is the mean of iterations 50-499, after excluding
the first 50 iterations. Reward and episode length are the means of the last
20 TensorBoard records.

PPO and off-policy throughput use global environment steps/s. Off-policy
tables also report global learner rows/s. Each rank retains the single-node
environment and batch sizes, so a dual-node iteration processes twice the
global samples. The dual/single throughput ratio is a weak-scaling ratio.

PPO's `completed_iterations=499` is a zero-based index. All 16 runs contain
500 TensorBoard `Perf/total_fps` records.

The off-policy workload uses these per-rank settings:
`num_envs=2048`, `batch_size=8192`, `updates_per_step=8`, and
`policy_frequency=4`. Each iteration performs eight critic updates and two
actor updates.

## 2. Bottlenecks in the earlier configuration

### 2.1 NCCL transport

The earlier launch profile set:

```bash
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
```

NCCL used sockets despite an active 200 Gb/s RoCE HCA, `rocep1s0f1`.
A FlashSAC diagnostic run reported about 43.36 ms for ten collectives per
iteration. The timing estimator for that value was not recorded.

### 2.2 FlashSAC CUDA Graph support

The earlier FlashSAC NVIDIA fast path rejected the distributed gradient
callback. Dual-node training therefore could not use the same whole-cycle
CUDA Graph path as single-node training.

### 2.3 Gradient copies and collective count

Each optimizer update copied parameter gradients into a flat buffer, reduced
that buffer, and copied the result back. The earlier update paths used
18 synchronization calls per SAC iteration and 12 per FlashSAC iteration.
These costs matter for learners whose iterations take roughly 50-90 ms.

## 3. Implementation changes

### 3.1 Explicit RoCE configuration

UniLab's `scripts/launch_distributed.py` adds:

- `--nccl-transport {tcp,auto,ib}` and `--nccl-ib-hca`.
- `NCCL_IB_DISABLE=0` and `NCCL_IB_HCA=rocep1s0f1` in the tested IB profile.
- `NCCL_P2P_DISABLE=0` and `NCCL_SHM_DISABLE=0` for the tested topology.
- TCP as the default compatibility profile.

`DpParameterSync.start()` now defaults to disabling P2P/SHM only for the
single-host FileStore path. It no longer forces those settings for a multi-node
TCPStore. Each Spark has one GPU; there is no local multi-GPU peer connection
in this topology. The transport diagnostics below changed IB, P2P, and SHM
together and do not isolate the contribution of each setting.

### 3.2 NCCL in the FlashSAC graph cycle

Graph warmup, capture, and replay hooks allow FlashSAC to capture all-reduce
operations as CUDA Graph nodes. Collective accounting runs before replay.
Capture failures retain the eager fallback.

Synchronization also carries finite-loss sentinels. If any rank reports a
non-finite loss, the corresponding optimizer skips its update on all ranks.
This keeps optimizer gating consistent across ranks.

### 3.3 Persistent gradient bucket views

CUDA Graph replay requires stable gradient storage addresses. During graph
warmup, the DP layer binds each parameter's `.grad` to a non-overlapping view
of the flat collective buffer. Later backward passes write directly to that
buffer, and replay reduces it in place, avoiding per-parameter packing and
unpacking.

The eager compatibility path still copies gradients. Finite-loss sentinels
remain copy-only so a scalar cannot alias multiple buckets. The September
report attributed roughly 20-26 ms/iteration to the DP wrapper, gradient packing,
and disruption of the asynchronous pipeline; storage reuse targets those costs.

### 3.4 Collective fusion

For the tested update schedule:

- SAC combines critic and alpha gradients: 18 to 10 collectives/iteration.
- FlashSAC combines actor and temperature gradients: 12 to 10 collectives/iteration.
- Each optimizer retains its own finite-loss sentinel and step gate.

The combined parameter groups have independent losses and optimizers. Combining
their transport reduces collective boundaries without changing the gradients
being averaged.

## 4. Diagnostic comparisons

### 4.1 Transport

| FlashSAC configuration | Collectives/iteration | DP sync time | Iteration time | Global env steps/s |
| --- | ---: | ---: | ---: | ---: |
| TCP; IB/P2P/SHM disabled | 10 | 43.36 ms | ~279.15 ms | ~14.67k |
| RoCE; P2P/SHM enabled | 10 | 11.43 ms median | 248.16 ms | 16.54k |

These are author-reported diagnostic observations. The first row has no stated
timing estimator; the second uses a median. Their ratio is not a verified
percentage reduction. Raw A/B artifacts are not included in the bundle.
Reported NCCL diagnostics show `NET/IB`, `rocep1s0f1`, 200000 Mb/s, and `GDR 0`.

### 4.2 SAC collective fusion

| Configuration | Mean iteration time, 50-299 | Env steps/s | Collectives/iteration |
| --- | ---: | ---: | ---: |
| Separate collectives | 65.57 ms | 62.74k | 18 |
| Fused critic + alpha | 64.37 ms | 63.85k | 10 |

The observations differ by about 1.8% over a 250-iteration window. The raw
artifacts and uncertainty across repeated runs are not bundled, so this remains
a diagnostic result rather than a demonstrated repeatable gain.

### 4.3 Historical iteration times

| Workload | September single-node | October single-node | September dual-node | October dual-node |
| --- | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 91.0 ms | 53.22 ms | 127.9 ms | 64.19 ms |
| SAC g1_motion_tracking | 64.5 ms | 72.00 ms | 92.0 ms | 80.01 ms |
| FlashSAC g1_walk_flat | 901.4 ms | 234.27 ms | 945.4 ms | 249.18 ms |
| FlashSAC g1_motion_tracking | 471.2 ms | 245.13 ms | 494.7 ms | 260.43 ms |

FlashSAC's historical iteration-time ratios are about 1.9-3.8x. They span
different runtimes and workloads: FlashSAC walk changes from 4096 to 2048
environments per rank, and the historical resolved update schedules are missing.
The full reduction cannot be attributed to RoCE or DP changes. The
[comparison report](historical_comparison_2026-10.md) separates these issues
from the October scaling measurements.

## 5. Results from the 500-iteration matrix

### 5.1 PPO

Throughput is global environment steps/s.

| Task | Envs/rank | Single-node | Dual-node | October scaling | September scaling |
| --- | ---: | ---: | ---: | ---: | ---: |
| g1_walk_flat | 2048 | 49.16k | 83.57k | 1.70× | 1.87× |
| g1_flip_tracking | 1024 | 22.86k | 43.04k | 1.88× | 1.78× |
| go2_joystick_flat | 2048 | 66.81k | 133.75k | 2.00× | 1.93× |
| allegro_inhand | 16384 | 42.64k | 91.44k | 2.14× | 1.97× |

Go2's final episode length is 1000 on both single and dual nodes, with a measured
ratio of 2.00×. G1 walk reaches different policy stages: final episode length is
about 890 on two nodes versus 77 on one. The resulting simulation workloads
differ, and the 500-iteration ratio is 1.70×, below the earlier 200-iteration
observation of 1.98×.

Allegro's dual-node reward/episode length is 11.74/399.63, versus 3.10/133.85
on one node. Collection time falls from 2738.69 to 2402.34 ms. Its 2.14× ratio
therefore includes changes in policy stage and environment state distribution;
it is not a hardware-only superlinear result. The report keeps the full window
for all tasks, including G1 walk's lower ratio.

### 5.2 SAC and FlashSAC

| Workload | Single-node iteration | Dual-node iteration | Single-node env steps/s | Dual-node env steps/s | Scaling |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 53.22 ms | 64.19 ms | 38.50k | 64.02k | 1.66× |
| SAC g1_motion_tracking | 72.00 ms | 80.01 ms | 28.47k | 51.32k | 1.80× |
| FlashSAC g1_walk_flat | 234.27 ms | 249.18 ms | 8.74k | 16.47k | 1.88× |
| FlashSAC g1_motion_tracking | 245.13 ms | 260.43 ms | 8.36k | 15.76k | 1.89× |

| Workload | Single-node learner rows/s | Dual-node learner rows/s | Scaling |
| --- | ---: | ---: | ---: |
| SAC g1_walk_flat | 1,231,354 | 2,041,779 | 1.66× |
| SAC g1_motion_tracking | 910,271 | 1,638,196 | 1.80× |
| FlashSAC g1_walk_flat | 279,748 | 526,023 | 1.88× |
| FlashSAC g1_motion_tracking | 267,349 | 503,298 | 1.88× |

Both FlashSAC tasks record 1.88-1.89× environment throughput. SAC motion has a
longer single-node iteration than SAC walk, so the added synchronization cost
takes a smaller share of the iteration; its ratio is 1.80× versus 1.66×.

## 6. Compute and communication costs

If single-node iteration time is `T` and dual-node execution adds `C`, doubling
the global samples gives an approximate throughput ratio of `2T / (T + C)`:

- FlashSAC walk: `T=234.27 ms`, `C=14.91 ms`, ratio about 1.88×.
- SAC walk: `T=53.22 ms`, `C=10.97 ms`, ratio about 1.66×.
- SAC motion: `T=72.00 ms`, `C=8.01 ms`, ratio about 1.80×.

This model assumes comparable work on each rank. PPO also depends on reset
frequency, episode length, and task-manager work as the policy changes.
A fast network alone does not make `C/T` small.

## 7. Training metrics

| Algorithm/task | Single-node reward / episode length | Dual-node reward / episode length | Observation |
| --- | ---: | ---: | --- |
| PPO g1_walk_flat | 1.02 / 76.70 | 25.04 / 890.35 | Different policy stages; dual nodes collect twice the samples per iteration |
| PPO g1_flip_tracking | 22.09 / 190.72 | 19.22 / 141.28 | Lower dual-node metrics; no reported divergence |
| PPO go2_joystick_flat | 53.38 / 1000 | 54.01 / 1000 | Similar short-run metrics |
| PPO allegro_inhand | 3.10 / 133.85 | 11.74 / 399.63 | Different policy stages; dual nodes collect twice the samples per iteration |
| SAC g1_walk_flat | 16.91 / 67.67 | 17.22 / 70.02 | Similar short-run metrics |
| SAC g1_motion_tracking | -0.034 / 32.44 | -0.073 / 33.94 | Both runs remain in early learning |
| FlashSAC g1_walk_flat | 4.45 / 47.87 | 4.14 / 48.30 | Similar short-run metrics |
| FlashSAC g1_motion_tracking | 1.13 / 36.99 | 1.14 / 37.89 | Similar short-run metrics |

No instability was reported in these runs. The short-run metrics do not
establish convergence equivalence. That comparison requires multiple seeds,
longer runs, and matched sample budgets, particularly where single- and dual-node
PPO policies reach different stages.

## 8. Other experiments and toolchain issues

1. RoCE with P2P/SHM still disabled recorded about 259 ms/iteration in a short
   FlashSAC test, between the earlier TCP observation and the 248 ms observation
   with P2P/SHM enabled. This is a combined-configuration diagnostic.
2. A 30-iteration SAC fusion test showed no clear gain. The longer 300-iteration
   test, using iterations 50-299, recorded the approximately 1.8% difference in
   Section 4.2. Compilation and learning-start warmup affect short comparisons.
3. NCCL reported IB transport but `GDR 0`. Diagnostics found missing DMA-BUF
   symbols in `libmlx5`; `nvidia_peermem` was present but not loaded. The
   experiment did not change the driver or kernel to pursue GDR.
4. GB10 has compute capability 12.1, while the tested Triton `ptxas` recognized
   up to 12.0. Initial autotuning emitted `sm_121a` failures, followed by fallback
   or cache use and completed training.

## 9. Reproduction

These commands use the bootstrapped runtime on `HOST0` (`spark-0a2a`). Launch
once from HOST0; the native launcher starts rank1 on `nvidia@192.168.110.40`.
Both hosts need the patched runtime at the same path and passwordless SSH from
HOST0 to HOST1. The earlier path
`/home/nvidia/unilab-dual-spark-opt-20261008/UniLab` identifies the original
experiment installation, not the current reproduction directory.

Prepare the shell on HOST0:

```bash
cd /home/nvidia/unilab-dual-spark-repro/UniLab
export PATH="$HOME/.local/bin:$PATH"
```

Run one workload at a time with a free rendezvous port and a new run directory.
The launcher rejects nonempty directories. `--no-sync` preserves the editable
dependencies installed by bootstrap. `++training.log_interval=1` requests the
per-iteration metrics used for the 50-499 measurement window.

### FlashSAC: dual node

For direct `train_sac.py` or `train_flashsac.py` commands, see
[Dual-node SAC: direct Hydra entrypoints](../README.md#dual-node-sac-direct-hydra-entrypoints).
That method requires a command on each host. The launcher below starts the peer
and retains the original off-policy Rich logger.

```bash
run_name="flashsac_dual_g1_walk_500_native_$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 uv run --no-sync scripts/launch_distributed.py \
  --algo flashsac --task g1_walk_flat --sim mujoco \
  --num-nodes 2 --peer nvidia@192.168.110.40 \
  --master-ip 10.77.0.1 --ifname enp1s0f1np1 \
  --nccl-transport ib --nccl-ib-hca rocep1s0f1 \
  --remote-dir /home/nvidia/unilab-dual-spark-repro/UniLab \
  --dp-port 29702 --log-dir "logs/$run_name" \
  algo.num_envs=2048 algo.batch_size=8192 \
  algo.updates_per_step=8 algo.policy_frequency=4 \
  algo.max_iterations=500 ++training.log_interval=1
```

### PPO: dual node

PPO defaults to compact console output: launch parameters, the first and final
iterations, and progress at most every 30 seconds. At startup, the launcher
prints a `tail -n 40 -F ...` command for the full log. It prints final statistics
after all ranks complete. The rank logs retain the full RSL-RL output.

Use `--ppo-console full` for the full console stream or `--progress-interval 60`
for less frequent progress. SAC/FlashSAC retain their original Rich Live panel.

```bash
run_name="ppo_dual_go2_500_native_$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 uv run --no-sync scripts/launch_distributed.py \
  --algo ppo --task go2_joystick_flat --sim mujoco \
  --num-nodes 2 --peer nvidia@192.168.110.40 \
  --master-ip 10.77.0.1 --ifname enp1s0f1np1 \
  --nccl-transport ib --nccl-ib-hca rocep1s0f1 \
  --remote-dir /home/nvidia/unilab-dual-spark-repro/UniLab \
  --port 29705 --log-dir "logs/$run_name" \
  algo.num_envs=2048 algo.max_iterations=500 ++training.log_interval=1
```

For a functional smoke test, use `algo.max_iterations=2` and a fresh run name.
Do not use smoke-test throughput as a benchmark. Full rank logs are saved
automatically:

```bash
# Run in another terminal on HOST0, replacing the name with the printed run name.
tail -n 40 -F "logs/<run_name>/launcher/rank0.log"
tail -n 40 -F "logs/<run_name>/launcher/rank1.log"
```

Rank0 is the only TensorBoard/checkpoint writer. Final statistics require
successful exits from all ranks and a verified completed summary. For report
comparisons, use the fixed-window extractor and update its manifest with the
new run name. The launcher's full-training PPO average and off-policy
final-iteration rates use different windows from the tables in this report.

## 10. Validation and remaining work

Validation recorded for the experiment code:

- uni_rl CPU-safe focused suite: 57 passed, 20 skipped, 1 deselected.
- UniLab RoCE launcher tests: 4 passed.
- Ruff and `git diff --check`: passed.
- All 16 runs used separate log directories and checkpoints.
- The tested off-policy dual-node runs recorded ten collectives per iteration,
  with no reported NaN/Inf failure or rank divergence.

The matrix records FlashSAC weak scaling of 1.88-1.89× and SAC ratios of
1.66× for walk and 1.80× for motion. PPO ranges from 1.70× to 2.14×, with policy
and simulation-work differences affecting the comparison. Historical FlashSAC
iteration-time reductions also include runtime integration and workload changes.

Further SAC scaling would require fewer optimizer synchronization boundaries,
working GDR, or more learner work per synchronization. Changes to the update
schedule require separate stability and convergence tests; GDR requires a
validated driver/network configuration.
