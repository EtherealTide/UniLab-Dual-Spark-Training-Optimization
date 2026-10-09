# Dual-DGX-Spark Training: Validation Results

Experiment date: 2026-10-08. Interpretation updated: 2026-10-09.

This report records 500-iteration single- and dual-node runs of PPO, SAC, and
FlashSAC on two NVIDIA DGX Spark hosts. It covers the September task matrix,
with each rank retaining the single-node workload size.

The throughput ratios measure weak scaling of the October configuration.
Historical off-policy runtimes, units, and workloads differ; these ratios do
not measure the optimization patch's gain over September. Each configuration
has one run. The results describe observed throughput and short-run training
metrics, not convergence equivalence. See the
[historical comparison](historical_comparison_2026-10.md).

## 1. Run coverage and results

### 1.1 Measurement method

All eight workloads completed a single-node and a dual-node run:

- PPO: `g1_walk_flat`, `g1_flip_tracking`, `go2_joystick_flat`, and `allegro_inhand`.
- SAC: `g1_walk_flat` and `g1_motion_tracking`.
- FlashSAC: `g1_walk_flat` and `g1_motion_tracking`.

Each run contains 500 iterations. Throughput and timing means use iterations
50-499; iterations 0-49 are excluded as warmup. Reward and episode length use
the final 20 TensorBoard records. PPO's summary field
`completed_iterations=499` is a zero-based index; the event files contain
500 performance records.

### 1.2 PPO throughput

Rates are global environment steps/s.

| Task | Envs/rank | Single-node steps/s | Dual-node steps/s | Scaling |
| --- | ---: | ---: | ---: | ---: |
| g1_walk_flat | 2048 | 49,160.65 | 83,567.30 | 1.70× |
| g1_flip_tracking | 1024 | 22,861.84 | 43,038.46 | 1.88× |
| go2_joystick_flat | 2048 | 66,808.68 | 133,751.39 | 2.00× |
| allegro_inhand | 16384 | 42,643.60 | 91,439.72 | 2.14× |

### 1.3 SAC and FlashSAC throughput

| Workload | Single-node iteration | Dual-node iteration | Single-node env steps/s | Dual-node env steps/s | Scaling |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 53.22 ms | 64.19 ms | 38,496.57 | 64,022.10 | 1.66× |
| SAC g1_motion_tracking | 72.00 ms | 80.01 ms | 28,471.35 | 51,320.66 | 1.80× |
| FlashSAC g1_walk_flat | 234.27 ms | 249.18 ms | 8,742.86 | 16,474.44 | 1.88× |
| FlashSAC g1_motion_tracking | 245.13 ms | 260.43 ms | 8,355.22 | 15,759.63 | 1.89× |

Learner rows/s count replay rows consumed by critic updates. The workload uses
`batch_size=8192` and `updates_per_step=8` per rank.

| Workload | Single-node rows/s | Dual-node rows/s | Scaling |
| --- | ---: | ---: | ---: |
| SAC g1_walk_flat | 1,231,354 | 2,041,779 | 1.66× |
| SAC g1_motion_tracking | 910,271 | 1,638,196 | 1.80× |
| FlashSAC g1_walk_flat | 279,748 | 526,023 | 1.88× |
| FlashSAC g1_motion_tracking | 267,349 | 503,298 | 1.88× |

### 1.4 Reward and episode length

| Algorithm/task | Single-node reward / episode length | Dual-node reward / episode length |
| --- | ---: | ---: |
| PPO g1_walk_flat | 1.02 / 76.70 | 25.04 / 890.35 |
| PPO g1_flip_tracking | 22.09 / 190.72 | 19.22 / 141.28 |
| PPO go2_joystick_flat | 53.38 / 1000.00 | 54.01 / 1000.00 |
| PPO allegro_inhand | 3.10 / 133.85 | 11.74 / 399.63 |
| SAC g1_walk_flat | 16.91 / 67.67 | 17.22 / 70.02 |
| SAC g1_motion_tracking | -0.034 / 32.44 | -0.073 / 33.94 |
| FlashSAC g1_walk_flat | 4.45 / 47.87 | 4.14 / 48.30 |
| FlashSAC g1_motion_tracking | 1.13 / 36.99 | 1.14 / 37.89 |

No NaN/Inf failure, rank divergence, or throughput collapse was reported in the
500-iteration runs. SAC and FlashSAC have similar final metrics across the
single- and dual-node configurations. PPO processes twice the global samples
per dual-node iteration: G1 walk reaches a later policy stage, while G1 flip has
lower final metrics on two nodes. These observations require longer, repeated
runs with matched sample budgets before drawing convergence conclusions.

## 2. Test environment

| Item | Configuration |
| --- | --- |
| Hosts | 2 × NVIDIA DGX Spark; GB10, 1 GPU and 20 CPU cores per host |
| Interconnect | Direct QSFP link, negotiated at 200 Gb/s |
| Interface / HCA | `enp1s0f1np1` / `rocep1s0f1` |
| Transport | NCCL over RoCE; `NET/IB`, P2P/SHM enabled, `GDR 0` |
| PPO launch | `torchrun --nnodes=2 --nproc_per_node=1` |
| Off-policy launch | uni_rl external DP; TCPStore rendezvous and NCCL in the graph update cycle |
| Simulation backend | MuJoCo |
| Software | PyTorch 2.9.0+cu130, CUDA 13, GB10 compute capability 12.1 |
| UniLab experiment code | `perf/dual-spark-20261008` @ `139524ac` |
| unilab_rl experiment code | `perf/dual-spark-20261008` @ `385a69f` |

`algo.num_envs` and `algo.batch_size` are per-rank settings. A dual-node
iteration therefore processes twice the global environment steps and learner
rows of a single-node iteration.

## 3. PPO timing

| Task | Envs/rank | Single-node collection | Dual-node collection | Single-node learning | Dual-node learning | Throughput scaling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| g1_walk_flat | 2048 | 842.66 ms | 949.74 ms | 165.51 ms | 230.49 ms | 1.70× |
| g1_flip_tracking | 1024 | 887.23 ms | 949.23 ms | 191.37 ms | 196.61 ms | 1.88× |
| go2_joystick_flat | 2048 | 529.40 ms | 509.18 ms | 211.64 ms | 231.43 ms | 2.00× |
| allegro_inhand | 16384 | 2738.69 ms | 2402.34 ms | 359.62 ms | 471.21 ms | 2.14× |

Equal iteration counts do not imply equal sample budgets or policy states.
For Allegro, higher dual-node reward and longer episodes accompany a collection
time decrease from 2738.69 to 2402.34 ms. Its 2.14× throughput ratio includes
state-distribution changes as well as additional hardware. It does not measure
hardware-only superlinear scaling or bitwise equivalence with one-node training.

## 4. Off-policy execution

### 4.1 Architecture

Each host runs a collector, local replay buffer, and learner. Experience stays
on the collecting host. Gradients are averaged by all-reduce at optimizer
boundaries; rank0 broadcasts the initial weights.

The optimized graph update paths use NCCL collectives and persistent flat
gradient bucket views. SAC combines critic and alpha synchronization;
FlashSAC combines actor and temperature synchronization. The recorded workload
uses ten collectives per iteration for both algorithms.

### 4.2 Iteration time and final metrics

| Workload | Single-node iteration | Dual-node iteration | Throughput scaling | Dual-node final reward / episode length |
| --- | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 53.22 ms | 64.19 ms | 1.66× | 17.22 / 70.02 |
| SAC g1_motion_tracking | 72.00 ms | 80.01 ms | 1.80× | -0.073 / 33.94 |
| FlashSAC g1_walk_flat | 234.27 ms | 249.18 ms | 1.88× | 4.14 / 48.30 |
| FlashSAC g1_motion_tracking | 245.13 ms | 260.43 ms | 1.89× | 1.14 / 37.89 |

## 5. Timing breakdown and historical comparison

### 5.1 SAC g1_walk_flat

| Component | Single-node | Dual-node | Dual/single time ratio |
| --- | ---: | ---: | ---: |
| Iteration | 53.22 ms | 64.19 ms | 1.21 |
| Learner training | 49.89 ms | 62.65 ms | 1.26 |
| Collector environment step | 36.89 ms | 38.27 ms/rank | 1.04 |
| DP collectives | — | 10/iteration | — |

Dual-node execution adds 10.97 ms per iteration while doubling the global
environment steps, yielding 1.66× throughput. SAC motion takes 72.00 ms on one
node and 80.01 ms on two, yielding 1.80×.

### 5.2 FlashSAC

FlashSAC walk takes 234.27 ms on one node and 249.18 ms on two, an increase
of 6.4%. Motion takes 245.13 and 260.43 ms, an increase of 6.2%. The corresponding
environment-throughput ratios are 1.88-1.89× across the two tasks.

### 5.3 September and October iteration times

| Workload | September single-node | October single-node | September dual-node | October dual-node |
| --- | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 91.0 ms | 53.22 ms | 127.9 ms | 64.19 ms |
| SAC g1_motion_tracking | 64.5 ms | 72.00 ms | 92.0 ms | 80.01 ms |
| FlashSAC g1_walk_flat | 901.4 ms | 234.27 ms | 945.4 ms | 249.18 ms |
| FlashSAC g1_motion_tracking | 471.2 ms | 245.13 ms | 494.7 ms | 260.43 ms |

October FlashSAC iterations are about 1.9-3.8× shorter than the historical
observations. Runtime integration, execution mode, and workload settings also
changed, so this is not an isolated estimate of the DP optimization's gain.
SAC motion's single-node time increases from 64.5 to 72.00 ms, while its
dual-node time decreases from 92.0 to 80.01 ms. The reported scaling ratio
changes from 1.40× to 1.80× across those configurations. The
[historical comparison](historical_comparison_2026-10.md) records the known
differences and missing baseline information.

## 6. Source provenance and validation

| Repository | Experiment branch / commit | Changes |
| --- | --- | --- |
| UniLab | `perf/dual-spark-20261008` / code `139524ac` | RoCE launcher, English/Chinese documentation, reproduction commands |
| unilab_rl | `perf/dual-spark-20261008` / `385a69f` | Graph DP, persistent bucket views, collective fusion, consistent finite-loss gates |

The public feature-branch bases are
UniLab `feat/dual-spark@d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3` and
unilab-rl `feat/dual-spark@a3ed997d5c25ff708d674778782bc1be08a53e15`.
The experiment commits `139524ac` and `385a69f` identify the optimized source
trees rebuilt by the bundle patches; they are not the public feature-branch
heads.

Recorded checks:

- uni_rl CPU-safe focused suite: 57 passed, 20 skipped, 1 deselected.
- UniLab RoCE launcher tests: 4 passed.
- Ruff and `git diff --check`: passed.
- All 16 runs, eight single-node and eight dual-node, used separate log directories.

Implementation details, diagnostic A/B results, toolchain issues, and current
reproduction commands are in the
[optimization report](dual_spark_optimization_2026-10.md).
