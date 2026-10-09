# September and October Results: Historical Comparison

Date: 2026-10-09.

This report compares the
[September throughput report](https://github.com/Motphys/UniLab/blob/feat/dual-spark/docs/reports/dual_spark_2026-09.md)
with the October measurements. October data are available as
[extracted metrics](../data/metrics_500_extracted.csv),
[per-iteration records](../data/metrics_500_per_iteration.csv), and a
[run manifest](../data/run_manifest.csv). Implementation details are in the
[optimization report](dual_spark_optimization_2026-10.md).

September values below are transcribed from its report. Historical resolved
configurations and raw event files are not included in this bundle. The
available comparison describes changes between measured configurations;
it cannot separate all runtime, workload, and optimization effects.

## 1. PPO throughput

Both reports use global environment steps/s. October averages iterations
50-499. September specifies 500 iterations with the first 50 excluded.
This is the closest available historical comparison, although identical
resolved settings and runtime conditions have not been established.

| Task | September single steps/s | October single steps/s | Single change | September dual global steps/s | October dual global steps/s | Dual change | September dual/single | October dual/single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `g1_walk_flat` | 43,558 | 49,160.65 | +12.86% | 81,620 | 83,567.30 | +2.39% | 1.874× | 1.700× |
| `g1_flip_tracking` | 23,244 | 22,861.84 | −1.64% | 41,354 | 43,038.46 | +4.07% | 1.779× | 1.883× |
| `go2_joystick_flat` | 68,659 | 66,808.68 | −2.69% | 132,516 | 133,751.39 | +0.93% | 1.930× | 2.002× |
| `allegro_inhand` | 46,956 | 42,643.60 | −9.18% | 92,671 | 91,439.72 | −1.33% | 1.974× | 2.144× |

Changes use `(October / September − 1) × 100%`. Scaling ratios are calculated
from the displayed rates rather than copied from historically rounded ratios.

Dual-node changes range from −1.33% to +4.07%. With one run per configuration
and no uncertainty estimate across independent runs, these differences do not
establish a repeatable improvement.

Allegro's larger scaling ratio accompanies slightly lower dual-node throughput
and a 9.18% lower single-node rate. It does not show an 8-9% improvement in
dual-node execution. G1 walk has higher single- and dual-node throughput, but
the larger single-node increase lowers its scaling ratio.

The October policies also reach different stages. G1 walk's final episode length
is about 77 on one node and 890 on two; Allegro's is about 134 and 400.
Reset frequency and state distribution affect simulation cost. The measured
ratios include those training effects, which rules out interpreting a ratio
above 2× as a hardware-only result.

## 2. Off-policy units and workload settings

September reports global learner samples/s. October reports global environment
steps/s and lists learner rows/s separately. Environment transitions and replay
rows consumed by learner updates count different work.

October uses:

```text
global environment steps/s = world_size × envs_per_rank × env_steps_per_iteration / iteration_time
global learner rows/s       = world_size × batch_size_per_rank × updates_per_iteration / iteration_time
```

Each October rank uses 2,048 environments, batch size 8,192, eight critic updates
per iteration, and policy frequency four. With one environment step per
iteration, the global environment-step numerator is 2,048 on one node and
4,096 on two. Learner rows count replay rows for critic updates, excluding a
separate actor-row term; they do not count unique newly collected transitions.

The historical comparison has four unresolved differences:

- September FlashSAC walk lists 4,096 environments per rank; October uses
  2,048. Both collection work and new experience per iteration change.
- The pinned UniLab base `d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3` defaults
  SAC motion and FlashSAC motion to `updates_per_step=4` and
  `policy_frequency=2`. October explicitly uses 8/4. FlashSAC walk defaults to
  policy frequency two and uses four in October. These defaults identify
  possible changes; the missing September resolved configurations prevent
  confirmation of its actual update schedules.
- Multiplying September learner rates by the reported iteration times does
  not consistently recover eight updates at batch size 8,192. Resolved
  configurations and the original metric definitions are needed to explain
  those rates.
- October applies `patches/unilab_rl/0001-integrate-tested-main-runtime.patch`
  before the DP optimization patch. That integration and the execution-mode
  changes affect single-node as well as distributed performance.

The reported iteration times are:

| Workload | September single ms | October single ms | September dual ms | October dual ms |
| --- | ---: | ---: | ---: | ---: |
| SAC `g1_walk_flat` | 91.0 | 53.22 | 127.9 | 64.19 |
| SAC `g1_motion_tracking` | 64.5 | 72.00 | 92.0 | 80.01 |
| FlashSAC `g1_walk_flat` | 901.4 | 234.27 | 945.4 | 249.18 |
| FlashSAC `g1_motion_tracking` | 471.2 | 245.13 | 494.7 | 260.43 |

October FlashSAC iterations are about 1.9-3.8× shorter; SAC motion's single-node
iteration is longer. These observations span runtime and workload changes.
They do not isolate the effects of RoCE, persistent gradient buckets, or
collective fusion. Equal iteration counts and batch sizes alone do not establish
equal work or learning progress.

## 3. Scaling and performance comparisons

The October matrix keeps environment and learner batch sizes fixed per rank.
Adding a second rank doubles the global samples per iteration. Its dual/single
throughput ratio measures weak scaling of that runtime.

Three other comparisons require different controls:

- Patch performance: hold node count and resolved workload fixed before and
  after the code change.
- Strong scaling: hold the global workload fixed and divide it across nodes.
- Learning efficiency: compare elapsed time and environment steps needed to
  reach a predefined evaluation target.

The current matrix cannot determine whether dual-node training reaches a given
reward in half the time. Adding a rank changes global batch size, collected
experience, and policy evolution.

Completion summaries also differ from the table estimator. PPO may print a
full-training average; off-policy summaries may print final-iteration rates.
Use the extracted 50-499 window for comparisons with these tables.

## 4. Available evidence

The manifest and per-iteration data cover 16 completed October runs: eight
single-node and eight dual-node runs, with 500 performance records each. They
support the reported throughput and run completion for those configurations.
The bundle does not contain independent repetitions, historical raw baselines,
or final convergence measurements.

The optimization report also records transport and collective-fusion A/B
diagnostics. Their raw logs and resolved configurations are outside the 16-run
manifest and are not bundled. Three limits apply:

- The transport test changes IB, P2P, and SHM together, measuring the combined
  profile rather than RoCE alone.
- The DP sync values are 43.36 ms and an explicitly labeled 11.43 ms median.
  The first estimator is unspecified, so their ratio is not a verified
  percentage reduction.
- The approximately 1.8% SAC fusion difference needs independent repetitions
  and an uncertainty estimate before it can be called a repeatable gain.

Short-run rewards and the absence of reported NaN/Inf or rank divergence provide
training-health observations for these runs. One seed and 500 iterations do
not establish convergence equivalence or unchanged final policy quality.

## 5. Follow-up A/B measurements

1. Publish historical resolved configurations, source trees, package versions,
   and metric definitions where available. Leave missing settings unresolved
   rather than substituting task defaults.
2. Compare the pinned feature baseline, baseline plus runtime integration, and
   integration plus the optimization patch. Use a common supported execution
   mode when a baseline cannot run a newer mode.
3. Match environment counts, replay batch sizes, update schedules, model sizes,
   AMP, simulation settings, thread allocation, and logging frequency. Record
   transport and graph-mode settings.
4. Reserve both machines for the tests, match cache and warmup conditions, and
   alternate or randomize A/B order. Repeat configurations across independent
   runs/seeds and estimate uncertainty across runs, not correlated iterations.
5. Publish event files, resolved configs, rank logs, summaries, and manifests.
   Use the same window and estimator on both sides. Report global environment
   steps/s, learner rows/s, iteration time, and scaling separately.
6. Test transport and fusion one factor at a time on the integrated runtime.
   Record graph capture/fallback status and use matching synchronization-time
   estimators.
7. Evaluate learning with longer repeated runs, fixed global-sample budgets,
   and a predefined target. Report time-to-target and sample-to-target.

The October matrix currently establishes measured weak scaling and completed
runs of the integrated runtime. Isolated optimization gains remain subject to
the matched comparisons above.
