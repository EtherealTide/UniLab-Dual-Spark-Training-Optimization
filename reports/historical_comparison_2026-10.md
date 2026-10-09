# Historical Comparison and Evidence Scope

Date: 2026-10-09

This audit compares the [September Dual-Spark Throughput Scaling Report](https://github.com/Motphys/UniLab/blob/feat/dual-spark/docs/reports/dual_spark_2026-09.md) with the October optimized results in this repository. The October measurements describe the tested runtime and workload. They do not, by themselves, isolate the performance contribution of the dual-Spark optimization patch.

The October sources are [extracted metrics](../data/metrics_500_extracted.csv), [per-iteration metrics](../data/metrics_500_per_iteration.csv), [run manifest](../data/run_manifest.csv), and the [optimization report](dual_spark_optimization_2026-10.md). September values below are transcribed from the linked report; its historical resolved run configurations and raw event files are not included in this reproduction bundle.

## PPO: absolute throughput and scaling are different comparisons

Both reports label PPO throughput as global environment steps/s. October throughput is the arithmetic mean of iteration records 50–499. September specifies 500 iterations with a 50-iteration warmup excluded. This is the closest historical comparison available, although identical resolved configurations, runtime conditions, and repeated runs have not been established.

| Task | September single steps/s | October single steps/s | Single change | September dual global steps/s | October dual global steps/s | Dual change | September dual/single | October dual/single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `g1_walk_flat` | 43,558 | 49,160.65 | +12.86% | 81,620 | 83,567.30 | +2.39% | 1.874× | 1.700× |
| `g1_flip_tracking` | 23,244 | 22,861.84 | −1.64% | 41,354 | 43,038.46 | +4.07% | 1.779× | 1.883× |
| `go2_joystick_flat` | 68,659 | 66,808.68 | −2.69% | 132,516 | 133,751.39 | +0.93% | 1.930× | 2.002× |
| `allegro_inhand` | 46,956 | 42,643.60 | −9.18% | 92,671 | 91,439.72 | −1.33% | 1.974× | 2.144× |

Changes are `(October / September − 1) × 100%`; scaling ratios are calculated from the displayed single/dual rates rather than the historically rounded ratios.

The observed dual-node PPO changes range from −1.33% to +4.07%. These small deltas do not establish a statistically significant or reliably repeatable improvement: the supplied matrix has one run per configuration and no confidence intervals from independent repeated runs.

For Allegro, the larger dual/single ratio accompanies slightly lower absolute dual throughput and a 9.18% lower single-node denominator. It is therefore not evidence that the dual implementation became 8–9% faster. Conversely, G1 walk's smaller ratio accompanies higher absolute throughput on both single and dual nodes, with a much larger increase on the single-node side.

The October PPO runs also reach different policy stages. G1 walk's final mean episode lengths are approximately 77 on one node and 890 on two; Allegro's are approximately 134 and 400. Reset frequency and environment state distributions can change simulation cost. These are observations from actual training, not controlled fixed-policy hardware measurements. A greater-than-2× ratio must not be interpreted as hardware-only superlinear scaling.

## Off-policy: unit and workload differences

September reports global **learner samples/s**. October's headline tables report global **environment steps/s** and supply **learner rows/s** separately. Environment transitions and replay rows consumed by learner updates are different quantities, so those headline rates must not be compared directly.

The October formulas are:

```text
global environment steps/s = world_size × envs_per_rank × env_steps_per_iteration / iteration_time
global learner rows/s       = world_size × batch_size_per_rank × updates_per_iteration / iteration_time
```

For the reported October off-policy matrix, each rank uses 2,048 environments, batch size 8,192, eight critic updates per iteration, and policy frequency four. With one environment step per iteration, the global environment numerator is 2,048 for single-node runs and 4,096 for dual-node runs. Learner rows count critic-update replay rows; they do not count unique newly collected transitions or an additional actor-row term.

Known or unresolved historical differences include:

- September FlashSAC walk lists **4,096 environments per rank**, versus **2,048** in October. This changes collection work and the amount of new experience per iteration.
- The pinned UniLab base `d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3` task defaults use `updates_per_step=4` and `policy_frequency=2` for SAC motion and FlashSAC motion. October explicitly overrides these to **8/4**. FlashSAC walk defaults to policy frequency two, while October uses four. These base defaults identify potential workload changes; the September report does not record the resolved update schedules, so they do not prove the exact settings of the historical runs.
- Multiplying the September learner rates by the reported iteration times does not consistently recover a common eight-update, batch-8,192 workload. Historical resolved configurations and metric definitions are needed before using those rates to estimate patch gains.
- The October runtime includes `patches/unilab_rl/0001-integrate-tested-main-runtime.patch` before the dedicated optimization patch. Runtime integration and execution-mode changes affect single-node as well as distributed performance.

The historical iteration-time observations remain useful context:

| Workload | September single ms | October single ms | September dual ms | October dual ms |
| --- | ---: | ---: | ---: | ---: |
| SAC `g1_walk_flat` | 91.0 | 53.22 | 127.9 | 64.19 |
| SAC `g1_motion_tracking` | 64.5 | 72.00 | 92.0 | 80.01 |
| FlashSAC `g1_walk_flat` | 901.4 | 234.27 | 945.4 | 249.18 |
| FlashSAC `g1_motion_tracking` | 471.2 | 245.13 | 494.7 | 260.43 |

These show substantially shorter observed FlashSAC iterations in the October configuration, while SAC motion's single-node iteration is longer. The approximately 1.9–3.8× FlashSAC iteration-time reductions are **historical observations across runtime/configuration changes**, not isolated causal estimates of RoCE, persistent gradient buckets, or collective fusion. Neither similar iteration counts nor similar batch sizes alone guarantee identical work or learning progress.

## Weak scaling is not optimization speedup

Within the October matrix, every rank retains the single-node environment and learner batch sizes. Adding the second node therefore doubles the global samples processed per iteration. The dual/single throughput ratio measures **weak scaling of this runtime**.

It does not measure how much faster the optimization patch is than the previous implementation. It also does not establish that training reaches a specified reward in half the time: global batch size, collected experience, and policy evolution change when a rank is added.

To measure code optimization, compare the same node count and the same resolved workload before and after the change. To measure strong scaling, keep the global workload fixed and divide it across nodes. To measure learning efficiency, compare time and environment steps to a predefined evaluation target.

Launcher completion summaries also use different windows from the report tables: PPO may report a full-training average, while off-policy summaries may report final-iteration rates. Compare the extracted steady-state window when reproducing these tables; a final console statistic is not the same estimator.

## Reproducible evidence and limits

The bundled manifest and per-iteration data cover 16 completed optimized runs: eight single-node and eight dual-node runs, with 500 performance records each. These artifacts support the reported throughput of those configurations and the observation that these runs completed. They do not supply independent repetitions, historical raw baselines, or final convergence evidence.

The optimization report additionally describes transport and collective-fusion A/B diagnostics. Their raw logs and resolved configurations are not included in the 16-run manifest. Treat them as author-reported diagnostic observations until their artifacts are supplied:

- The transport comparison changes IB, P2P, and SHM settings together. It measures the combined configuration, not the isolated effect of RoCE.
- The reported DP sync comparison uses 43.36 ms and an explicitly labeled 11.43 ms median. The statistic for the first value is unspecified; verify matching estimators before describing their ratio as a quantified percentage reduction.
- The reported SAC fusion gain of approximately 1.8% needs independent repetitions and uncertainty estimates to establish a reliable effect.

Similar short-run reward metrics and the absence of observed NaN/Inf or rank divergence support training health in the tested runs. One seed and 500 iterations do not prove convergence equivalence or unchanged final policy quality. This distinction limits attribution; it does not invalidate the recorded measurements or imply that they were fabricated.

## Matched A/B plan

1. Preserve the historical results and publish the resolved configuration, source tree, package versions, and metric definitions for each historical run when available. Do not reconstruct missing settings from task defaults and present them as known historical facts.
2. Compare three explicitly identified runtimes: the pinned feature baseline; that baseline plus runtime integration; and integration plus the optimization patch. This separates runtime integration from the distributed optimization. Where an execution mode is unsupported by a baseline, document that limitation and use a common supported mode for isolated comparisons.
3. Use identical resolved environment counts, replay batch sizes, update counts, actor frequencies, model dimensions, AMP settings, simulation settings, thread allocation, and logging frequency. Record transport and graph-mode settings explicitly.
4. Run the same single/dual matrix with exclusive use of both machines, matching cache/warmup conditions and randomized or alternating A/B execution order. Repeat configurations across independent runs/seeds and report uncertainty across runs, rather than treating correlated iterations as independent repetitions.
5. Publish raw event files, resolved configs, rank logs, summaries, and manifests. Use the same iteration window and estimator for both sides; report absolute global environment steps/s, learner rows/s, iteration time, and dual/single scaling separately.
6. For transport and fusion attribution, change one factor at a time on the same integrated runtime. Publish graph capture/fallback status and synchronization timing with the same estimator for both sides.
7. Evaluate learning separately with longer, repeated runs, fixed global-sample budgets, and a predefined evaluation target. Report time-to-target and sample-to-target rather than equating reward at the same iteration with equivalent learning.

Until matched evidence is available, describe the October result as **measured weak scaling and end-to-end validation of an integrated optimized runtime**, with specific diagnostic optimization evidence and clearly qualified historical comparisons.
