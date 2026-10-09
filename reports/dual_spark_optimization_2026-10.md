# UniLab 双 DGX Spark 训练优化报告

日期：2026-10-08

> **Interpretation update (2026-10-09):** The tables report weak scaling within
> the optimized runtime. Historical off-policy iteration-time ratios include
> runtime and workload changes and are not isolated optimization gains. PPO
> dual-node absolute throughput changes by +2.39%, +4.07%, +0.93%, and -1.33%
> versus September. Read the [comparison audit](historical_comparison_2026-10.md)
> before drawing a code-performance or convergence conclusion. Original exported
> measurements are retained unchanged.

范围：PPO、SAC、FlashSAC；MuJoCo；G1 locomotion / motion tracking、Go2 joystick、
Allegro in-hand；2 × NVIDIA DGX Spark（每机单卡）。

版本基线：UniLab `feat/dual-spark@d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3`；
unilab-rl `feat/dual-spark@a3ed997d5c25ff708d674778782bc1be08a53e15`；UniSim
`v1.7.4@b48e91bbc62603299580a951c142a13c33bedae9`。实验优化代码由这些 pin
按复现仓库中的顺序 patch 构造。

## 1. 优化目标和验收口径

目标不是只优化 `g1_walk_flat`，而是在 2026-09 报告的全任务矩阵上验证优化
是否具有普适性。正式矩阵包括：

- PPO：`g1_walk_flat`、`g1_flip_tracking`、`go2_joystick_flat`、
  `allegro_inhand`；
- SAC：`g1_walk_flat`、`g1_motion_tracking`；
- FlashSAC：`g1_walk_flat`、`g1_motion_tracking`。

每个负载都新跑一次单机 500 轮和双机 500 轮，共 16 个正式 run。统计口径：

- 性能：iteration 50–499（450 个样本）的均值；
- 质量：末 20 条 TensorBoard 记录平均；
- PPO 和 off-policy 均报全局 env steps/s；
- off-policy 额外报 learner rows/s；
- 双机每 rank 保持单机相同规模，因此每 iteration 全局样本数翻倍。

PPO summary 中的 `completed_iterations=499` 使用 0-based 编号；16 个正式 run 的
TensorBoard `Perf/total_fps` 均核对为 500 条，不是少跑一轮。

off-policy 默认负载统一为每 rank `num_envs=2048`、`batch_size=8192`、
`updates_per_step=8`、`policy_frequency=4`，即每轮 8 次 critic 和 2 次 actor update。

## 2. 起点瓶颈

2026-09 版本已实现数学正确的外部 DP，但有三个性能瓶颈。

### 2.1 传输层仍使用 TCP 兼容配置

旧启动配置强制：

```bash
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
```

虽然两台 Spark 的 200 Gb/s RoCE HCA `rocep1s0f1` 处于 active 状态，NCCL 仍走 socket
路径。FlashSAC 旧配置每轮 10 次 collective 约需 43.36 ms。

### 2.2 FlashSAC whole-cycle CUDA Graph 与 DP 互斥

FlashSAC 的 NVIDIA 高速路径原先会直接拒绝 DP callback，导致双机时不能同时享受
whole-cycle CUDA Graph 的绝对性能。

### 2.3 频繁 pack/unpack 和过多 collective

每次 optimizer update 都要把所有参数梯度复制到 flat buffer，all-reduce 后再拷回。
SAC 默认每轮有 18 次同步；FlashSAC 有 12 次。对 50–90 ms 的轻量 learner，
这些固定成本足以主导扩展效率。

## 3. 具体优化及原因

### 3.1 显式 RoCE 传输配置

UniLab `scripts/launch_distributed.py` 新增：

- `--nccl-transport {tcp,auto,ib}`；
- `--nccl-ib-hca`；
- `ib` 模式设置 `NCCL_IB_DISABLE=0`、`NCCL_IB_HCA=rocep1s0f1`；
- 验证拓扑中显式设置 `NCCL_P2P_DISABLE=0`、`NCCL_SHM_DISABLE=0`；
- `tcp` 保留为兼容默认值。

`DpParameterSync.start()` 也改为只对单机 FileStore 路径默认禁用 P2P/SHM，
多节点 TCPStore 不再库内强制关闭。

**原因**：每节点只有一块 GPU，没有需要回避的本地多 GPU peer edge。保留兼容开关
只会让跨节点 collective 走慢路径。

### 3.2 FlashSAC 整周期 graph 内 NCCL

为 FlashSAC 实现 graph warmup/capture/replay lifecycle hooks，把 all-reduce 作为 CUDA Graph
节点捕获，并在 replay 前统计 collective。graph capture 失败时仍保留 eager fallback。

为了保持 NaN/Inf 安全性，同步额外携带 finite-loss sentinel。任意 rank 报告非有限
时，对应 optimizer 在所有 rank 同步 skip，避免参数分歧。

**原因**：单机 FlashSAC 的主要加速来自 whole-cycle graph。双机如果退回 eager，即使
网络足够快，也不可能接近单机两倍。

### 3.3 持久 flat gradient bucket view

CUDA Graph 要求 gradient storage 地址稳定。在 graph warmup 时，DP 层把每个参数的
`.grad` 绑定到 flat collective buffer 的不重叠 view。后续 backward 直接写入同一
buffer，replay 时就地 all-reduce，不再做逐参数 pack/unpack。

eager 兼容路径仍使用 copy。finite sentinel 明确标记为 copy-only，避免一个标量同时
别名到多个 bucket。

**原因**：旧报告将约 20–26 ms/iter 归因于 DP wrapper、梯度打包以及通信对
异步流水线的打断。直接复用 gradient storage 是消除这部分开销的关键。

### 3.4 合并数学独立的 collective

- SAC：critic + alpha 梯度合并，18 次降至 10 次/iter；
- FlashSAC：actor + temperature 梯度合并，12 次降至 10 次/iter；
- 每个 optimizer 保留独立 finite sentinel 和 step gate。

**原因**：这些参数集的 loss 和 optimizer 数学独立，合并传输不改变梯度值，
但可以减少高延迟的跨机 collective 边界。

## 4. 单项 A/B 证据

### 4.1 传输层

| FlashSAC 配置 | collective | DP sync | iteration | 全局 env steps/s |
| --- | ---: | ---: | ---: | ---: |
| TCP，IB/P2P/SHM 禁用 | 10 | 43.36 ms | ~279.15 ms | ~14.67k |
| RoCE，P2P/SHM 开启 | 10 | 11.43 ms 中位数 | 248.16 ms | 16.54k |

These author-reported diagnostic observations change IB/P2P/SHM together.
The old row does not specify its timing estimator, while 11.43 ms is a median;
the previously stated 73.6% reduction is therefore not a verified like-for-like
statistic. Raw A/B artifacts are not included in this bundle. Reported NCCL
diagnostics show `NET/IB`, `rocep1s0f1`, 200000 Mb/s and `GDR 0`.

### 4.2 SAC collective 融合

| 版本 | iter 50–299 | env steps/s | collective/iter |
| --- | ---: | ---: | ---: |
| 未融合 | 65.57 ms | 62.74k | 18 |
| critic + alpha 融合 | 64.37 ms | 63.85k | 10 |

The point observations differ by about 1.8%. This diagnostic A/B used a
250-iteration steady window, but the raw artifacts and repeat-run uncertainty
are not bundled; it does not establish a reproducible isolated gain yet.

### 4.3 绝对性能

| 负载 | 2026-09 单机 | 当前单机 | 2026-09 双机 | 当前双机 |
| --- | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 91.0 ms | 53.22 ms | 127.9 ms | 64.19 ms |
| SAC g1_motion_tracking | 64.5 ms | 72.00 ms | 92.0 ms | 80.01 ms |
| FlashSAC g1_walk_flat | 901.4 ms | 234.27 ms | 945.4 ms | 249.18 ms |
| FlashSAC g1_motion_tracking | 471.2 ms | 245.13 ms | 494.7 ms | 260.43 ms |

The historical FlashSAC iteration-time ratios are about 1.9-3.8x, across changed
runtime and workload configurations. FlashSAC walk changes from 4096 to 2048
environments per rank, and update schedules require historical resolved configs
for confirmation. These ratios cannot be attributed entirely to RoCE/DP.
See the comparison audit for the separate integration patch and metric units.

## 5. 500 轮全任务实测

### 5.1 PPO

| 任务 | 每节点 env | 单机 | 双机 | 加速 | 与旧报告 |
| --- | ---: | ---: | ---: | ---: | ---: |
| g1_walk_flat | 2048 | 49.16k | 83.57k | **1.70×** | 1.87× |
| g1_flip_tracking | 1024 | 22.86k | 43.04k | **1.88×** | 1.78× |
| go2_joystick_flat | 2048 | 66.81k | 133.75k | **2.00×** | 1.93× |
| allegro_inhand | 16384 | 42.64k | 91.44k | **2.14×** | 1.97× |

Go2 的单/双机尾部 episode length 都是 1000，环境状态分布高度一致，实测达到
2.00×。G1 walk 在 500 轮时双机已进入长 episode 阶段（尾部 890 vs 单机 77），
两侧真实仿真工作负载不再同构，因此全窗口为 1.70×，低于早期 200 轮窗口的
1.98×。Allegro 的双机 reward/episode length 为 11.74/399.63，单机为
3.10/133.85；双机 collection 反而从 2738.69 ms 降至 2402.34 ms，最终得到
2.14×。这是真实训练 benchmark 的状态反馈：Allegro 的超线性数值包含策略阶段和
环境状态分布贡献，不能解释为硬件本身超过 2×；同理也不应通过挑选早期窗口隐藏
G1 walk 的 1.70×。

### 5.2 off-policy

| 负载 | 单机 iter | 双机 iter | 单机 env steps/s | 双机 env steps/s | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 53.22 ms | 64.19 ms | 38.50k | 64.02k | **1.66×** |
| SAC g1_motion_tracking | 72.00 ms | 80.01 ms | 28.47k | 51.32k | **1.80×** |
| FlashSAC g1_walk_flat | 234.27 ms | 249.18 ms | 8.74k | 16.47k | **1.88×** |
| FlashSAC g1_motion_tracking | 245.13 ms | 260.43 ms | 8.36k | 15.76k | **1.89×** |

| 负载 | 单机 learner rows/s | 双机 learner rows/s | 加速 |
| --- | ---: | ---: | ---: |
| SAC g1_walk_flat | 1,231,354 | 2,041,779 | 1.66× |
| SAC g1_motion_tracking | 910,271 | 1,638,196 | 1.80× |
| FlashSAC g1_walk_flat | 279,748 | 526,023 | 1.88× |
| FlashSAC g1_motion_tracking | 267,349 | 503,298 | 1.88× |

两个 FlashSAC 任务都在 1.88–1.89×，证明近线性扩展不是 `g1_walk_flat`
特例。SAC motion 比 walk 更重，固定通信成本占比更小，因此扩展率从 1.66×
上升到 1.80×。

## 6. 为什么不是所有负载都达到 2×

设单机每轮时间为 `T`，双机的额外通信/同步成本为 `C`，每轮全局样本数翻倍，
则理想扩展约为 `2T / (T + C)`。

- FlashSAC walk：`T=234.27 ms`，`C=14.91 ms`，理论/​实测均约 1.88×；
- SAC walk：`T=53.22 ms`，`C=10.97 ms`，只有约 1.66×；
- SAC motion：`T=72.00 ms`，`C=8.01 ms`，上升到 1.80×；
- PPO 除了 DDP learner 成本，还受策略进化导致的 reset 频率、episode length 和任务
  manager 工作量变化影响。

因此“总吞吐接近两倍”的前提是 `C/T` 足够小，不是单纯把网卡换成 200 Gb/s。

## 7. 训练质量

| 算法/任务 | 单机 reward / ep len | 双机 reward / ep len | 解读 |
| --- | ---: | ---: | --- |
| PPO g1_walk_flat | 1.02 / 76.70 | 25.04 / 890.35 | 双机每轮全局样本翻倍，学习阶段显著超前 |
| PPO g1_flip_tracking | 22.09 / 190.72 | 19.22 / 141.28 | 双机略低，但无发散；500 轮不足以判定收敛等价 |
| PPO go2_joystick_flat | 53.38 / 1000 | 54.01 / 1000 | Similar short-run metrics |
| PPO allegro_inhand | 3.10 / 133.85 | 11.74 / 399.63 | 双机每轮样本翻倍，学习阶段显著超前 |
| SAC g1_walk_flat | 16.91 / 67.67 | 17.22 / 70.02 | Similar short-run metrics |
| SAC g1_motion_tracking | -0.034 / 32.44 | -0.073 / 33.94 | 同一早期学习区间 |
| FlashSAC g1_walk_flat | 4.45 / 47.87 | 4.14 / 48.30 | Similar short-run metrics |
| FlashSAC g1_motion_tracking | 1.13 / 36.99 | 1.14 / 37.89 | Similar short-run metrics |

These 500-iteration observations show no reported instability in the measured
runs. Similar short-run rewards do not establish learning or convergence
equivalence. Multiple seeds and matched sample budgets are needed, especially
when PPO single/dual policies enter different learning stages.

## 8. 失败实验和有限收益

1. **RoCE 但继续禁用 P2P/SHM**：FlashSAC 短测约 259 ms/iter，优于旧 TCP，
   但不如 RoCE + P2P/SHM 开启的 248 ms。
2. **SAC collective 融合的短测**：30 轮无法观察收益；300 轮固定窗口才确认
   +1.8%。这说明冷启编译和 learning starts 会误导过短 A/B。
3. **GDR 探测**：NCCL 已使用 IB，但仍报 `GDR 0`。`libmlx5` 缺少需要的 DMA-BUF
   符号，`nvidia_peermem` 存在但未加载。本轮没有为追求最后几个百分点而修改
   驱动/内核。
4. **GB10 PTXAS**：GB10 是 compute capability 12.1，当前 Triton 捆绑 `ptxas` 最高
   识别 12.0。首次 autotune 会输出 `sm_121a` 失败，随后回退/命中缓存并正常完成。

## 9. 复现

FlashSAC 双机：

```bash
.venv/bin/python scripts/launch_distributed.py \
  --algo flashsac --task g1_walk_flat --sim mujoco \
  --num-nodes 2 --peer <rank1-ssh-host> \
  --master-ip 10.77.0.1 --ifname enp1s0f1np1 \
  --nccl-transport ib --nccl-ib-hca rocep1s0f1 \
  --remote-dir /home/nvidia/unilab-dual-spark-opt-20261008/UniLab \
  --dp-port 29702 --log-dir logs/flashsac_dual_g1_walk_500 \
  algo.num_envs=2048 algo.batch_size=8192 \
  algo.updates_per_step=8 algo.policy_frequency=4 \
  algo.max_iterations=500
```

PPO 双机：

```bash
.venv/bin/python scripts/launch_distributed.py \
  --algo ppo --task go2_joystick_flat --sim mujoco \
  --num-nodes 2 --peer <rank1-ssh-host> \
  --master-ip 10.77.0.1 --ifname enp1s0f1np1 \
  --nccl-transport ib --nccl-ib-hca rocep1s0f1 \
  --remote-dir /home/nvidia/unilab-dual-spark-opt-20261008/UniLab \
  --port 29705 --log-dir logs/ppo_dual_go2_500 \
  algo.num_envs=2048 algo.max_iterations=500
```

每个 run 使用独立 port 和 log directory。rank 0 是唯一 TensorBoard/checkpoint 写入者。

## 10. 工程验证和结论

- uni_rl CPU-safe focused suite：57 passed, 20 skipped, 1 deselected；
- UniLab RoCE launcher：4 passed；
- Ruff、`git diff --check` 通过；
- 16 个正式 500 轮 run 使用独立日志目录和 checkpoint；
- off-policy 所有双机运行均为 10 collective/iteration，无 NaN/Inf 或 rank 分歧。

结论：

1. The optimized configuration records 1.88-1.89x FlashSAC weak scaling across
   walk/motion. Historical iteration-time reductions also include runtime and
   workload changes; this report does not isolate a 1.9-3.8x optimization gain.
2. SAC 的扩展率从 walk 的 1.66× 到 motion 的 1.80×，清晰验证了计算/通信比模型。
3. PPO 在状态分布匹配的 Go2 上达到 2.00×；G1 walk 的 500 轮真实训练受
   策略阶段差异影响，全窗口为 1.70×。
4. 若要让轻量 SAC 从 1.66× 继续接近 2×，需要减少 optimizer synchronization
   boundary、启用真正 GDR，或提高 learner 计算重量。前两者需要新的稳定性/收敛对照。
