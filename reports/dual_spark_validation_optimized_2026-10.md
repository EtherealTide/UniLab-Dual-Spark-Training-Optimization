# UniLab 双 Spark 多节点训练验证报告（优化后）

日期：2026-10-08

> **Interpretation update (2026-10-09):** These are weak-scaling measurements of
> the optimized configuration, not isolated optimization speedups over September.
> Some historical off-policy units/configurations differ. One seed and 500
> iterations establish observed throughput and short-run health, not convergence
> equivalence. See the [historical comparison audit](historical_comparison_2026-10.md).

目标：以与 2026-09 报告一致的任务矩阵和每节点满配口径，对优化后的
PPO、SAC 和 FlashSAC 双 NVIDIA DGX Spark 训练进行 500 轮真实验证。

---

## 1. 核心结论

### 1.1 可行性：完整任务矩阵通过

- PPO：`g1_walk_flat`、`g1_flip_tracking`、`go2_joystick_flat`、
  `allegro_inhand` 全部按 500 轮完成单机/双机对照。
- SAC：`g1_walk_flat` 和 `g1_motion_tracking` 均完成 500 轮对照。
- FlashSAC：`g1_walk_flat` 和 `g1_motion_tracking` 均完成 500 轮对照。
- 统计口径：iteration 0–49 为预热，吞吐和分段时间取 50–499 平均；
  质量指标取末 20 条 TensorBoard 记录平均。
- PPO summary 中的 `completed_iterations=499` 是 0-based 编号；事件文件均有
  500 条性能记录，不代表少跑一轮。

### 1.2 PPO：采集分片，大规模任务接近线性扩展

| 任务 | 每节点 env | 单机 steps/s | 双机 steps/s | 加速 |
| --- | ---: | ---: | ---: | ---: |
| g1_walk_flat | 2048 | 49,160.65 | 83,567.30 | **1.70×** |
| g1_flip_tracking | 1024 | 22,861.84 | 43,038.46 | **1.88×** |
| go2_joystick_flat | 2048 | 66,808.68 | 133,751.39 | **2.00×** |
| allegro_inhand | 16384 | 42,643.60 | 91,439.72 | **2.14×** |

### 1.3 off-policy：FlashSAC 绝对性能大幅提升，扩展率由计算/通信比决定

| 负载 | 单机 iter | 双机 iter | 单机 env steps/s | 双机 env steps/s | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 53.22 ms | 64.19 ms | 38,496.57 | 64,022.10 | **1.66×** |
| SAC g1_motion_tracking | 72.00 ms | 80.01 ms | 28,471.35 | 51,320.66 | **1.80×** |
| FlashSAC g1_walk_flat | 234.27 ms | 249.18 ms | 8,742.86 | 16,474.44 | **1.88×** |
| FlashSAC g1_motion_tracking | 245.13 ms | 260.43 ms | 8,355.22 | 15,759.63 | **1.89×** |

按 `batch_size=8192`、`updates_per_step=8` 计算的 learner rows/s：

| 负载 | 单机 rows/s | 双机 rows/s | 加速 |
| --- | ---: | ---: | ---: |
| SAC g1_walk_flat | 1,231,354 | 2,041,779 | 1.66× |
| SAC g1_motion_tracking | 910,271 | 1,638,196 | 1.80× |
| FlashSAC g1_walk_flat | 279,748 | 526,023 | 1.88× |
| FlashSAC g1_motion_tracking | 267,349 | 503,298 | 1.88× |

### 1.4 训练健康性：通过；收敛等价性需长程验证

| 算法/任务 | 单机 reward / ep len | 双机 reward / ep len |
| --- | ---: | ---: |
| PPO g1_walk_flat | 1.02 / 76.70 | 25.04 / 890.35 |
| PPO g1_flip_tracking | 22.09 / 190.72 | 19.22 / 141.28 |
| PPO go2_joystick_flat | 53.38 / 1000.00 | 54.01 / 1000.00 |
| PPO allegro_inhand | 3.10 / 133.85 | 11.74 / 399.63 |
| SAC g1_walk_flat | 16.91 / 67.67 | 17.22 / 70.02 |
| SAC g1_motion_tracking | -0.034 / 32.44 | -0.073 / 33.94 |
| FlashSAC g1_walk_flat | 4.45 / 47.87 | 4.14 / 48.30 |
| FlashSAC g1_motion_tracking | 1.13 / 36.99 | 1.14 / 37.89 |

500 轮内全部负载均持续学习且无 NaN/Inf、rank 分歧或性能崩溃。SAC 和
FlashSAC 的末段指标相近；PPO 的双机每轮处理两倍全局样本，G1 walk 已明显进入
更晚学习阶段，G1 flip 仍存在阶段性差异。These observations do not establish
convergence equivalence; repeated, longer matched runs are required.

---

## 2. 测试环境

| 项 | 值 |
| --- | --- |
| 主机 | 2 × NVIDIA DGX Spark（GB10，每机 1 GPU、20 CPU cores） |
| 互联 | QSFP 直连，协商 200 Gb/s |
| 网卡/HCA | `enp1s0f1np1` / `rocep1s0f1` |
| 传输 | NCCL over RoCE，`NET/IB`，P2P/SHM 开启，`GDR 0` |
| PPO 启动 | `torchrun --nnodes=2 --nproc_per_node=1` |
| off-policy 启动 | uni_rl 外部 DP（TCPStore rendezvous + graph 内 NCCL） |
| 仿真后端 | MuJoCo |
| 软件 | PyTorch 2.9.0+cu130，CUDA 13，GB10 compute capability 12.1 |
| UniLab | `perf/dual-spark-20261008` @ `139524ac` |
| unilab_rl | `perf/dual-spark-20261008` @ `385a69f` |

口径：`algo.num_envs` 和 `algo.batch_size` 是每 rank 配置。双机因此每轮处理单机
两倍的全局 env steps 和 learner rows。

---

## 3. PPO 双机验证

| 任务 | 每节点规模 | 单机 collection | 双机 collection | 单机 learning | 双机 learning | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| g1_walk_flat | 2048 | 842.66 ms | 949.74 ms | 165.51 ms | 230.49 ms | **1.70×** |
| g1_flip_tracking | 1024 | 887.23 ms | 949.23 ms | 191.37 ms | 196.61 ms | **1.88×** |
| go2_joystick_flat | 2048 | 529.40 ms | 509.18 ms | 211.64 ms | 231.43 ms | **2.00×** |
| allegro_inhand | 16384 | 2738.69 ms | 2402.34 ms | 359.62 ms | 471.21 ms | **2.14×** |

质量解读使用末 20 条记录。双机每 iteration 看到的全局样本数是单机的两倍，
因此不要把“相同 iteration 下学得更快”误解为与单机逐位等价；评判标准是无退化、
无发散、无 rank 分歧。Allegro 的 2.14× 是真实训练窗口的系统吞吐，但不是纯硬件
超线性扩展：双机在 500 轮内学到更高 reward、更长 episode，使 collection 时间由
2738.69 ms 降至 2402.34 ms；状态分布变化与双倍硬件同时贡献了该结果。

---

## 4. off-policy 双机验证

### 4.1 架构

每台 Spark 运行一套 collector + local replay + learner。经验数据不过网络；每次
optimizer boundary 对梯度做 all-reduce 并取平均。rank 0 广播初始权重，每次同步
后两 rank 权重保持一致。

优化后 FlashSAC 和 SAC 均把 NCCL collective 放入 whole-cycle CUDA Graph，并使用
持久 flat gradient bucket view。SAC 的 critic + alpha、FlashSAC 的 actor + temperature
合并同步，默认负载下均为 10 collective/iteration。

### 4.2 结果

| 负载 | 单机 | 双机 | 双/单 | 双机末 20 reward / ep len |
| --- | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 53.22 ms | 64.19 ms | **1.66×** | 17.22 / 70.02 |
| SAC g1_motion_tracking | 72.00 ms | 80.01 ms | **1.80×** | -0.073 / 33.94 |
| FlashSAC g1_walk_flat | 234.27 ms | 249.18 ms | **1.88×** | 4.14 / 48.30 |
| FlashSAC g1_motion_tracking | 245.13 ms | 260.43 ms | **1.89×** | 1.14 / 37.89 |

---

## 5. 性能剖析

### 5.1 SAC g1_walk_flat 分段时间

| 维度 | 单机 | 双机 | 双/单 |
| --- | ---: | ---: | ---: |
| 总 iteration | 53.22 ms | 64.19 ms | 1.21 |
| learner 训练 | 49.89 ms | 62.65 ms | 1.26 |
| collector env step | 36.89 ms | 38.27 ms/rank | 1.04 |
| DP collective | — | 10/iter | — |

双机每轮比单机多 10.97 ms，但每轮全局 env steps 翻倍，所以吞吐为
1.66×。SAC motion 的单机 iteration 更重（72.00 ms），双机只增至 80.01 ms，
因而达到 1.80×。

### 5.2 FlashSAC 时间线

FlashSAC walk 的单机/​双机 iteration 为 234.27/249.18 ms，双机额外开销只有
6.4%。motion 为 245.13/260.43 ms，额外开销 6.2%。两项任务均在 1.88–1.89×，
证明近线性扩展不是 `g1_walk_flat` 的特例。

### 5.3 与 2026-09 报告的绝对性能对比

| 负载 | 旧单机 iter | 新单机 iter | 旧双机 iter | 新双机 iter |
| --- | ---: | ---: | ---: | ---: |
| SAC g1_walk_flat | 91.0 ms | 53.22 ms | 127.9 ms | 64.19 ms |
| SAC g1_motion_tracking | 64.5 ms | 72.00 ms | 92.0 ms | 80.01 ms |
| FlashSAC g1_walk_flat | 901.4 ms | 234.27 ms | 945.4 ms | 249.18 ms |
| FlashSAC g1_motion_tracking | 471.2 ms | 245.13 ms | 494.7 ms | 260.43 ms |

FlashSAC 的绝对 iteration 时间下降约 1.9–3.8×。由于 learner 本身更快，固定
通信成本的相对比例反而上升，所以扩展比例仍约 1.88–1.89×；但总训练时间远低于
旧版。SAC motion 单机由 64.5 ms 增至 72.00 ms，不能归因于本轮分布式优化；其
双机时间仍由 92.0 ms 降至 80.01 ms，扩展率由 1.40× 提高到 1.80×。

---

## 6. 工程交付

| 仓库 | 分支/提交 | 内容 |
| --- | --- | --- |
| UniLab | `perf/dual-spark-20261008` / code `139524ac` | RoCE launcher；英中文文档；复现命令 |
| unilab_rl | `perf/dual-spark-20261008` / `385a69f` | graph 内 DP；持久 bucket view；collective 融合；finite gate 一致性 |

公开分支基线固定为 UniLab `feat/dual-spark@d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3`
和 unilab-rl `feat/dual-spark@a3ed997d5c25ff708d674778782bc1be08a53e15`；表中的
`139524ac` / `385a69f` 是应用本项目 patch 后的实验优化 tree 对应提交，不是远程
`feat/dual-spark` HEAD。

验证：

- uni_rl CPU-safe focused suite：57 passed, 20 skipped, 1 deselected；
- UniLab RoCE launcher：4 passed；
- Ruff 和 `git diff --check` 通过；
- 本报告 16 个正式 run（8 个单机 + 8 个双机）均使用独立日志目录。

完整优化原因、A/B 过程、失败实验和复现命令见
[`dual_spark_optimization_2026-10.md`](dual_spark_optimization_2026-10.md)。
