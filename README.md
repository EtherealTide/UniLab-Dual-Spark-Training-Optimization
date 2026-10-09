# UniLab Dual-Spark Training Optimization

两台 NVIDIA DGX Spark（每机单卡、200 Gb/s QSFP 直连）上的 UniLab 多节点训练优化、
TorchEnv 主线能力移植到指定 dual-spark pin、CPU/GPU carrier 选择、500 轮实验矩阵与可复现交付。
本修正版分支恢复并迁移旧矩阵中的 `g1_flip_tracking` 与 `allegro_inhand`，覆盖 PPO、SAC、
FlashSAC 的 MuJoCo 任务。

> `feat/torchenv-dual-spark` 是此前把 main 整体合入的历史 checkpoint；本分支才是当前
> 以 `d2fef27e`/`a3ed997d` 为 dual-spark 基线、再选择性加入 TorchEnv 与任务迁移的交付线。

## 结果先看：绝对吞吐

以下均为真实训练，iteration 0–49 作为预热，报告 iteration 50–499 的均值。双机列是
两台机器合计的全局吞吐，而不是单 rank 吞吐。

### PPO：全局环境步数/秒（历史 NumPy baseline）

| 任务 | 每节点 env | 单机 steps/s | 双机总 steps/s | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| `g1_walk_flat` | 2,048 | 49,160.65 | **83,567.30** | 1.70× |
| `go2_joystick_flat` | 2,048 | 66,808.68 | **133,751.39** | 2.00× |
本表仅保留旧版 NumPy baseline；TorchEnv CPU/GPU 的新鲜 500 轮结果写入
`data/torchenv_selectable_500.csv`，没有完成的项明确标记为 `pending`，不会用 smoke 数字填充。

### Off-policy：环境吞吐与 learner 吞吐

每 rank 使用 2,048 env、batch 8,192、每轮 8 次 critic 和 2 次 actor update。

| 负载 | 单机 env steps/s | 双机总 env steps/s | 单机 learner rows/s | 双机总 learner rows/s |
| --- | ---: | ---: | ---: | ---: |
| SAC `g1_walk_flat` | 38,496.57 | **64,022.10** | 1,231,354 | **2,041,779** |
| SAC `g1_motion_tracking` | 28,471.35 | **51,320.66** | 910,271 | **1,638,196** |
| FlashSAC `g1_walk_flat` | 8,742.86 | **16,474.44** | 279,748 | **526,023** |
| FlashSAC `g1_motion_tracking` | 8,355.22 | **15,759.63** | 267,349 | **503,298** |

完整分段时间、质量指标、旧版对照和结果解释见：

- [`reports/dual_spark_validation_optimized_2026-10.md`](reports/dual_spark_validation_optimized_2026-10.md)
- [`reports/dual_spark_optimization_2026-10.md`](reports/dual_spark_optimization_2026-10.md)
- [`data/throughput_500.csv`](data/throughput_500.csv)
- [`data/quality_500.csv`](data/quality_500.csv)
- [`data/metrics_500_extracted.csv`](data/metrics_500_extracted.csv)（提取器原始精度汇总）
- [`data/metrics_500_per_iteration.csv`](data/metrics_500_per_iteration.csv)（8,000 行逐轮数据）

## 优化内容

### 本轮 TorchEnv 迁移

- 基线严格从 UniLab `d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3` 和 unilab-rl
  `a3ed997d5c25ff708d674778782bc1be08a53e15` 开始；不把 main 的任务删除/新增当作基线变化。
- `training.env_device=null|cpu|cuda|cuda:0` 路由 Manager tensors、obs、reward、reset RNG；
  `cuda:1` 被拒绝，避免多节点 rank-local ordinal 歧义。
- Allegro action/observation/reward/termination/reset terms 改为 Torch-native；reset 仍通过
  UniSim 公共 transaction 边界提交，MuJoCo 物理保持 CPU-authoritative。
- G1 motion/flip 的 MuJoCo owner 改用 `TensorMotionCommandCfg`，motion sampler 支持
  `start/clip_start/uniform/adaptive/mixed`。
- UniSim 依赖在 `pyproject.toml` 与 `uv.lock` 中固定为 `==1.7.12`。

- NCCL TCP 改为 200 Gb/s RoCE，保留 P2P/SHM；
- FlashSAC whole-cycle CUDA Graph 内捕获 NCCL collective；
- 持久 flat gradient bucket view，消除重复 pack/unpack；
- SAC critic + alpha、FlashSAC actor + temperature 合并同步；
- 默认负载 collective 分别从 18/12 次降到 10 次/iteration；
- 用跨 rank finite-loss sentinel 保证 NaN/Inf 时 optimizer gate 一致。

## 精确版本锁定

版本的机器可读定义在 [`versions.env`](versions.env)。TorchEnv 实验使用：

| 项目 | 可复现版本 |
| --- | --- |
| UniLab | dual pin `d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3` + TorchEnv main source `5c8877ec6ec1bb0aba5c7682d6efdaed43fd49ee` 合并；集成提交 `d71875c0dc2bf8a054514b69547502664eb7b2e0`；包版本 `1.3.3` |
| UniSim | tag `v1.7.12`，commit `d082150631f16c8d3c8913281f62474cd9dffb93`，包版本 `unisim-core==1.7.12` |
| unilab-rl | dual pin `a3ed997d5c25ff708d674778782bc1be08a53e15` + TorchEnv main source `8c5a322bac66d4e8150867f24913da3ab916dea3` 合并；集成提交 `a7f82c9476eaa61e97bf9ec6dd725f9a055ceb40`；包版本 `1.4.10` |
| PyTorch | `2.9.0+cu130` |
| MuJoCo | `3.11.0` |

这里明确区分“公开 `feat/dual-spark` 分支 pin”和“实际实验 tree”。UniLab patch 还包含
实验需要的 off-policy logging 修复；unilab-rl 的第一份 patch 把公开分支集成到测试时
使用的 main runtime tree，第二份才是本轮 DP/CUDA Graph 优化。每一步来源和最终 tree
hash 都记录在 [`patches/README.md`](patches/README.md)，不会把本地实验提交误写成远端
分支 HEAD。

## 仓库结构

```text
.
├── README.md
├── SHA256SUMS                    # 补丁、报告与原始导出数据校验
├── versions.env                 # 唯一版本锁定入口
├── config/cluster.env.example   # 双机地址、网卡和 HCA 示例
├── data/                        # 500 轮汇总及逐 iteration 指标
├── patches/                     # UniLab / unilab-rl 可应用补丁
├── reports/                     # 优化后验证报告与详细优化报告
└── scripts/
    ├── bootstrap.sh             # 克隆、打补丁、安装和版本校验
    ├── run_one.sh               # 单项单机/双机 500 轮复现
    ├── run_matrix.sh            # 完整 16-run 矩阵
    └── extract_metrics.py       # TensorBoard 指标提取
```

## 环境前提

两台机器均需要：

- NVIDIA DGX Spark / GB10，Ubuntu aarch64；
- Python 3.12、`uv`、Git；
- PyTorch 2.9.0+cu130；
- 200 Gb/s 直连接口，例如 `enp1s0f1np1`；
- active RoCE HCA，例如 `rocep1s0f1`；
- 相同代码路径和资产缓存；
- 协调机可以无交互 SSH 到两台 Spark。

先验证链路：

```bash
ip -br link show enp1s0f1np1
ibv_devinfo -d rocep1s0f1
iperf3 -c <peer-200g-ip> -P 4
```

## 1. 在两台 Spark 上安装精确版本

在每台 Spark 上执行：

```bash
git clone https://github.com/EtherealTide/UniLab-Dual-Spark-Training-Optimization.git
cd UniLab-Dual-Spark-Training-Optimization
bash scripts/bootstrap.sh /home/nvidia/unilab-dual-spark-repro
```

脚本会：

1. 克隆三个上游仓库；
2. checkout `versions.env` 中的 base commit；
3. 对 UniLab 和 unilab-rl 应用 TorchEnv 集成 patch；
4. checkout UniSim 1.7.12 对应 commit；
5. 用 UniLab 的 frozen lock 创建 `.venv`；
6. 用精确源码覆盖安装 UniSim 和 unilab-rl；
7. 验证 tree hash 和包版本。

脚本不会覆盖已有脏工作树；目录状态不符合预期时会直接失败。

## 2. 配置集群

在协调机复制配置：

```bash
cp config/cluster.env.example config/cluster.env
```

修改至少以下字段：

```bash
HOST0=nvidia@192.168.110.48
HOST1=nvidia@192.168.110.40
MASTER_ADDR=10.77.0.1
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_HCA=rocep1s0f1
REMOTE_ROOT=/home/nvidia/unilab-dual-spark-repro
```

## 3. 复现单项实验

参数顺序为 `mode algo task envs run_name port [cpu|cuda]`；不填 carrier 时使用
`config/cluster.env` 的 `ENV_DEVICE`（默认 `cuda`）：

```bash
# 单机 PPO，500 轮
bash scripts/run_one.sh single ppo go2_joystick_flat 2048 \
  ppo_single_go2_500 29705

# 双机 PPO，500 轮
bash scripts/run_one.sh dual ppo go2_joystick_flat 2048 \
  ppo_dual_go2_500 29705

# 双机 SAC，固定 batch/update 口径
bash scripts/run_one.sh dual sac g1_walk_flat 2048 \
  sac_dual_g1_walk_500 29700

# 双机 FlashSAC
bash scripts/run_one.sh dual flashsac g1_motion_tracking 2048 \
  flashsac_dual_g1_motion_500 29703 cuda
```

完整矩阵：

```bash
bash scripts/run_matrix.sh
```

该脚本按顺序运行 CPU/GPU 两种 carrier 下的完整任务矩阵，每个 carrier 为 16 个 run
（8 个单机 + 8 个双机），耗时较长。每个 run 使用独立日志目录；
rank 0 是唯一 TensorBoard/checkpoint 写入者。

## 4. 提取绝对吞吐

```bash
source config/cluster.env
$REMOTE_ROOT/UniLab/.venv/bin/python scripts/extract_metrics.py \
  --root /home/nvidia/unilab-dual-spark-repro/UniLab/logs \
  --manifest data/run_manifest.csv \
  --output metrics.csv
```

若日志分布在两台机器，分别在对应机器运行提取器后合并 CSV。提取器固定使用：

- 性能：iteration 50–499；
- 质量：末 20 条记录；
- PPO：`Perf/total_fps`；
- off-policy 环境吞吐：`Perf/total_fps`；
- off-policy learner 吞吐：`world_size × batch_size × updates_per_step / iteration_time`。

PPO summary 中 `completed_iterations=499` 是 0-based 编号；事件文件应有 500 条
`Perf/total_fps`，提取器会校验样本数。

每个 run 的 `run_config.json` 同时记录 `training.env_device`，因此 CPU/GPU 结果不会因
日志目录命名或 shell 默认值混淆。

## 结果边界

- 500 轮足以验证吞吐和训练健康性，不等价于 5000–10000 轮、多 seed 的最终收敛结论；
- Allegro 与 G1 walk 的单/双机在 500 轮内进入不同策略阶段，吞吐包含状态分布反馈；
- NCCL 使用 `NET/IB`，但实验仍为 `GDR 0`；没有修改驱动或内核模块；
- GB10 compute capability 12.1 会触发当前 Triton/PTXAS 的 `sm_121a` autotune 回退日志，
  缓存命中后训练可正常完成。

## 补丁应用

如不使用 `bootstrap.sh`，可手工执行：

```bash
git clone https://github.com/Motphys/UniLab.git
git -C UniLab checkout d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3
git -C UniLab apply ../patches/UniLab/*.patch

git clone https://github.com/unilabsim/unilab_rl.git
git -C unilab_rl checkout a3ed997d5c25ff708d674778782bc1be08a53e15
git -C unilab_rl apply ../patches/unilab_rl/*.patch
```

详见 [`patches/README.md`](patches/README.md)。
