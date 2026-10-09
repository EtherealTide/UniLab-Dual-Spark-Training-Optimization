# TorchEnv 集成与优化分析报告

## 1. 变更范围

* UniLab main 的 `TorchEnv`、tensor-only Manager、tensor device override、RSL-RL
  adapter 和新的 task/runtime tests 合入 dual-spark。
* unilab-rl main 的 tensor env contract、SAC/FlashSAC double-buffer、tensor replay
  ingress、inference ring、warmup/manifest 和 DP graph lifecycle 合入 dual-spark。
* dual-spark 特有能力保留：外部 TCP rendezvous、`UNILAB_DP_EXTERNAL`、rank/world
  topology、200G 网卡上的 NCCL 配置和每个非共址 rank 的完整 CPU affinity。

## 2. 冲突决策和原因

主线的 rank-local visibility API 是唯一设备来源；旧 `training.devices`、host-global
CUDA index 和 Numpy EnvProtocol 会造成 TorchEnv/多 GPU 拓扑双重真相，因此没有恢复。
外部 Spark rank 不应创建本机 `DpRankSupervisor`，而是直接构造
`DpParameterSync(rendezvous_url=...)`；PPO 的 launcher 也增加了 worker 检测，避免每个
节点的 local-rank 0 递归再起 torchrun。

## 3. 性能假设与验证顺序

TorchEnv 并不自动把 MuJoCo 物理搬到 GPU：当前 backend 仍是 CPU-authoritative，变化
主要在环境状态、Manager、H2D/D2H 边界和 learner IPC。先做 env-only phase timing，再
做单机 smoke、双机 TCP/NCCL smoke，最后才做 12 个 500 轮 E2E。任何绝对吞吐低于旧
值 95% 的项需复跑；稳定低于 90% 才进入 kernel/collective 优化，而不是先猜测。

## 4. 不应直接移植的旧优化

旧 dual-spark patch 针对 FastSAC/NumPy runtime；main 已将 FastSAC 重命名 SAC，并重做
replay、warmup、graph lifecycle 和 tensor learner。旧 patch 不能 cherry-pick。只有在
实测 profile 证明回归后，才按新 API 重写 persistent gradient bucket、collective fusion、
RoCE/P2P/SHM 策略或 whole-cycle CUDA Graph，并且每项单独 A/B。

## 5. 当前风险

UniLab main 与 dual-spark 的任务 owner、reward/curriculum 已变化，且旧矩阵的 Allegro
和 G1 flip owner 被删除；因此报告必须把 N/A 与性能失败区分。依赖也从旧
UniSim 1.7.4 / unilab-rl 1.4.4 升到 UniSim 1.7.12 / unilab-rl 1.4.10，历史吞吐只能
作为整合栈基线，不能宣称是单一 TorchEnv commit 的因果实验。

## 6. 首个 500 轮实测

Go2 PPO 单机 52,857.46、双机全局 72,317.12 steps/s，相比旧栈分别下降 20.88% 和
45.93%；扩展效率为 1.37×。质量仍对齐（末 20 reward/episode length：单机
53.14/998.13，双机 53.53/1000.00）。因此当前状态是“功能和质量通过、性能门禁未
通过”。下一优化优先级是 PPO collection hot path 和跨 rank iteration barrier，而不是
盲目移植旧 off-policy collective fusion。
