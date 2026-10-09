# Selectable TorchEnv dual-Spark 迁移与性能报告

日期：2026-10-09  
分支：`feat/dual-spark-torchenv-selectable`

## 1. 实验边界

本轮不是把 UniLab main 整体升级，而是从两个公开 dual-spark pin 开始：

- UniLab `d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3`
- unilab-rl `a3ed997d5c25ff708d674778782bc1be08a53e15`
- UniSim `v1.7.12` / `d082150631f16c8d3c8913281f62474cd9dffb93`

在此基础上移植 main 的 tensor-native Manager/TorchEnv 所需代码，并保留 dual-spark 的
外部 TCP rendezvous、RoCE/NCCL 和非共址 CPU 拓扑。实验中的物理后端为 MuJoCo；它仍是
CPU-authoritative，`training.env_device=cuda` 表示 Manager state、观测、reward、reset
随机数和 action carrier 在 GPU，不表示 MuJoCo 物理已经搬到 GPU。

## 2. 实现改动

1. `training.env_device` 统一接受 `null`、`cpu`、`cuda`、`cuda:0`，拒绝 `cuda:1` 等
   host-global ordinal。多节点时每个 rank 使用自己的 local CUDA namespace。
2. 恢复的 Allegro owner 全部改为 `torch.Tensor`：增量 action、ball state observation、
   drop termination、rotation/pose/torque/work reward、reset RNG。MuJoCo host bridge reset
   仍通过公开 reset transaction，避免绕过 UniSim ABI。
3. MuJoCo G1 motion/flip、SAC、FlashSAC owner 使用 `TensorMotionCommandCfg`；sampler 补齐
   `start`、`clip_start`、`uniform`、`adaptive`、`mixed` 五种模式。
4. UniSim 依赖在 UniLab `pyproject.toml`/`uv.lock` 中从范围约束改为 `unisim-core==1.7.12`。
5. Allegro include XML 中的 mesh/texture 改为显式 `assets/...` 相对路径，避免
   UniSim 1.7.12 在 `MjSpec.to_xml()` 重序列化时丢失 include-local `meshdir`语义。
   因此干净部署不再需要人工 symlink。
6. `scripts/run_one.sh` 增加第七个参数 `cpu|cuda`；`run_matrix.sh` 对 PPO 四任务、SAC/
   FlashSAC 两任务执行 CPU/GPU × 单/双机矩阵。

## 3. 已完成门禁（不是 500 轮结果）

| 负载 | carrier | 规模 | 结果 |
| --- | --- | ---: | --- |
| Allegro PPO | CPU | 32 env × 2 iter | 通过；完成 reset、obs、action、PPO update |
| Allegro PPO | CUDA | 32 env × 2 iter | 通过；完成 reset、obs、action、PPO update |
| G1 flip PPO | CUDA | 32 env × 2 iter | 通过；motion command、actor/critic、PPO update |
| G1 walk SAC | CPU | 64 env × 2 iter | 启动并进入 off-policy runtime；首次 compile 预热超过短命令窗口，需在正式 run 中完整收集 |

Allegro CPU/GPU smoke 的第 2 轮分别约 572/339 steps/s；这两个数字仅用于功能门禁，
不用于和 500 轮 NumPy baseline 比较。G1 flip GPU smoke 首轮包含 motion JIT/warm-up，
后续轮约 521 steps/s，同样不是正式吞吐结论。

## 4. 500 轮矩阵状态

完整目标矩阵及字段在 `data/torchenv_selectable_500.csv`。当前工作区中尚未完成的行保留
`status=pending`，没有用 smoke 数字填充。正式统计口径固定为：iteration 50–499 平均
`Perf/total_fps`，off-policy 另报 learner rows/s；质量取最后 20 个 iteration 的 reward
与 episode length。这样即使单/双机出现不同 warm-up，也不会把启动开销混进主结果。

## 5. 复现

```bash
cp config/cluster.env.example config/cluster.env
# 编辑 HOST0/HOST1、200G MASTER_ADDR、NCCL_SOCKET_IFNAME、NCCL_IB_HCA、REMOTE_ROOT
bash scripts/bootstrap.sh /home/nvidia/unilab-dual-spark-repro

# 单项 A/B
bash scripts/run_one.sh single ppo allegro_inhand 16384 allegro_cpu_500 0 cpu
bash scripts/run_one.sh dual ppo allegro_inhand 16384 allegro_cuda_500 29720 cuda

# 完整 CPU/GPU 矩阵
bash scripts/run_matrix.sh
```

`bootstrap.sh` 会校验源码 tree、UniSim/unilab-rl/UniLab 包版本及 Torch/MuJoCo 版本；
`run_config.json` 保存 `training.env_device`，`scripts/extract_metrics.py` 固定 50–499
窗口提取绝对吞吐。

## 6. 当前限制

本报告已经证明两项被 main 删除的旧任务可以在 dual pin 上恢复并满足 TorchEnv contract，
但在 500 轮全矩阵完成前，不应宣称吞吐提升或无退化。下一步只需按 README 启动矩阵、将
生成的 TensorBoard 汇总写入 `data/torchenv_selectable_500.csv`，再更新验证报告中的
绝对吞吐、扩展效率和质量列。
