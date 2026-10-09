# TorchEnv + dual-Spark 集成实验报告

日期：2026-10-09  
分支：`feat/torchenv-dual-spark`  
UniLab 集成提交：`d71875c0dc2bf8a054514b69547502664eb7b2e0`  
unilab-rl 集成提交：`a7f82c9476eaa61e97bf9ec6dd725f9a055ceb40`

## 结论

本次工作把 UniLab main 的 TorchEnv/tensor-native manager、tensor replay/inference
runtime 合入 dual-spark 的外部 TCP data-parallel 拓扑。合并时保留了 rank-local
`CUDA_VISIBLE_DEVICES`（每进程 `cuda:0`）语义，并恢复了 `UNILAB_DP_EXTERNAL`、TCP
rendezvous、非共址 rank 的完整主机 CPU 预算。旧的 `training.devices` API 没有恢复。

旧报告中的 8 项负载不能全部直接复现：main 已删除 `g1_flip_tracking` 和
`allegro_inhand` owner，因此本报告的严格矩阵是 6 项×单/双机=12 runs：PPO Go2/G1
walk，SAC G1 walk/motion，FlashSAC G1 walk/motion。吞吐主指标是 `Perf/total_fps`
的绝对全局 steps/s，off-policy 另外记录 learner rows/s。

## 公平口径

Python 3.12、Torch 2.9.0+cu130、MuJoCo 3.11.0、UniSim 1.7.12；PPO 每机 2048 env，
off-policy 每 rank 2048 env、batch 8192、8 critic + 2 actor update、seed 1，500
iterations，去掉前 50 个 iteration 后取均值。双机数值为两台机器合计，而非单 rank。

## 实验表

| 算法/任务 | TorchEnv 单机 steps/s | TorchEnv 双机总 steps/s | 双/单 | 旧 NumPy 单机 | 旧 NumPy 双机 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| PPO `g1_walk_flat` | 待跑 | 待跑 | 待算 | 49,160.65 | 83,567.30 | smoke/500轮 |
| PPO `go2_joystick_flat` | 52,857.46 | 72,317.12 | 1.37× | 66,808.68 | 133,751.39 | 完成 |
| SAC `g1_walk_flat` | 待跑 | 待跑 | 待算 | 38,496.57 | 64,022.10 | smoke/500轮 |
| SAC `g1_motion_tracking` | 待跑 | 待跑 | 待算 | 28,471.35 | 51,320.66 | smoke/500轮 |
| FlashSAC `g1_walk_flat` | 待跑 | 待跑 | 待算 | 8,742.86 | 16,474.44 | smoke/500轮 |
| FlashSAC `g1_motion_tracking` | 待跑 | 待跑 | 待算 | 8,355.22 | 15,759.63 | smoke/500轮 |

“待跑”是实验状态，不是将旧结果冒充 TorchEnv 结果；正式 run 完成后应直接更新
`data/throughput_torchenv_500.csv` 和本表。`g1_flip_tracking`/`allegro_inhand`
列为 N/A，而不是失败。

Go2 单机 50–499 均值相对旧栈下降 20.88%，双机下降 45.93%，双/单扩展为 1.37×；
末 20 轮分别为 54,350.65 与 103,287.60 steps/s。末 20 轮质量单机
53.14/998.13、双机 53.53/1000.00（reward/episode length），训练健康但吞吐已超过
10% 的明显退化阈值。由于依赖和 owner 同时变化，此数值是整合栈 delta；phase
timing 完成前不把下降全部归因于 TorchEnv。

## 质量与诊断

除吞吐外必须保存末 20 iteration reward/episode length、iteration time、PPO
collection/learning、off-policy replay/inference wait、DP sync time、正常 shutdown
和 runtime manifest。旧任务 owner 在 main 期间也有 reward/curriculum 调整，因此质量
对比是同任务名同预算的端到端栈对比，不声称逐 bit MDP 等价。
