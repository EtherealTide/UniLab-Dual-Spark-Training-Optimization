#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bundle_root=$(cd -- "$script_dir/.." && pwd)
cluster_file=${CLUSTER_FILE:-$bundle_root/config/cluster.env}

if [[ ! -f $cluster_file ]]; then
    echo "Missing $cluster_file; copy config/cluster.env.example and edit it." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$cluster_file"

if [[ $# -ne 6 ]]; then
    echo "usage: $0 <single|dual> <ppo|sac|flashsac> <task> <envs> <run_name> <port>" >&2
    exit 2
fi

mode=$1
algo=$2
task=$3
envs=$4
run_name=$5
port=$6
max_iterations=${MAX_ITERATIONS:-500}

for value in "$mode" "$algo" "$task" "$envs" "$run_name" "$port"; do
    if [[ ! $value =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "Unsafe argument: $value" >&2
        exit 2
    fi
done
if [[ $mode != single && $mode != dual ]]; then
    echo "mode must be single or dual" >&2
    exit 2
fi
if [[ $algo != ppo && $algo != sac && $algo != flashsac ]]; then
    echo "algo must be ppo, sac, or flashsac" >&2
    exit 2
fi

remote_dir=$REMOTE_ROOT/UniLab
python=.venv/bin/python
common_args="task=$task/mujoco training.log_dir=logs/$run_name training.no_play=true algo.num_envs=$envs algo.max_iterations=$max_iterations"

if [[ $algo == ppo ]]; then
    entry=src/unilab/scripts/train_rsl_rl.py
    algo_args=
elif [[ $algo == sac ]]; then
    entry=src/unilab/scripts/train_sac.py
    algo_args="algo.batch_size=8192 algo.updates_per_step=8 algo.policy_frequency=4"
else
    entry=src/unilab/scripts/train_flashsac.py
    algo_args="algo.batch_size=8192 algo.updates_per_step=8 algo.policy_frequency=4"
fi

if [[ $mode == single ]]; then
    ssh -F /dev/null "$HOST0" \
        "cd $remote_dir && $python $entry $common_args $algo_args > /tmp/${run_name}_single.log 2>&1"
    echo "DONE:$run_name:0"
    exit 0
fi

nccl_env="NCCL_IB_DISABLE=0 NCCL_IB_HCA=$NCCL_IB_HCA NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_NET_GDR_LEVEL=SYS"

if [[ $algo == ppo ]]; then
    rank1_cmd="cd $remote_dir && env $nccl_env $python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --master_addr=$MASTER_ADDR --master_port=$port --node_rank=1 $entry $common_args > /tmp/${run_name}_rank1.log 2>&1"
    rank0_cmd="cd $remote_dir && env $nccl_env $python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --master_addr=$MASTER_ADDR --master_port=$port --node_rank=0 $entry $common_args > /tmp/${run_name}_rank0.log 2>&1"
else
    rank1_cmd="cd $remote_dir && env $nccl_env UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE=2 UNILAB_DP_RANK=1 UNILAB_DP_RENDEZVOUS_URL=tcp://$MASTER_ADDR:$port UNILAB_DP_LOG_DIR=logs/$run_name $python $entry $common_args $algo_args > /tmp/${run_name}_rank1.log 2>&1"
    rank0_cmd="cd $remote_dir && env $nccl_env UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE=2 UNILAB_DP_RANK=0 UNILAB_DP_RENDEZVOUS_URL=tcp://$MASTER_ADDR:$port UNILAB_DP_LOG_DIR=logs/$run_name $python $entry $common_args $algo_args > /tmp/${run_name}_rank0.log 2>&1"
fi

ssh -F /dev/null "$HOST1" "$rank1_cmd" &
rank1_ssh_pid=$!
sleep 2

set +e
ssh -F /dev/null "$HOST0" "$rank0_cmd"
rank0_rc=$?
wait "$rank1_ssh_pid"
rank1_rc=$?
set -e

if [[ $rank0_rc -ne 0 || $rank1_rc -ne 0 ]]; then
    echo "FAILED:$run_name:rank0=$rank0_rc:rank1=$rank1_rc" >&2
    exit 1
fi
echo "DONE:$run_name:0"
