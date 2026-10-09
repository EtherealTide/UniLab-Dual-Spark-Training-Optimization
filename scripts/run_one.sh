#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bundle_root=$(cd -- "$script_dir/.." && pwd)
cluster_file=${CLUSTER_FILE:-$bundle_root/config/cluster.env}
iterations_override=${MAX_ITERATIONS:-}

if [[ $# -ne 6 ]]; then
    echo "usage: $0 <single|dual> <ppo|sac|flashsac> <task> <envs> <run_name> <port>" >&2
    exit 2
fi
if [[ ! -f $cluster_file ]]; then
    echo "Missing $cluster_file; copy config/cluster.env.example and edit it on the coordinator (the machine running this script)." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$cluster_file"

mode=$1
algo=$2
task=$3
envs=$4
run_name=$5
port=$6
max_iterations=${iterations_override:-${MAX_ITERATIONS:-500}}

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
if [[ ! $envs =~ ^[1-9][0-9]*$ || ! $max_iterations =~ ^[1-9][0-9]*$ || $run_name == . || $run_name == .. ]]; then
    echo "envs and MAX_ITERATIONS must be positive integers; run_name must not be . or .." >&2
    exit 2
fi
if [[ ! $port =~ ^[0-9]{1,5}$ ]] || (( 10#$port > 65535 )) || [[ $mode == dual && $port =~ ^0+$ ]]; then
    echo "port must be 0-65535 (1-65535 for dual mode)" >&2
    exit 2
fi
required=(HOST0 REMOTE_ROOT)
if [[ $mode == dual ]]; then
    required+=(HOST1 MASTER_ADDR NCCL_SOCKET_IFNAME NCCL_IB_HCA)
fi
for name in "${required[@]}"; do
    if [[ -z ${!name:-} ]]; then
        echo "Missing cluster setting: $name" >&2
        exit 2
    fi
done
if [[ $REMOTE_ROOT != /* ]]; then
    echo "REMOTE_ROOT must be an absolute path on each Spark." >&2
    exit 2
fi

# Respect ~/.ssh/config, including aliases and IdentityFile. Authentication must
# succeed without prompts, including coordinator -> HOST0 when it is localhost.
ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
remote_dir=$REMOTE_ROOT/UniLab
printf -v quoted_dir '%q' "$remote_dir"
printf -v quoted_run_dir '%q' "$remote_dir/logs/$run_name"
printf -v quoted_python '%q' "$remote_dir/.venv/bin/python"

preflight() {
    local host=$1
    echo "[preflight] Checking SSH, Python, and a fresh run directory on $host"
    if ! ssh -n "${ssh_options[@]}" "$host" \
        "test -x $quoted_python && { test ! -e $quoted_run_dir || { echo 'Run directory already exists; choose a new run_name.' >&2; exit 1; }; }"; then
        echo "Preflight failed on $host. Check passwordless SSH, REMOTE_ROOT, and run_name before retrying." >&2
        exit 1
    fi
}
preflight "$HOST0"
if [[ $mode == dual ]]; then
    preflight "$HOST1"
fi

python='.venv/bin/python -u'
common_args="task=$task/mujoco training.log_dir=logs/$run_name training.no_play=true ++training.log_interval=1 algo.num_envs=$envs algo.max_iterations=$max_iterations"
algo_args=
case "$algo" in
    ppo) entry=src/unilab/scripts/train_rsl_rl.py ;;
    sac) entry=src/unilab/scripts/train_sac.py ;;
    flashsac) entry=src/unilab/scripts/train_flashsac.py ;;
esac
if [[ $algo != ppo ]]; then
    algo_args='algo.batch_size=8192 algo.updates_per_step=8 algo.policy_frequency=4'
fi

run_rank() {
    local host=$1 label=$2 train_cmd=$3 log_file=$4
    local remote_cmd quoted_cmd quoted_log
    printf -v quoted_log '%q' "$log_file"
    # pipefail preserves the training exit code rather than the exit code of tee.
    printf -v remote_cmd 'set -o pipefail; cd %s && %s 2>&1 | tee %s' \
        "$quoted_dir" "$train_cmd" "$quoted_log"
    printf -v quoted_cmd '%q' "$remote_cmd"
    ssh -n "${ssh_options[@]}" "$host" "bash -c $quoted_cmd" 2>&1 |
        sed -u "s/^/[$label] /"
}

started=$SECONDS
echo "[run] mode=$mode algo=$algo task=$task envs_per_rank=$envs iterations=$max_iterations"
echo "[run] TensorBoard/checkpoints/summary: $HOST0:$remote_dir/logs/$run_name"
if [[ $mode == single ]]; then
    echo "[run] Console log: $HOST0:/tmp/${run_name}_single.log"
    if run_rank "$HOST0" single "env PYTHONUNBUFFERED=1 $python $entry $common_args $algo_args" \
        "/tmp/${run_name}_single.log"; then
        rank0_rc=0
    else
        rank0_rc=$?
    fi
    rank1_rc=0
    world_size=1
else
    printf -v nccl_env 'PYTHONUNBUFFERED=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=%q NCCL_SOCKET_IFNAME=%q NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_NET_GDR_LEVEL=SYS' \
        "$NCCL_IB_HCA" "$NCCL_SOCKET_IFNAME"
    printf -v master_addr '%q' "$MASTER_ADDR"
    if [[ $algo == ppo ]]; then
        launch="env $nccl_env $python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --master_addr=$master_addr --master_port=$port"
        rank0_cmd="$launch --node_rank=0 $entry $common_args"
        rank1_cmd="$launch --node_rank=1 $entry $common_args"
    else
        printf -v rendezvous '%q' "tcp://$MASTER_ADDR:$port"
        launch="env $nccl_env UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE=2 UNILAB_DP_RENDEZVOUS_URL=$rendezvous UNILAB_DP_LOG_DIR=logs/$run_name"
        rank0_cmd="$launch UNILAB_DP_RANK=0 $python $entry $common_args $algo_args"
        rank1_cmd="$launch UNILAB_DP_RANK=1 $python $entry $common_args $algo_args"
    fi
    echo "[run] Rendezvous: $MASTER_ADDR:$port"
    echo "[run] Console logs: $HOST0:/tmp/${run_name}_rank0.log and $HOST1:/tmp/${run_name}_rank1.log"
    run_rank "$HOST1" rank1 "$rank1_cmd" "/tmp/${run_name}_rank1.log" &
    rank1_ssh_pid=$!
    if run_rank "$HOST0" rank0 "$rank0_cmd" "/tmp/${run_name}_rank0.log"; then
        rank0_rc=0
    else
        rank0_rc=$?
        echo "[run] rank0 failed (exit=$rank0_rc); waiting for rank1 to exit." >&2
    fi
    if wait "$rank1_ssh_pid"; then
        rank1_rc=0
    else
        rank1_rc=$?
    fi
    world_size=2
fi

echo "[run] Launcher elapsed: $((SECONDS - started)) s; rank0_exit=$rank0_rc rank1_exit=$rank1_rc"
if [[ $rank0_rc -ne 0 || $rank1_rc -ne 0 ]]; then
    echo "FAILED:$run_name:rank0=$rank0_rc:rank1=$rank1_rc (see console logs above)" >&2
    exit 1
fi

# Send the formatter over stdin: only the coordinator needs this bundle.
printf -v quoted_summary '%q' "$remote_dir/logs/$run_name/run_summary.json"
if ! ssh "${ssh_options[@]}" "$HOST0" \
    "$quoted_python - $quoted_summary $max_iterations $world_size $envs $algo" \
    < "$script_dir/print_run_summary.py"; then
    echo "FAILED:$run_name:summary (training exited successfully, but final statistics could not be verified)" >&2
    exit 1
fi
echo "DONE:$run_name:0"
