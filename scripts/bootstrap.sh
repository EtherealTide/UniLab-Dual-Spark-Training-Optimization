#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bundle_root=$(cd -- "$script_dir/.." && pwd)
# shellcheck disable=SC1091
source "$bundle_root/versions.env"

target_root=${1:-/home/nvidia/unilab-dual-spark-repro}
uv_bin=${UV_BIN:-uv}

require_clean_repo() {
    local repo_dir=$1
    if [[ -n $(git -C "$repo_dir" status --porcelain) ]]; then
        echo "Refusing to modify dirty repository: $repo_dir" >&2
        exit 1
    fi
}

clone_if_missing() {
    local url=$1
    local repo_dir=$2
    if [[ -e $repo_dir && ! -d $repo_dir/.git ]]; then
        echo "Path exists but is not a Git repository: $repo_dir" >&2
        exit 1
    fi
    if [[ ! -d $repo_dir/.git ]]; then
        git clone "$url" "$repo_dir"
    fi
    require_clean_repo "$repo_dir"
}

apply_pinned_patch() {
    local repo_dir=$1
    local base_commit=$2
    local expected_tree=$3
    local patch_file=$4
    local current_tree
    current_tree=$(git -C "$repo_dir" rev-parse HEAD^{tree})
    if [[ $current_tree == "$expected_tree" ]]; then
        return
    fi
    git -C "$repo_dir" fetch origin "$base_commit"
    git -C "$repo_dir" checkout --detach "$base_commit"
    git -C "$repo_dir" am "$patch_file"
    current_tree=$(git -C "$repo_dir" rev-parse HEAD^{tree})
    if [[ $current_tree != "$expected_tree" ]]; then
        echo "Tree hash mismatch after applying $patch_file" >&2
        echo "expected=$expected_tree actual=$current_tree" >&2
        exit 1
    fi
}

mkdir -p "$target_root"

clone_if_missing "$UNILAB_REPO" "$target_root/UniLab"
apply_pinned_patch \
    "$target_root/UniLab" \
    "$UNILAB_BASE_COMMIT" \
    "$UNILAB_OPT_TREE" \
    "$bundle_root/patches/UniLab/0001-perf-distributed-enable-RoCE-dual-Spark-training.patch"

clone_if_missing "$UNILAB_RL_REPO" "$target_root/unilab_rl"
apply_pinned_patch \
    "$target_root/unilab_rl" \
    "$UNILAB_RL_BASE_COMMIT" \
    "$UNILAB_RL_OPT_TREE" \
    "$bundle_root/patches/unilab_rl/0001-perf-dp-optimize-dual-node-CUDA-graph-synchronizatio.patch"

clone_if_missing "$UNISIM_REPO" "$target_root/UniSim"
git -C "$target_root/UniSim" fetch origin "$UNISIM_COMMIT"
git -C "$target_root/UniSim" checkout --detach "$UNISIM_COMMIT"
if [[ $(git -C "$target_root/UniSim" rev-parse HEAD) != "$UNISIM_COMMIT" ]]; then
    echo "UniSim commit mismatch" >&2
    exit 1
fi

cd "$target_root/UniLab"
"$uv_bin" sync --frozen --no-dev --extra mujoco --extra uni_rl
"$uv_bin" pip install --python .venv/bin/python --no-deps -e "$target_root/UniSim"
"$uv_bin" pip install --python .venv/bin/python --no-deps -e "$target_root/unilab_rl"

.venv/bin/python - "$UNILAB_PACKAGE_VERSION" "$UNISIM_PACKAGE_VERSION" \
    "$UNILAB_RL_PACKAGE_VERSION" "$TORCH_VERSION" "$MUJOCO_VERSION" <<'PY'
import importlib.metadata as metadata
import sys

expected = {
    "unilab": sys.argv[1],
    "unisim-core": sys.argv[2],
    "unilab-rl": sys.argv[3],
    "torch": sys.argv[4],
    "mujoco": sys.argv[5],
}
actual = {name: metadata.version(name) for name in expected}
for name, wanted in expected.items():
    if actual[name] != wanted:
        raise SystemExit(f"{name}: expected {wanted}, got {actual[name]}")
print("Pinned runtime verified:")
for name, version in actual.items():
    print(f"  {name}=={version}")
PY

echo "Reproducible checkout ready at $target_root"
