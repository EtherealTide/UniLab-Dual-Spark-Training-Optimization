# Sourced in each remote rank shell before Python starts. The coordinator sends
# this snippet over SSH, so the worker does not need a copy of the bundle.
if [[ -z ${TRITON_PTXAS_PATH:-} && -x /usr/local/cuda-13.0/bin/ptxas ]]; then
    export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
fi
if [[ -n ${TRITON_PTXAS_PATH:-} ]]; then
    if [[ $TRITON_PTXAS_PATH != /* || ! -f $TRITON_PTXAS_PATH || ! -x $TRITON_PTXAS_PATH ]]; then
        echo "[launcher] TRITON_PTXAS_PATH must be an absolute executable path: $TRITON_PTXAS_PATH" >&2
        exit 1
    fi
    export TRITON_PTXAS_PATH
    printf '[launcher] Triton assembler: %s\n' "$TRITON_PTXAS_PATH"
fi
