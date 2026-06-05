#!/bin/bash
# Benchmark wrapper with trajectory tracking and remote execution support
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

# --- Remote Execution Configuration ---
# Set REMOTE_HOST to enable remote execution via SSH
# Example: export REMOTE_HOST="user@192.168.1.100"
# Example: export REMOTE_HOST="user@壁仞-gpu-server"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/ako4all_bench}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"

# --- Docker Container Configuration ---
# Set DOCKER_CONTAINER to execute inside a Docker container
# Example: export DOCKER_CONTAINER="biren-vllm"
# Example: export DOCKER_CONTAINER="my-gpu-container"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-}"
DOCKER_WORK_DIR="${DOCKER_WORK_DIR:-/workspace}"

# Detect toolkit-vs-torch ambiguity and warn — don't auto-set CUDA_HOME.
# Auto-set is unreliable when active conda env != target python env (e.g.,
# base shell running envs/X/bin/python directly), and a wrong CUDA_HOME
# silently produces ABI mismatches in torch.cpp_extension / load_inline
# (e.g., cu130 torch + cu117 nvcc → cudaDeviceProp mismatch → SIGFPE).
# Common on multi-CUDA hosts.
if ! [ -x "${CUDA_HOME%%:*}/bin/nvcc" ]; then
    TORCH_CU=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
    if [ -n "$TORCH_CU" ]; then
        echo "[bench-wrapper] CUDA_HOME=${CUDA_HOME:-(unset)} → no nvcc found; torch built with CUDA $TORCH_CU." >&2
        echo "  For load_inline / cpp_extension, export CUDA_HOME to the env whose nvcc matches CUDA $TORCH_CU, e.g.:" >&2
        echo "    export CUDA_HOME=\$(python -c 'import sys; print(sys.prefix)')" >&2
    fi
fi

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Remote execution function ---
run_remote() {
    local cmd="$1"
    if [ -n "$REMOTE_HOST" ]; then
        echo "[bench-wrapper] Executing remotely on $REMOTE_HOST"
        ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "cd $REMOTE_DIR && $cmd"
    else
        eval "$cmd"
    fi
}

# --- Docker execution function ---
run_in_docker() {
    local cmd="$1"
    if [ -n "$DOCKER_CONTAINER" ]; then
        echo "[bench-wrapper] Executing in Docker container: $DOCKER_CONTAINER"
        if [ -n "$REMOTE_HOST" ]; then
            # Remote Docker execution
            ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
                "docker exec -w $DOCKER_WORK_DIR $DOCKER_CONTAINER bash -c '$cmd'"
        else
            # Local Docker execution
            docker exec -w "$DOCKER_WORK_DIR" "$DOCKER_CONTAINER" bash -c "$cmd"
        fi
    else
        eval "$cmd"
    fi
}

# --- Sync files to remote ---
sync_to_remote() {
    if [ -n "$REMOTE_HOST" ]; then
        echo "[bench-wrapper] Syncing files to $REMOTE_HOST:$REMOTE_DIR"
        ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"
        rsync -avz --exclude='trajectory' --exclude='.git' --exclude='__pycache__' \
            solution/ scripts/ bench/ knowledge/ "$REMOTE_HOST:$REMOTE_DIR/"
        [ -f HINTS.md ] && rsync -avz HINTS.md "$REMOTE_HOST:$REMOTE_DIR/"
        [ -f ITERATIONS.md ] && rsync -avz ITERATIONS.md "$REMOTE_HOST:$REMOTE_DIR/"

        # Copy files into Docker container if specified
        if [ -n "$DOCKER_CONTAINER" ]; then
            echo "[bench-wrapper] Copying files to Docker container: $DOCKER_CONTAINER"
            ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
                "docker cp $REMOTE_DIR/. $DOCKER_CONTAINER:$DOCKER_WORK_DIR/"
        fi
    fi
}

# --- Fetch results from remote ---
fetch_from_remote() {
    if [ -n "$REMOTE_HOST" ]; then
        echo "[bench-wrapper] Fetching results from $REMOTE_HOST"
        if [ -n "$DOCKER_CONTAINER" ]; then
            # Fetch from Docker container
            ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
                "docker cp $DOCKER_CONTAINER:$DOCKER_WORK_DIR/trajectory/ trajectory/ 2>/dev/null || true"
            ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
                "docker cp $DOCKER_CONTAINER:$DOCKER_WORK_DIR/_bench_output.txt . 2>/dev/null || true"
        else
            rsync -avz "$REMOTE_HOST:$REMOTE_DIR/trajectory/" trajectory/ 2>/dev/null || true
            rsync -avz "$REMOTE_HOST:$REMOTE_DIR/_bench_output.txt" . 2>/dev/null || true
        fi
    fi
}

# --- Main execution ---
if [ -n "$REMOTE_HOST" ] || [ -n "$DOCKER_CONTAINER" ]; then
    # Remote/Docker execution mode
    sync_to_remote

    # Run benchmark remotely/in Docker
    set +e
    run_in_docker "$REMOTE_PYTHON bench/kernelbench/bench.py --ref <ref> --solution solution/<kernel> --verbose 2>&1 | tee _bench_output.txt"
    BENCH_EXIT=$?
    set -e

    fetch_from_remote
else
    # Local execution mode (original behavior)
    set +e
    {{BENCH_COMMAND}}
    BENCH_EXIT=$?
    set -e
fi

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/"
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
