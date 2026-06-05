# Hints

<!-- User-supplied behavior directives. The skill reads this at session start
     and respects any constraint named here. Examples:
     - Optimization constraints or focus areas (e.g., "Prefer Triton over raw CUDA")
     - Strategies to try or avoid     (e.g., "Avoid shared memory")
     - Agent behavior controls         (e.g., "Stop after 5 iterations")
     - Dependency policies             (e.g., "Do not install new packages")
     - Environment constraints         (e.g., "ncu is unavailable on this host",
                                             "8GB device memory limit")

     The skill's own protocol (iteration steps, stall handling, ncu fallback,
     stopping rules) lives in SKILL.md — do not duplicate it here.
-->

## 壁仞 BIRENSUPA Specific Hints

<!-- When optimizing 壁仞 SUPA kernels, the following directives help guide the agent. -->

### SUPA Compilation
- Use `brcc` compiler for SUPA kernel compilation
- Compile flags: `-std=c++17 -O3 --supa-arch=<arch>` (e.g., `--supa-arch=br106`)
- Link against SUPA runtime: `-lsupa_runtime`

### SUPA Profiling
- Use `suProfiler` for kernel profiling (instead of `ncu`)
- Environment: `export SUPA_VISIBLE_DEVICES=0` to select device
- For synchronous execution: `export SUPA_LAUNCH_BLOCKING=1`

### SUPA Optimization Strategies
- Memory optimization: Use `__shared__` memory for data reuse
- Thread block sizing: 512-1024 threads per block is typical
- Use `__global_mega__` for tensor core operations
- Consider memory coalescing for global memory accesses

## Remote Execution Configuration

<!-- Configure remote GPU server execution for benchmark runs. -->

### Remote Server Settings

```bash
# SSH connection to remote GPU server
REMOTE_HOST="user@192.168.1.100"  # or user@壁仞-gpu-server

# Remote working directory
REMOTE_DIR="/tmp/ako4all_bench"

# Remote Python executable
REMOTE_PYTHON="python3"
```

### Remote Environment Setup

On the remote server, ensure:
- GPU drivers installed (NVIDIA/CUDA or 壁仞 BIRENSUPA)
- Python 3.10+ with PyTorch
- Required dependencies (triton, etc.)
- SSH key-based authentication configured

### Remote Workflow

1. Files are synced via `rsync` to `REMOTE_DIR`
2. Benchmark executes on remote GPU
3. Results fetched back to local `trajectory/`

### Example

```bash
# Local machine
export REMOTE_HOST="user@壁仞-server"
export REMOTE_DIR="/work/ako4all"

# Then optimize
claude
# "优化 source/kernel.py"
```
