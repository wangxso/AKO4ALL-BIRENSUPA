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
