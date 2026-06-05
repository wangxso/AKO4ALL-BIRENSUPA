<h1 align="center">AKO4ALL-BIRENSUPA</h1>
<p align="center"><b>Agentic Kernel Optimization for All — 壁仞 BIRENSUPA Extension</b></p>

<p align="center">
  <a href="https://github.com/TongmingLAIC/AKO4ALL"><img src="https://img.shields.io/badge/Fork%20of-AKO4ALL-blue" alt="Fork of AKO4ALL"></a>
  <a href="https://tongminglaic.github.io/AKO"><img src="https://img.shields.io/badge/Original%20Project-Page-blue" alt="Original Project Page"></a>
  <a href="https://github.com/TongmingLAIC/AKO4X"><img src="https://img.shields.io/badge/Original-GitHub-blue?logo=github" alt="Original GitHub"></a>
</p>

<p align="center"><b>If you find our work useful, please consider giving us a star 🌟</b></p>

---

## About This Fork

**This is a forked and extended version of [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL)** with added support for **壁仞 BIRENSUPA** GPU kernels.

### What's Added

- **BIRENSUPA Backend Support** — Auto-detection and optimization for 壁仞 SUPA kernels
- **SUPA Programming Knowledge** — Integrated documentation for BIRENSUPA programming model
- **BRCC Compiler Support** — Compilation guidance for 壁仞's BRCC compiler
- **SupaProfiler Integration** — Profiling support for 壁仞 GPU

### Original Project

For the original AKO4ALL project (NVIDIA CUDA/Triton optimization), please visit:
- **GitHub**: https://github.com/TongmingLAIC/AKO4ALL
- **Project Page**: https://tongminglaic.github.io/AKO
- **Tech Report**: https://tongminglaic.github.io/AKO/assets/ako-tech-report.pdf

---

## News

<p align="center">
  <img src="assets/speedup_vs_expert.png" alt="Bar chart: AKO4ALL geomean speedup over the FlashInfer expert on four inference operators — GQA decode 1.36x, MLA decode 1.21x, MLA prefill 1.50x, RMSNorm 1.12x — on NVIDIA B200; all four bars clear the 1.0x expert line." width="640" />
  <br/>
  <i>Geomean speedup over the FlashInfer <b>expert</b> baseline (the strongest kernel FlashInfer ships) on NVIDIA B200 — every op clears the expert. Full table in <a href="#results">Results</a>.</i>
</p>

## News

- 📄 **[2026.05.31]** The **[AKO tech report](https://tongminglaic.github.io/AKO/assets/ako-tech-report.pdf)** is now available.
- 🚀 **[2026.05.31]** [**AKO4X**](https://github.com/TongmingLAIC/AKO4X) is now open-source — the closed-loop, campaign-based system behind our [MLSys 2026 competition](https://mlsys26.flashinfer.ai/) entry.
- ✨ **[2026.05.31]** **AKO4ALL** is now a single drop-in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill — invoke it in any working directory.
- 🚀 **[2026.03.24]** AKO4ALL is released.

**Table of Contents**

- [What is AKO4ALL?](#what-is-ako4all)
- [What You Provide](#what-you-provide)
- [Install](#install)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [Results](#results)
- [Agent Behavior](#agent-behavior)
- [Permissions](#permissions)
- [Repo Layout](#repo-layout)
- [Example: SOL-ExecBench](#example-sol-execbench)
- [Anti-Cheat](#anti-cheat)
- [FAQ](#faq)
- [Tech Report](#tech-report)
- [Acknowledgments](#acknowledgments)

## What is AKO4ALL?

**AKO4ALL is automated GPU kernel optimization, packaged as a single [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill.** Drop a kernel into your working directory, invoke the skill, and the agent bootstraps a workspace in place and iteratively rewrites the kernel for maximum performance — profile, edit, benchmark, repeat, until the kernel stops getting faster. The skill is a single `SKILL.md` protocol document; other coding agents can drive the same loop by following it directly.

<p align="center">
  <img src="assets/hero.png" alt="A cartoon robot agent at a single workshop bench iterates on a glowing GPU kernel cube — with iteration logbook, profiler, timer, and commit stamp at hand, and a speedup chart on the chalkboard — illustrating AKO4ALL's single-session, drop-in kernel optimization loop." width="780" />
</p>

**AKO4ALL vs [AKO4X](https://github.com/TongmingLAIC/AKO4X)** — two sibling projects with different framings.

- **Form:** AKO4ALL is a single Claude Code skill that runs in place in your working directory. AKO4X is a template repository that builds a fresh isolated child environment for each run.
- **Unit of work:** AKO4ALL is *one continuous optimization run on one kernel* — resumable across chat sessions, but with no memory carried into the next run on the same kernel. AKO4X is a *multi-round campaign on one operator* — a persistent master agent spawns rounds back-to-back, and a per-operator archive accumulates each round's results so the Nth round is informed by all earlier rounds.
- **Harness evolution:** AKO4ALL's protocol is static — you can supply behavior directives, but the skill itself does not change. AKO4X has an opt-in *harness co-evolution* mode in which the agent, after each round, can propose improvements to its own knowledge and tooling layer that the master gates and applies — no analogue here.

Reach for AKO4ALL when you just want a kernel optimized; reach for AKO4X when you want infrastructure around the optimization.

## What You Provide

A kernel and at least one set of test inputs are required. Everything else is optional.

- **Kernel** (required) — The kernel to optimize. Can be a single file or a directory. Supports Triton, CUDA, C++, TileLang, CuTe DSL, Python, or any language that can be benchmarked.
- **Reference implementation** (optional) — Used as the correctness golden. If absent, the original kernel is used.
- **Inputs** (at least one set required) — In any form: hardcoded inside the kernel/reference, defined in your benchmark script, a separate Python file with `get_inputs()`, or raw data files (`.npz`, `.bin`, shape lists, etc.) that the agent wires up itself. The agent assembles whatever is provided into something the bench script can call.
- **Benchmark script** (optional) — Your own benchmark script; the agent reads it to figure out how to run it. If no benchmark script is provided, the built-in [KernelBench](https://github.com/ScalingIntelligence/KernelBench) evaluator is used automatically (no setup needed beyond PyTorch).
- **Knowledge** (optional) — Reference materials for the agent: algorithm descriptions, papers, design docs, or any background knowledge that helps inform the optimization.
- **Hints** (optional) — Directives for the agent: optimization constraints, focus areas, and behavior controls (e.g., whether to allow web search).

## Install

Clone the repo directly into Claude Code's skills directory:

```bash
git clone https://github.com/wangxso/AKO4ALL-BIRENSUPA.git ~/.claude/skills/ako4all
```

Or, if you already have it cloned somewhere else, symlink it:

```bash
ln -s /path/to/AKO4ALL-BIRENSUPA ~/.claude/skills/ako4all
```

## BIRENSUPA Prerequisites

> **⚠️ Important: 壁仞 BIRENSUPA knowledge documents are NOT included in this repository due to commercial confidentiality.**

To use BIRENSUPA kernel optimization, you **must** obtain the following documents from 壁仞科技 (Biren Technology):

### Required Documents

| Document | Description | Purpose |
|----------|-------------|---------|
| BIRENSUPA 编程指南 | Core programming model, kernel functions, memory hierarchy | Understand SUPA programming |
| BIRENSUPA Driver 安装指南 | Driver installation and setup | Environment setup |
| 壁仞 BRPyTorch 插件用户指南 | PyTorch integration | Framework support |

### How to Obtain

Contact 壁仞科技产品服务部门 to request these documents. You will need:

1. A valid business relationship with 壁仞科技
2. Agreement to their confidentiality terms
3. Proper authorization for document access

### Setup After Obtaining Documents

Once you have the documents:

1. Place them in the `knowledge/` directory:
   ```bash
   cp /path/to/壁仞文档/*.md knowledge/
   ```

2. Verify the files are in place:
   ```bash
   ls knowledge/
   # Should show: BIRENSUPA_编程指南_V1.8.0_202507.md, etc.
   ```

3. The optimization agent will automatically read these files when optimizing 壁仞 SUPA kernels.

> **Note:** These documents are excluded from git tracking via `.gitignore` to protect commercial information. They remain local to your workspace only.

## How to Use

### Mode 1 — Interactive (default)

Open Claude Code in a directory containing your kernel, then ask it to optimize the kernel. Files in the working directory are inspected automatically; you can also point at external paths. Optional constraints — iteration cap, language preference, dependency policy, etc. — included in the prompt will be merged into `HINTS.md`.

The skill presents a resolved plan, asks if anything is genuinely ambiguous, and starts iterating. You can interrupt at any point to redirect.

<details>
<summary>Optional: pre-organize your workspace</summary>

If you want the conventional layout the skill recognizes:

```
<workspace>/
├── source/                      # Kernel + optional reference and inputs
│   ├── kernel.py
│   ├── reference.py             # Optional
│   └── inputs.py / data.npz ... # Optional — input data, any format
├── bench/                       # Your own benchmark script (optional)
│   └── ...
├── knowledge/                   # Reference materials (optional)
└── HINTS.md                     # Agent directives (optional)
```

These are conventions, not requirements. If your files are organized differently, just point Claude at them.

</details>

### Mode 2 — Batch via subagents (advanced)

For optimizing several related kernels in parallel, in Claude Code the main agent can spawn one subagent per kernel via the `Task` tool.

> ⚠️ **Subagents can't ask back.** In Claude Code, subagents lack `AskUserQuestion`. The main agent composes each subagent's prompt from your instructions plus what's in each kernel's directory; any decision you don't pin down there will fall to `SKILL.md` defaults. So include the things you care about (per-kernel paths, optional reference / inputs locations, language or iteration-cap constraints, custom branch names) — and let the rest default.

### Mode 3 — Remote GPU execution

For optimizing kernels on remote GPU servers (e.g., 壁仞 GPU cluster), configure remote execution:

```bash
# In HINTS.md or before running claude:
export REMOTE_HOST="user@192.168.1.100"  # SSH connection
export REMOTE_DIR="/tmp/ako4all_bench"   # Remote working directory
export REMOTE_PYTHON="python3"           # Remote Python
```

**How it works:**
1. Solution files are synced to remote via `rsync`
2. Benchmark runs on remote GPU via SSH
3. Results are fetched back to local machine
4. Trajectory stored locally in `trajectory/`

**Requirements:**
- SSH key-based authentication (no password prompts)
- `rsync` on both local and remote machines
- Remote GPU with drivers installed (NVIDIA or 壁仞)

**Example workflow:**

```bash
# Configure remote server
export REMOTE_HOST="user@壁仞-server"
export REMOTE_DIR="/work/ako4all"

# Start optimization
claude
# "优化 source/kernel.py"
```

The agent automatically handles file sync, remote execution, and result collection.

## Requirements

### Common Requirements

- A coding agent (e.g., [Claude Code](https://docs.anthropic.com/en/docs/claude-code))
- Python **>= 3.10** for the built-in evaluator (or whatever your custom bench script requires)
- Git

### NVIDIA CUDA/Triton Optimization

- NVIDIA GPU with CUDA-enabled PyTorch (only for the built-in evaluator; a remote-execution bench script may not need either locally)
- NVIDIA Nsight Compute (`ncu`) — recommended; its version must match your CUDA / PyTorch CUDA build. If unavailable, the loop still proceeds (no profiling — the agent reasons from runtime stats instead).

### 壁仞 BIRENSUPA Optimization

- **壁仞 GPU** (BR106 series or later) with BIRENSUPA Driver installed
- **BIRENSUPA SDK** — includes BRCC compiler, runtime, and tools
- **Knowledge Documents** — Required 壁仞 documentation (see [BIRENSUPA Prerequisites](#birensupa-prerequisites)):
  - BIRENSUPA 编程指南
  - BIRENSUPA Driver 安装指南
  - 壁仞 BRPyTorch 插件用户指南 (if using PyTorch)
- **BRCC Compiler** — for compiling SUPA kernels
- **SupaProfiler** (optional) — for kernel profiling

> **Note:** During the loop the agent may run `pip install` / `apt install` etc. to fill in missing dependencies. A container or virtual environment is recommended for isolation; to forbid installs, add a directive to `HINTS.md` (see [Agent Behavior](#agent-behavior)) or restrict via [Permissions](#permissions).

## How It Works

The agent creates an `opt/<kernel>` branch, copies your kernel to `solution/`, and generates `scripts/bench.sh` to wrap your benchmark. After verifying the baseline (and profiling it with `ncu` if available), it iterates: edit `solution/` → benchmark → log → commit. After 3 stagnant iterations it re-profiles, plans the next direction, and continues. Defaults like the stall threshold live in `HINTS.md`; iteration history lives in `ITERATIONS.md`. See [`SKILL.md`](SKILL.md) for the full protocol.

## Results

<details>
<summary><b>A drop-in skill that turns one prompt into an expert-beating kernel in under an hour — a clean sweep over FlashInfer's expert on all 4 inference ops.</b></summary>

<br>

Driven by **Claude Opus 4.8** (1M context, max thinking effort), on four FlashInfer-Bench inference operators AKO4ALL rewrote the reference into Triton kernels that **all beat FlashInfer's strongest _expert_ kernel** — geomean speedup over the FlashInfer expert on NVIDIA B200, all correctness-passing:

| Operator | Speedup vs FlashInfer expert | Workloads |
|---|---|---|
| GQA paged decode | **1.36×** | 48 |
| MLA paged decode | **1.21×** | 47 |
| MLA paged prefill | **1.50×** | 38 |
| RMSNorm h128 | **1.12×** | 14 |

*All four operators we ran are shown here — we time-boxed this round to four kernels, so these are not the best four cherry-picked from a larger sweep.*

**What one kernel takes to optimize** — a typical run, end-to-end from the initial prompt to the agent finishing:

| End-to-end | Input | Output | Cache read | Cache write |
|---|---|---|---|---|
| ~55 min | ~15k | ~253k | ~17.6M | ~752k |

A single drop-in skill — competitive-with-expert kernels in under an hour, from a single prompt.

</details>

## Agent Behavior

A few defaults worth knowing:

- **Language switching** — the agent may rewrite your kernel in a different language (e.g., Triton → CUDA) to chase performance.
- **Web search** — enabled; the agent searches for optimization ideas after consecutive stagnant iterations.
- **Profiling** — `ncu` runs on the baseline before iter-1 and again after 3 stagnant iterations.

To adjust any of this — or to add your own directives (optimization constraints, dependency policies, iteration caps, ...) — edit `HINTS.md` in your workspace. The file is self-documenting.

## Permissions

The optimization loop involves running shell commands (compiling, benchmarking, profiling). By default, most coding agents prompt for approval on each command. To run fully unattended, grant the necessary permissions through your agent's configuration.

For Claude Code, the simplest option is to bypass all permission checks:

```bash
claude --dangerously-skip-permissions
```

<details>
<summary>Granular control via settings file</summary>

Create `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)", "Read(*)", "Write(*)", "Edit(*)",
      "Glob(*)", "Grep(*)", "Agent(*)",
      "WebFetch(*)", "WebSearch(*)"
    ]
  }
}
```

</details>

For other agents, consult their documentation on permission / auto-approve settings.

## Repo Layout

```
AKO4ALL-BIRENSUPA/
├── SKILL.md              # Entry point — loaded by Claude Code when the skill triggers
├── HINTS.md              # Scaffold: agent behavior defaults
├── ITERATIONS.md         # Scaffold: iteration log template
├── bench-wrapper.sh      # Scaffold: bench script template
├── bench/kernelbench/    # Scaffold: built-in KernelBench evaluator
└── knowledge/            # ⚠️ Local only — NOT in git (see BIRENSUPA Prerequisites)
    └── .gitkeep
```

Scaffold files are copied into each workspace on first bootstrap, never overwriting existing files. `README.md`, `LICENSE`, and `assets/` are project metadata — not installed anywhere.

**⚠️ Note about `knowledge/`:** This directory is excluded from git tracking (via `.gitignore`) to protect commercial 壁仞 documents. You must manually obtain and place the BIRENSUPA documents there (see [BIRENSUPA Prerequisites](#birensupa-prerequisites)).

## Example: SOL-ExecBench

[SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) contains 235 real-world DL kernel problems from NVIDIA. AKO4ALL can optimize any of them by pointing at the problem directory directly — no copying into a local workspace.

1. Install the skill (see [Install](#install)) and clone SOL-ExecBench separately. Set up its environment and `ncu` per its own instructions.

2. Activate the SOL-ExecBench environment and open Claude Code in any working directory:

   ```bash
   claude
   ```

3. Give it a prompt (replace the paths and iteration cap):

   ```
   Optimize the kernel at <sol-bench>/data/benchmark/L1/001_attention_softmax_dropout_value_matmul_backward.
   Bench with SOL-ExecBench at <sol-bench>; its dependencies are already installed.
   Optimize for up to 30 iterations. Stop early only if all viable approaches are exhausted.
   ```

The skill resolves the external path, wraps SOL-ExecBench's bench command as `scripts/bench.sh`, and starts iterating.

## Anti-Cheat

Three layers of protection against the agent gaming the metric instead of achieving real speedup:

1. **Agent rules.** The skill instructs the agent to pursue genuine latency reduction — no CUDA stream injection, no timing manipulation, no returning uninitialized results. Full list in [`SKILL.md`](SKILL.md) "Gotchas".

2. **Evaluator checks** (built-in KernelBench). Flags suspicious >10× speedups for review, and strips any input-generating code (e.g., `get_inputs`) from the solution file before evaluation — so the solution can't choose what it's tested against.

3. **Stricter enforcement** (optional). Provide a custom bench script with static analysis — e.g., KernelBench's [`kernel_static_checker.py`](https://github.com/ScalingIntelligence/KernelBench) — to reject solutions that contain disallowed patterns before they are even timed.

## FAQ

**1. Can I use AKO4ALL for a whole benchmark suite?**
For ad-hoc optimization of several kernels at once, see [Mode 2](#mode-2--batch-via-subagents-advanced). For campaign-grade multi-round optimization — cross-run memory, an archive of variants feeding the next round, and master/sub agent separation — see [AKO4X](https://github.com/TongmingLAIC/AKO4X) (default benchmark: flashinfer-bench, swappable via adapter).

**2. My agent isn't trying hard enough — keeps tuning configs instead of rewriting.**
Agent laziness is a known failure mode (staying in PyTorch, only tuning configurations, skipping profiling). Intervene with specific guidance, or add directives to `HINTS.md` ahead of time. The point of the loop is to *rewrite* the kernel — switch languages (Triton → CUDA, etc.) when it helps.

**3. Which model should I use?**
Model capability strongly influences optimization quality. We recommend [Claude Opus](https://docs.anthropic.com/en/docs/about-claude/models).

**4. How do I cap iterations?**
By default there is no cap — the agent decides when to stop on its own. To enforce a limit, add a directive to `HINTS.md`, or include it in your prompt (e.g., `Optimize for up to 30 iterations. Stop early only if all viable approaches are exhausted.`).

**5. My bench script uses a remote service (e.g., Modal). Does that work?**
Yes. As long as your bench script runs from the command line and prints results to stdout. The local machine doesn't need a GPU / CUDA / PyTorch in that case — just whatever your wrapper script imports.

**6. Can I intervene during optimization?**
Yes. Interrupt to redirect, or to manually edit files in `solution/` — then tell the agent to continue.

## Tech Report

The **[AKO tech report](https://tongminglaic.github.io/AKO/assets/ako-tech-report.pdf)** is now available.

## Citation

If you find AKO useful, please cite the original work:

```bibtex
@misc{ako2026,
  title        = {{AKO}: Agentic Kernel Optimization},
  author       = {Shuxiao Xie and Shuyang Xie and Dezhi Ran and Wei Yang and Tao Xie},
  year         = {2026},
  howpublished = {\url{https://tongminglaic.github.io/AKO}},
  note         = {Technical report}
}
```

### Acknowledgments

This project is a fork of [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) by TongmingLAIC. We extend our gratitude to the original authors for their excellent work on agentic kernel optimization.

## Acknowledgments

We would like to thank the following open-source projects that inspired and supported the development of AKO:

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — for providing the benchmark and evaluation format used by AKO4ALL's built-in evaluator.
- [autoresearch](https://github.com/karpathy/autoresearch) and [autokernel](https://github.com/RightNow-AI/autokernel) — AKO's design was inspired by their work on autonomous optimization loops.
- **壁仞科技** — for the BIRENSUPA programming model documentation and support.
