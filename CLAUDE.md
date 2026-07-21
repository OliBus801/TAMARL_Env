# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**TrafficGym** (package name `tamarl`) is a GPU-accelerated Dynamic Traffic Assignment (DTA) environment for Multi-Agent Reinforcement Learning research. Its core is **TorchDNL**, a vectorized Dynamic Network Loading traffic simulator implemented entirely in PyTorch tensor ops (no Python-level per-agent loops in the hot path). RL agents choose routes for vehicles; TorchDNL simulates the resulting traffic dynamics (queueing, spillback, flow capacity) and returns travel times as rewards.

## Environment setup

Commands must run inside the `ml` conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml
```

If you see `ModuleNotFoundError: No module named 'torch'` (or similar), the conda env was not activated — prepend the line above.

Install (editable): `pip install -e .` (also installs `dev` extras: `pytest`, `ruff`).

## Common commands

```bash
# Run the full test suite
pytest

# Run a single test file / test
pytest tests/test_torchdnl.py
pytest tests/test_torchdnl.py::test_spillback

# Lint / format (ruff, configured in pyproject.toml)
ruff check .
ruff format .

# Benchmark the core DNL engine (use this exact scenario for perf comparisons — it runs fast)
PYTHONPATH=. python tests/benchmark_matsim_dnl.py tamarl/data/scenarios/grid_world/3x3 --population 100

# Bandit-formulation RL training (single-shot per-episode route choice)
python -m tamarl.rl.train_bandit --scenario tamarl/data/scenarios/grid_world/3x3 --population 100 --agent ucb
# Or via a JSON config (CLI flags override config values, which override the defaults in
# tamarl/data/configs/default_config.json):
python -m tamarl.rl.train_bandit --config tamarl/data/configs/train_config_1.json

# Sequential POMDP-formulation MARL training (IPPO/MAPPO, one decision per departure event)
python scripts/train_marl.py --scenario tamarl/data/scenarios/sioux_falls --algo mappo

# Generate the README demo GIF
./scripts/generate_demo_gif.sh
```

Tests are plain `assert`-based (pytest-discoverable, function names `test_*`); some also `print()` progress — read output on failure rather than relying solely on the assertion message. `tests/` also contains standalone benchmarking/profiling scripts (`benchmark_torchdnl.py`, `profile_dnl_workload_compare.py`, `compare_paths.py`) that are not pytest tests despite living in the same directory.

## Architecture

### Layered structure

```
tamarl/core/torchdnl.py          TorchDNL — the simulation engine (no RL/gym awareness)
tamarl/envs/dta_bandit_env.py    DTABanditEnv — thin orchestrator: reset(paths) → step() → rewards
tamarl/envs/*_wrapper.py         Gym/PettingZoo-facing formulations built on top of DTABanditEnv
tamarl/rl/                       Training loops + bandit-style RL agents
scripts/                         CLI entry points, profiling, network scaling, plot regeneration
```

### TorchDNL (`tamarl/core/torchdnl.py`)

A Structure-of-Arrays traffic simulator: every agent-state field (`status`, `current_edge`, `wakeup_time`, ...) is a flat tensor indexed by agent id, and every operation is a vectorized tensor op over *all* agents/edges at once — there is deliberately no per-agent Python loop in `step()`.

- Paths are stored in **CSR format** (`paths_flat` + `path_offsets`), not as a dense padded `[A, MaxPathLen]` tensor — this is the memory-critical representation for large scenarios (e.g. Berlin at 1%: dense would be ~1GB+). The old dense `paths` argument is deprecated and converted to CSR internally.
- Multi-leg agents (activity chains, not just single trips) are supported via `num_legs`/`act_end_times`/`act_durations`; leg boundaries inside `paths_flat` are marked with sentinel `-2`, and `-1` marks end-of-path.
- Each `step()` runs three phases in order, mirroring the MATSim DNL queue model:
  1. **`_process_nodes_A`** — capacity buffer → downstream spatial buffer, gated by storage capacity (with a `stuck_threshold` forced-entry override) and randomized per-upstream-link priority at merges.
  2. **`_process_links_B`** — spatial buffer → capacity buffer, gated by flow capacity and FIFO link ordering; also detects network exits.
  3. **`_schedule_demand_C`** — dispatches agents whose activity/departure time has arrived, and advances agents with more legs into their next activity.
- Agent status codes: `0` waiting/activity, `1` traveling (spatial buffer), `2` capacity buffer, `3` done, `4` exiter (awaiting next-leg dispatch).
- `finalize_stuck_agents()` must be called after the simulation loop ends to impute travel times for agents still in-network at `max_steps` (via a CPU/numpy fallback loop — this is intentionally not vectorized since it only runs once, on a small residual set).
- Optional instrumentation (`track_events`, `collect_link_tt`) is opt-in and adds pre-allocated GPU buffers; `collect_link_tt` in particular drives the empirical Nash-regret metrics used by the wrappers (see below) and interval-based dynamic link travel times.

### DTABanditEnv (`tamarl/envs/dta_bandit_env.py`)

The minimal orchestration layer: `reset(paths_flat, path_offsets)` (re)constructs a `TorchDNL`, `step()` runs the simulation to completion in a tight loop and returns `-travel_time` as reward. It knows nothing about RL formulations, action spaces, or route candidates — that's the wrappers' job.

### Formulation wrappers (`tamarl/envs/*_wrapper.py`)

All three implement the same `gymnasium.vector.VectorEnv`-like interface (`reset`/`step` over a *batch* of decision-makers) around a shared `DTABanditEnv`, differing only in how learning signal is aggregated across vehicles via an `aggregation_indices` tensor passed to the agent's `update()`:

- **`AgentLevelWrapper`** — one sub-env per *leg* of every vehicle (`aggregation_indices = arange(N)`, no sharing). Each leg picks among top-K candidate routes for its OD pair.
- **`ODLevelWrapper`** — same per-leg action space, but exposes `od_indices` so agents can maintain one parameter block *per OD pair* (`aggregation_indices = od_indices`).
- **`CentralizedLevelWrapper`** — all vehicles share a single parameter block (`aggregation_indices = zeros(N)`).

This aggregation-index abstraction (see `tamarl/rl/train_bandit.py:run_episode`, "Strategic Ignorance" comments) is why the bandit agents in `tamarl/rl/agents/` are formulation-agnostic: they always index into `self.weights[aggregation_indices]` and never need to know which wrapper is in use.

Candidate routes (top-K loopless paths per OD pair) are precomputed once per wrapper construction via `tamarl/envs/components/path_enumerator.py` (Yen/igraph, Sidetrack/rustworkx, or Penalty methods — see that file's docstring for complexity tradeoffs) and cached to disk per scenario; pass `--reload_paths` / `reload_paths=True` to force recomputation. Routes are stored in the same CSR layout as `TorchDNL` paths (`tamarl/envs/components/route_utils.py`) to avoid the dense `[NumOD, K, MaxRouteLen]` blowup on large networks.

`tamarl/envs/pomdp_wrapper.py` (`POMDPWrapper`) sits on top of `AgentLevelWrapper` and turns the one-shot bandit into a **sequential** decision process: it fast-forwards the DNL until an agent is due to depart, requests one action, writes it into the paths buffer, and resumes — used by `scripts/train_marl.py` for IPPO/MAPPO-style training where agents observe partial global state (`edge_occupancy`) at decision time rather than choosing all routes upfront.

`tamarl/envs/components/time_dependent_evaluator.py` reconstructs experienced (not free-flow) link travel times from a completed simulation's `collect_link_tt` data and evaluates all top-K candidates against them — this is what produces the empirical-regret / Nash-gap metrics reported during training.

### Training entry points

- **`tamarl/rl/train_bandit.py`** — the primary trainer, for the one-shot bandit formulations (`agent`/`od_pair`/`centralized`). Config resolution order: `_build_parser()` defaults (`None`) → JSON `--config` file (`tamarl/data/configs/*.json`, see `default_config.json` for the full schema) → explicit CLI flags (highest priority; see `_CLI_TO_KWARGS` for the CLI-arg → `train()` kwarg mapping). Supports rendering (`render_helper.py`), W&B logging (`wandb_logger.py`), memory profiling, and sanity-check plot generation.
- **`scripts/train_marl.py`** — standalone IPPO/MAPPO trainer over `POMDPWrapper`, with its own minimal from-scratch PPO implementation (`PPOBase`/`IPPOAgent`/`MAPPOAgent`); not routed through `train_bandit.py`'s config system.

### RL agents (`tamarl/rl/agents/`)

Bandit-style agents (`random`, `epsilon_greedy`, `ucb`, `thompson_sampling`, `exp3`, `msa`, `aon`/Frank-Wolfe, `evo_swap`, `replicator_dynamics`) implementing a common `get_actions_batched(obs, masks, aggregation_indices)` + `update(actions, rewards, aggregation_indices, valid_mask)` interface, vectorized over all decision-makers at once (no per-agent Python loop). `valid_mask` distinguishes legs that actually departed within `max_steps` from ones that never got a chance to (multi-leg agents whose earlier leg didn't finish in time) — agents must handle masked-out entries without corrupting per-OD/per-block statistics.

### Scenario data (`tamarl/data/scenarios/`)

MATSim-style `network.xml` + `population.xml`(`plans.xml`) pairs, parsed by `tamarl/envs/scenario_loader.py` into the tensors `TorchDNL` needs. `population_filter` selects among multiple population files in a scenario folder by substring/token match (e.g. `"100"` for a 100-agent variant); `--save_pickle` caches the parsed `ScenarioData` to disk per scenario+filter combination. Includes synthetic grid-world benchmarks (`grid_world/{3x3,8x8,...,128x128}`, with generator scripts) alongside real/derived networks (Sioux Falls, Berlin, Los Angeles, Ingolstadt, Braess, toy corridors).

## Notes

- Code in `tamarl/envs/*_wrapper.py` mixes French and English comments/docstrings — this is existing style, not an error.
- Prefer CSR-style (`*_flat` + `*_offsets`) tensor representations over dense padded tensors when touching paths/routes code — this is a deliberate, repeated pattern for memory efficiency on large scenarios, not an accident to "simplify away".
- `.script/`, `scratch/`, `wandb/`, `profiler_logs/`, `profile.stats` are local/generated artifacts, not part of the maintained codebase.
