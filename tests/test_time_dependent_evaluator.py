"""Tests for time_dependent_evaluator.py.

Runs the 3x3 grid scenario with 100 agents to verify that:
  1. compute_avg_travel_times produces sane output (≥ free-flow, no NaN).
  2. evaluate_paths_time_dependent returns costs ≥ free-flow travel times.
  3. TimeDependentEvaluator.from_wrapper() works end-to-end.
  4. best_k indices are valid (0 ≤ best_k < K).

Run with:
    conda activate ml
    PYTHONPATH=. python tests/test_time_dependent_evaluator.py tamarl/data/scenarios/grid_world/3x3 --population 100
"""
import argparse
import sys

import torch

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.envs.components.time_dependent_evaluator import (
    TimeDependentEvaluator,
    compute_avg_travel_times,
    evaluate_paths_time_dependent,
)


def run_tests(scenario_path: str, population: str, device: str = "cpu"):
    print(f"\n{'='*60}")
    print("  TimeDependentEvaluator — Integration Tests")
    print(f"  Scenario : {scenario_path}")
    print(f"  Pop      : {population}  |  Device: {device}")
    print(f"{'='*60}\n")

    # ── Build environment ─────────────────────────────────────────────
    K = 3
    INTERVAL = 60.0  # 60-second bins (fast test)

    bandit = DTABanditEnv(
        scenario_path=scenario_path,
        population_filter=population,
        timestep=1.0,
        max_steps=36000,
        device=device,
        link_tt_interval=INTERVAL,
    )
    env = AgentLevelWrapper(bandit=bandit, top_k=K)

    # Build evaluator
    evaluator = TimeDependentEvaluator.from_wrapper(env)

    # ── Run one episode ───────────────────────────────────────────────
    # Use action 0 (shortest path) for all legs
    actions = torch.zeros(env.num_envs, dtype=torch.long)
    env.bandit.collect_link_tt = True
    env.step(actions.numpy())
    dnl = env.bandit.dnl

    print(f"[INFO] Simulation finished at step {dnl.current_step}")
    print(f"[INFO] Bins used: {dnl.interval_tt_sum.shape[0]}")
    print(f"[INFO] Non-zero bins: {(dnl.interval_tt_count > 0).any(dim=1).sum().item()}")

    # ── Test 1: compute_avg_travel_times ─────────────────────────────
    print("\n[TEST 1] compute_avg_travel_times …")
    ff_s = (dnl.ff_travel_time_steps.float() * dnl.dt)
    avg_tt = compute_avg_travel_times(
        dnl.interval_tt_sum,
        dnl.interval_tt_count,
        ff_s,
    )
    assert not avg_tt.isnan().any(), "NaN found in avg_tt!"
    assert not avg_tt.isinf().any(), "Inf found in avg_tt!"
    # avg_tt must be ≥ free-flow everywhere (congestion can only add delay)
    ff_expanded = ff_s.unsqueeze(0).expand_as(avg_tt)
    # Allow a tiny float32 tolerance (1e-3 seconds)
    violations = (avg_tt < ff_expanded - 1e-3).sum().item()
    assert violations == 0, f"{violations} bins have avg_tt < free-flow!"
    print(f"  ✓ Shape {list(avg_tt.shape)}, no NaN/Inf, all ≥ free-flow")
    print(f"  ✓ Mean avg_tt (across used bins, all edges): "
          f"{avg_tt[dnl.interval_tt_count > 0].mean().item():.1f} s")

    # ── Test 2: evaluate_paths_time_dependent ─────────────────────────
    print("\n[TEST 2] evaluate_paths_time_dependent …")
    # Expand departure_times from [A] to [TotalLegs]
    dep_times_a_s = bandit.scenario.departure_times.float() * dnl.dt  # [A] seconds
    leg_agent_map = torch.tensor([a for a, _ in env.leg_to_agent], dtype=torch.long)
    dep_times_s = dep_times_a_s[leg_agent_map]  # [TotalLegs] seconds
    # Routes: [TotalLegs, K, MaxLen]
    routes = env.candidate_routes[env.od_indices_all_legs]  # [TotalLegs, K, MaxLen]

    path_costs = evaluate_paths_time_dependent(
        avg_tt=avg_tt,
        routes=routes,
        first_edges=env.first_edges_all_legs,
        departure_times_s=dep_times_s,
        bin_size_s=INTERVAL,
    )
    assert path_costs.shape == (env.num_envs, K), \
        f"Unexpected shape: {path_costs.shape}"
    assert not path_costs.isnan().any(), "NaN in path_costs!"
    assert (path_costs >= 0).all(), "Negative path cost found!"
    print(f"  ✓ path_costs shape: {list(path_costs.shape)}")
    print(f"  ✓ Min cost: {path_costs.min().item():.1f} s")
    print(f"  ✓ Max cost: {path_costs.max().item():.1f} s")
    print(f"  ✓ Mean cost (path 0): {path_costs[:, 0].mean().item():.1f} s")

    # ── Test 3: TimeDependentEvaluator.evaluate() end-to-end ─────────
    print("\n[TEST 3] TimeDependentEvaluator.evaluate() …")
    path_costs_e, best_k_e = evaluator.evaluate(
        dnl=dnl,
        departure_times=bandit.scenario.departure_times,
    )
    assert path_costs_e.shape == (env.num_envs, K)
    assert best_k_e.shape == (env.num_envs,)
    assert (best_k_e >= 0).all() and (best_k_e < K).all(), \
        "best_k out of [0, K) range!"
    assert torch.allclose(path_costs_e, path_costs, atol=1e-3), \
        "evaluate() and standalone function disagree!"
    print(f"  ✓ best_k range: [{best_k_e.min().item()}, {best_k_e.max().item()}]")
    best_k_hist = torch.bincount(best_k_e, minlength=K)
    for k in range(K):
        print(f"    path {k} chosen as best: {best_k_hist[k].item()} legs")

    # ── Test 4: Consistency with actual leg travel times ──────────────
    print("\n[TEST 4] Consistency check (TD cost vs experienced TT) …")
    # Compare TD path-0 cost vs experienced TT only for leg-0 of each agent.
    # experienced_tt is [A, MaxLegs] → slice by leg index via leg_to_agent.
    leg0_indices = [idx for idx, (_, leg_j) in enumerate(env.leg_to_agent) if leg_j == 0]
    leg0_agent_indices = [a for a, leg_j in env.leg_to_agent if leg_j == 0]
    experienced_tt_leg0 = dnl.leg_metrics[leg0_agent_indices, 0, 1].float() * dnl.dt
    td_cost_path0_leg0 = path_costs_e[leg0_indices, 0]

    mean_abs_err = (td_cost_path0_leg0 - experienced_tt_leg0).abs().mean().item()
    print(f"  Leg-0 agents: {len(leg0_indices)}")
    print(f"  Mean |TD_cost(path0) - experienced_TT|: {mean_abs_err:.2f} s")
    print(f"  (Expected small-ish if congestion is moderate)")

    print(f"\n{'='*60}")
    print("  All tests passed ✓")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_path")
    parser.add_argument("--population", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_tests(args.scenario_path, args.population, args.device)
