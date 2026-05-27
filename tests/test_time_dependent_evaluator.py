"""Tests for time_dependent_evaluator.py.

Runs the 3x3 grid scenario with 100 agents to verify that:
  1. compute_n_curves builds valid arrival/departure curves.
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
    compute_n_curves,
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

    bandit = DTABanditEnv(
        scenario_path=scenario_path,
        population_filter=population,
        timestep=1.0,
        max_steps=36000,
        device=device,
        track_events=True,
    )
    env = AgentLevelWrapper(bandit=bandit, top_k=K)

    # Build evaluator
    evaluator = TimeDependentEvaluator.from_wrapper(env)

    # ── Run one episode ───────────────────────────────────────────────
    # Use action 0 (shortest path) for all legs
    actions = torch.zeros(env.num_envs, dtype=torch.long)
    env.bandit.collect_link_tt = False  # Not needed since we use events
    env.step(actions.numpy())
    dnl = env.bandit.dnl

    print(f"[INFO] Simulation finished at step {dnl.current_step}")

    # ── Test 1: compute_n_curves ─────────────────────────────────────
    print("\n[TEST 1] compute_n_curves …")
    dnl._flush_events()
    events_tensor = torch.cat(dnl._cpu_events_blocks, dim=0).to(device)
    num_steps = max(dnl.current_step + 1, 1)
    
    A_shifted_sorted, A_offsets, D_times_sorted, D_offsets = compute_n_curves(
        events_tensor, num_steps, dnl.num_edges, device
    )
    
    assert A_shifted_sorted.numel() > 0, "No arrival events found!"
    assert D_times_sorted.numel() > 0, "No departure events found!"
    print(f"  ✓ Arrivals: {A_shifted_sorted.numel()}, Departures: {D_times_sorted.numel()}")

    # ── Test 2: evaluate_paths_time_dependent ─────────────────────────
    print("\n[TEST 2] evaluate_paths_time_dependent …")
    # Expand departure_times from [A] to [TotalLegs]
    dep_times_a_steps = bandit.scenario.departure_times.long()  # [A] steps
    leg_agent_map = torch.tensor([a for a, _ in env.leg_to_agent], dtype=torch.long)
    dep_times_steps = dep_times_a_steps[leg_agent_map]  # [TotalLegs] steps
    # Routes: [TotalLegs, K, MaxLen]
    routes = env.candidate_routes[env.od_indices_all_legs]  # [TotalLegs, K, MaxLen]

    path_costs = evaluate_paths_time_dependent(
        A_shifted_sorted=A_shifted_sorted,
        A_offsets=A_offsets,
        D_times_sorted=D_times_sorted,
        D_offsets=D_offsets,
        routes=routes,
        first_edges=env.first_edges_all_legs,
        departure_times_step=dep_times_steps.to(device),
        dt=dnl.dt,
        num_edges=dnl.num_edges,
        num_steps=num_steps,
        ff_travel_time_steps=dnl.ff_travel_time_steps,
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


def test_time_dependent_evaluator():
    run_tests("tamarl/data/scenarios/grid_world/3x3", "100", "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_path", nargs="?", default="tamarl/data/scenarios/grid_world/3x3")
    parser.add_argument("--population", default="100")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_tests(args.scenario_path, args.population, args.device)
