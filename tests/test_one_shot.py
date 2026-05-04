"""Validation script for the one-shot bandit DTA environment.

Usage:
    PYTHONPATH=. python tests/test_one_shot.py \\
        tamarl/data/scenarios/grid_world/3x3 --population 100 --k 3 --episodes 10
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tamarl.envs.vehicle_level_wrapper import VehicleLevelWrapper
from tamarl.rl_models.random_agent import RandomAgent


def run(
    scenario_path: str,
    population_filter: str | None,
    k: int,
    episodes: int,
    device: str,
    seed: int | None,
):
    print("=" * 60)
    print("  One-Shot Bandit DTA — Validation")
    print("=" * 60)

    # ── Build environment ────────────────────────────────────────────
    t0 = time.perf_counter()
    from tamarl.envs.dta_bandit_env import DTABanditEnv
    bandit = DTABanditEnv(
        scenario_path=scenario_path,
        population_filter=population_filter,
        timestep=1.0,
        scale_factor=1.0,
        max_steps=36000,
        device=device,
        seed=seed,
    )
    env = VehicleLevelWrapper(
        bandit=bandit,
        top_k=k,
    )
    t_init = time.perf_counter() - t0

    A = env.num_envs
    info = env.get_candidate_paths_info()

    print(f"\n📦 Environment initialised in {t_init:.3f}s")
    print(f"   Agents (A) : {A}")
    print(f"   Unique ODs : {info.get('num_ods_leg0', info.get('num_ods', '?'))}")
    print(f"   K paths    : {info['K']}")
    print(f"   MaxPathLen : {info['max_path_len']}")
    print(f"   Device     : {device}")

    # ── Build agent ──────────────────────────────────────────────────
    agent = RandomAgent(num_agents=A, k=k, seed=seed)

    # ── Run episodes ─────────────────────────────────────────────────
    print(f"\n🚀 Running {episodes} episodes...")
    all_mean_tt = []
    all_times = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed)
        actions = agent.act()

        t_start = time.perf_counter()
        obs, rewards, terminated, truncated, infos = env.step(actions)
        t_elapsed = time.perf_counter() - t_start

        mean_tt = infos["mean_travel_time"]
        all_mean_tt.append(mean_tt)
        all_times.append(t_elapsed)

        print(
            f"  Episode {ep+1:3d} | "
            f"Mean TT: {mean_tt:8.1f} steps | "
            f"Mean reward: {rewards.mean():9.2f} | "
            f"Time: {t_elapsed*1000:7.1f} ms"
        )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("📊 Summary")
    print(f"   Avg Mean Travel Time : {np.mean(all_mean_tt):.1f} steps")
    print(f"   Avg Sim Time         : {np.mean(all_times)*1000:.1f} ms")
    print(f"   Total Wall Time      : {sum(all_times):.2f}s")
    
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"   Peak VRAM            : {peak_vram:.1f} MB")

    env.close()
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-Shot Bandit DTA Validation")
    parser.add_argument("root_folder", help="Scenario directory")
    parser.add_argument("--population", default=None, help="Population file filter")
    parser.add_argument("--k", type=int, default=3, help="Top-K paths per OD")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--device", default=None, help="Device (auto-detect)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run(
        scenario_path=args.root_folder,
        population_filter=args.population,
        k=args.k,
        episodes=args.episodes,
        device=device,
        seed=args.seed,
    )
