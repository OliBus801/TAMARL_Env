"""Sequential ablation-sweep harness for the "Additional Results and Ablations"
appendix section.

Runs ``tamarl.rl.train_bandit.train()`` directly (no subprocess overhead) across
a grid of configs x seeds, ONE RUN AT A TIME — this sandbox OOMs on concurrent
heavy runs, so parallelism is intentionally not offered here. Writes two CSVs
per axis under ``--output_dir``:

  - ``<axis>_final.csv``   one row per (config, seed): final-window stats +
                            H_global (population-weighted Shannon entropy of
                            realized path-choice shares, same definition as
                            the main paper's Section "Routing Performance and
                            Network Entropy").
  - ``<axis>_curves.csv``  one row per (config, seed, episode): the full
                            learning curve, for axes where convergence
                            trajectories matter (MSA step-size / horizon).

Usage:
    python scripts/run_ablation_sweep.py \\
        --scenario tamarl/data/scenarios/ingolstadt \\
        --axis A_k_sweep --episodes 100 --seeds 0 1 2 \\
        --output_dir tamarl/data/scenarios/ingolstadt/ablation_results
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from tamarl.rl.train_bandit import run_episode, train

# ---------------------------------------------------------------------------
# Ablation axis definitions: axis_name -> list of (label, agent_type, overrides)
# ---------------------------------------------------------------------------

AXES: dict[str, list[tuple[str, str, dict]]] = {
    # A2: candidate-route action-set size, at the paper's default method.
    "A_k_sweep": [
        (f"k{k}_{agent}", agent, {"top_k_paths": k})
        for k in (3, 5, 9, 15)
        for agent in ("ucb", "msa")
    ],
    # A3: candidate-route generation method, at the paper's default k=9.
    "A_method_sweep": [
        (f"{method}_{agent}", agent, {"top_k_paths": 9, "path_method": method})
        for method in ("penalty", "yen")
        for agent in ("ucb", "msa")
    ],
    # B1: MSA step-size schedule sensitivity.
    "B_msa_alpha_sweep": [
        ("default", "msa", {"msa_alpha_max": 1.0, "msa_alpha_min": 0.05, "msa_alpha_decay": 0.01}),
        ("slow_decay", "msa", {"msa_alpha_max": 1.0, "msa_alpha_min": 0.05, "msa_alpha_decay": 0.002}),
        ("fast_decay", "msa", {"msa_alpha_max": 1.0, "msa_alpha_min": 0.05, "msa_alpha_decay": 0.05}),
        ("near_constant_avg", "msa", {"msa_alpha_max": 0.2, "msa_alpha_min": 0.2, "msa_alpha_decay": 0.0}),
    ],
    # B3: TD-oracle time-bin granularity (shared by MSA's update target and
    # the Wardrop Regret metric).
    "B_link_tt_interval_sweep": [
        (f"bin{b}_{agent}", agent, {"link_tt_interval": float(b)})
        for b in (60, 150, 300, 600, 900)
        for agent in ("ucb", "msa")
    ],
    # C: bandit hyperparameter sensitivity.
    "C_ucb_c_sweep": [
        (f"c{c}", "ucb", {"ucb_c": float(c)}) for c in (1, 10, 100, 500)
    ],
    "C_ts_prior_std_sweep": [
        (f"std{s}", "ts", {"ts_prior_std": float(s)}) for s in (100, 1000, 10000, 100000)
    ],
    "C_exp3_grid": [
        (f"eta{eta}_gamma{gamma}", "exp3", {"exp3_eta": eta, "exp3_gamma": gamma})
        for eta in (0.001, 0.1)
        for gamma in (0.01, 0.2)
    ],
    "C_rd_beta_sweep": [
        (f"beta{b}", "rd", {"rd_beta": b}) for b in (0.01, 0.1, 0.5, 1.0, 5.0)
    ],
    "C_epsilon_decay_sweep": [
        (f"decay{d}", "epsilon_greedy", {"epsilon_decay": d}) for d in (0.90, 0.95, 0.995, 0.999)
    ],
}

# Axes for which the full per-episode learning curve is worth keeping
# (convergence-trajectory questions), not just the final-window snapshot.
CURVE_AXES = {"B_msa_alpha_sweep", "B_msa_horizon"}


def compute_h_global(env, infos) -> float:
    """Population-weighted Shannon entropy of realized path-choice shares.

    Same definition as the main paper's H_global: per-OD-pair entropy of the
    empirical action distribution, averaged across OD pairs weighted by the
    number of departed legs.
    """
    ep = infos.get("_episode", {})
    act = ep.get("act")
    orig_idx = infos.get("original_leg_idx")
    if act is None or orig_idx is None or act.size == 0:
        return float("nan")

    od_idx_all = env.od_indices_all_legs.cpu().numpy()
    od_idx = od_idx_all[orig_idx]
    k = env.K
    num_od = int(od_idx_all.max()) + 1

    counts = np.zeros((num_od, k), dtype=np.float64)
    np.add.at(counts, (od_idx, act), 1.0)
    totals = counts.sum(axis=1)
    active = totals > 0
    if not active.any():
        return float("nan")

    p = counts[active] / totals[active, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return float(np.average(ent, weights=totals[active]))


def run_one(base_kwargs: dict, overrides: dict, seed: int, window: int) -> tuple[dict, list[dict]]:
    """Run one (config, seed) training call and return (final_row, curve_rows)."""
    kwargs = {**base_kwargs, **overrides, "seed": seed, "return_env": True}
    t0 = time.time()
    all_stats, env, agent = train(**kwargs)
    wall = time.time() - t0

    tail = all_stats[-window:]
    row = {"seed": seed, "wall_time_s": wall, "n_episodes": len(all_stats)}
    for key in (
        "mean_travel_time",
        "mean_regret",
        "max_regret",
        "epsilon_compliance_rate",
        "arrival_rate",
    ):
        vals = [s[key] for s in tail if key in s]
        if vals:
            row[f"{key}_final"] = float(np.mean(vals))

    # One extra evaluation episode (agent keeps acting/learning as normal) to
    # get a realized action distribution for H_global.
    _, eval_infos = run_episode(env, agent, return_raw_infos=True)
    row["h_global"] = compute_h_global(env, eval_infos)

    curve_rows = []
    for ep_idx, s in enumerate(all_stats, start=1):
        curve_rows.append(
            {
                "seed": seed,
                "episode": ep_idx,
                "mean_travel_time": s.get("mean_travel_time"),
                "mean_regret": s.get("mean_regret"),
                "epsilon_compliance_rate": s.get("epsilon_compliance_rate"),
            }
        )
    return row, curve_rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--population", default=None)
    parser.add_argument("--axis", required=True, choices=list(AXES.keys()) + ["B_msa_horizon"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--horizon_episodes",
        type=int,
        default=300,
        help="Episode count for the B_msa_horizon convergence check.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max_steps", type=int, default=86400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=int, default=10, help="Final-window size for aggregation.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    base_kwargs = dict(
        scenario_path=args.scenario,
        population_filter=args.population,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        top_k_paths=9,
        formulation="agent",
    )

    if args.axis == "B_msa_horizon":
        configs = [
            (f"ep{args.episodes}", "msa", {"n_episodes": args.episodes}),
            (f"ep{args.horizon_episodes}", "msa", {"n_episodes": args.horizon_episodes}),
        ]
    else:
        configs = AXES[args.axis]

    final_rows: list[dict] = []
    curve_rows: list[dict] = []
    total = len(configs) * len(args.seeds)
    done = 0

    for label, agent_type, overrides in configs:
        for seed in args.seeds:
            done += 1
            print(f"[{done}/{total}] axis={args.axis} label={label} agent={agent_type} seed={seed}")
            row, curves = run_one(
                base_kwargs, {**overrides, "agent_type": agent_type}, seed, args.window
            )
            row.update({"axis": args.axis, "label": label, "agent_type": agent_type})
            for k, v in overrides.items():
                row.setdefault(k, v)
            final_rows.append(row)

            if args.axis in CURVE_AXES or args.axis == "B_msa_horizon":
                for c in curves:
                    c.update({"axis": args.axis, "label": label, "agent_type": agent_type})
                curve_rows.extend(curves)

    write_csv(os.path.join(args.output_dir, f"{args.axis}_final.csv"), final_rows)
    if curve_rows:
        write_csv(os.path.join(args.output_dir, f"{args.axis}_curves.csv"), curve_rows)

    print(f"\nDone. Wrote {len(final_rows)} final rows" + (f" and {len(curve_rows)} curve rows" if curve_rows else "") + f" to {args.output_dir}")


if __name__ == "__main__":
    main()
