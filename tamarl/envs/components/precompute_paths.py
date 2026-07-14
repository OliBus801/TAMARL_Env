"""Script to pre-compute top-k paths for a given scenario.

Usage:
    PYTHONPATH=. python tamarl/envs/components/precompute_paths.py \\
        --scenario tamarl/data/scenarios/grid_world/3x3 \\
        --k 9 \\
        --method sidetrack

Methods:
    yen        Exact k-shortest loopless paths (igraph). Slow on large networks.
    sidetrack  Exact k-shortest loopless paths (rustworkx/Rust). Recommended for
               large networks (Berlin, LA).
    penalty    Approximate spatially-diverse paths via penalized Dijkstra.
               Fastest option; ideal for MARL/DTA where diversity matters.
"""

import argparse
import os

import numpy as np
import torch

from tamarl.envs.components.path_enumerator import get_or_compute_top_k_paths
from tamarl.envs.scenario_loader import load_scenario


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute top-k paths for a scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario", type=str, required=True, help="Path to the scenario directory"
    )
    parser.add_argument(
        "--population", type=str, default=None, help="Population filter (e.g. '100' or '1pct')"
    )
    parser.add_argument(
        "--k", type=int, default=9, help="Number of paths to compute per OD pair (default: 9)"
    )
    parser.add_argument(
        "--timestep", type=float, default=1.0, help="Timestep duration in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sidetrack",
        choices=["yen", "sidetrack", "penalty"],
        help=(
            "Algorithm used to compute top-k paths. "
            "'yen' = exact via igraph (slow on large nets); "
            "'sidetrack' = exact via rustworkx/Rust (recommended for large nets); "
            "'penalty' = approximate spatially-diverse paths (fastest, best for MARL). "
            "Default: sidetrack."
        ),
    )
    parser.add_argument(
        "--penalty_factor",
        type=float,
        default=1.5,
        help=(
            "Multiplicative penalty applied to edges of each found path "
            "before the next Dijkstra iteration (only used with --method penalty). "
            "Higher values force more spatial diversity. Default: 1.5."
        ),
    )
    args = parser.parse_args()

    print(f"Loading scenario from: {args.scenario}")
    scenario = load_scenario(args.scenario, population_filter=args.population)

    edge_eps = scenario.edge_endpoints.numpy()

    # Collect all unique OD pairs from the population
    print("Extracting unique OD pairs...")
    leg_origins = []
    leg_dests = []

    A = scenario.num_legs.shape[0]  # num_agents
    num_legs_np = scenario.num_legs.numpy()
    fe_np = scenario.first_edges.numpy()
    dest_np = scenario.destinations.numpy()

    for i in range(A):
        for leg in range(num_legs_np[i]):
            fe = int(fe_np[i, leg])
            dest = int(dest_np[i, leg])
            if fe >= 0:
                orig = int(edge_eps[fe, 1])
                leg_origins.append(orig)
                leg_dests.append(dest)

    unique_od = np.unique(np.stack([leg_origins, leg_dests], axis=1), axis=0)
    print(f"Found {len(unique_od)} unique OD pairs.")
    print(
        f"Algorithm: {args.method}"
        + (f"  |  penalty_factor: {args.penalty_factor}" if args.method == "penalty" else "")
    )

    ff_times = torch.floor(scenario.edge_static[:, 4] / args.timestep).numpy().astype(np.float64)

    # Call get_or_compute_top_k_paths which handles caching automatically
    _ = get_or_compute_top_k_paths(
        scenario_dir=args.scenario,
        num_nodes=scenario.num_nodes,
        edge_endpoints=edge_eps,
        ff_times=ff_times,
        od_pairs=unique_od.astype(np.int32),
        k=args.k,
        method=args.method,
        penalty_factor=args.penalty_factor,
    )

    print("Done!")


if __name__ == "__main__":
    main()
