"""Replay a converged/fixed MATSim route population directly through TorchDNL,
across N random seeds, for DNL-fidelity validation against MATSim.

This deliberately bypasses DTABanditEnv/the bandit route-choice wrappers: routes
come verbatim from a MATSim population's `<route>` tags (see
tamarl.validation.matsim_io.parse_fixed_route_population), not from TorchDNL's own
top-K candidate search. Only TorchDNL's queue-model/DNL stochasticity should vary
across seeds — this isolates the DNL engine from route-choice-algorithm
differences (see the "Validating TorchDNL against MATSim and SUMO" appendix plan).

Usage:
    python scripts/run_torchdnl_fixed_routes.py \\
        --network /path/to/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_network_PT.xml \\
        --population /path/to/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_route_population.xml \\
        --seeds 24 25 26 27 28 \\
        --output-dir tamarl/data/scenarios/sioux_falls/dnl_validation
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from tamarl.core.torchdnl import EVENT_TYPE_NAMES, TorchDNL
from tamarl.envs.scenario_loader import parse_network
from tamarl.validation.matsim_io import parse_fixed_route_population


def build_edge_tensors(edges: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror scenario_loader.load_scenario's edge tensor construction."""
    edge_static = torch.tensor([e["attr"] for e in edges], dtype=torch.float32)
    edge_endpoints = torch.tensor([[e["u"], e["v"]] for e in edges], dtype=torch.int32)
    return edge_static, edge_endpoints


def write_events_csv(events: list[tuple], idx_to_link_id: dict[int, str], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "type", "person", "link", "extra"])
        for time_val, evt_type_int, agent_id, edge_id in events:
            evt_name = EVENT_TYPE_NAMES.get(int(evt_type_int), f"unknown_{int(evt_type_int)}")
            link_str = idx_to_link_id.get(int(edge_id), str(int(edge_id)))
            writer.writerow([time_val, evt_name, f"agent_{int(agent_id)}", link_str, ""])


def run_one_seed(
    seed: int,
    edge_static: torch.Tensor,
    edge_endpoints: torch.Tensor,
    departure_times: torch.Tensor,
    act_end_times: torch.Tensor,
    act_durations: torch.Tensor,
    num_legs: torch.Tensor,
    paths_flat: torch.Tensor,
    path_offsets: torch.Tensor,
    max_steps: int,
    device: str,
    link_tt_interval: float,
) -> tuple[list[tuple], torch.Tensor, dict, torch.Tensor]:
    dnl = TorchDNL(
        edge_static=edge_static,
        paths=None,
        device=device,
        departure_times=departure_times,
        edge_endpoints=edge_endpoints,
        act_end_times=act_end_times,
        act_durations=act_durations,
        num_legs=num_legs,
        dt=1.0,
        seed=seed,
        track_events=True,
        collect_link_tt=True,
        link_tt_interval=link_tt_interval,
        paths_flat=paths_flat,
        path_offsets=path_offsets,
        max_steps=max_steps,
    )

    while dnl.current_step < max_steps:
        if dnl.active_agents_count == 0:
            break
        dnl.step()

    dnl.finalize_stuck_agents()

    events = dnl.get_events()
    dynamic_tt = dnl.get_dynamic_link_travel_times()
    metrics = dnl.get_metrics()
    leg_travel_times = dnl.leg_metrics[:, :, 1].clone().cpu()  # [A, MaxLegs]
    return events, dynamic_tt, metrics, leg_travel_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, help="MATSim network XML (same file used for the MATSim reference run).")
    parser.add_argument("--population", required=True, help="MATSim population XML with pre-assigned <route> per leg (fixed/converged).")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100000, help="Simulation horizon in seconds (dt=1s); Sioux Falls full-day runs need >86400.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--link-tt-interval", type=float, default=300.0, help="Link travel-time aggregation bin size in seconds (default 300s = 5min; use e.g. 900 for 15min bins).")
    args = parser.parse_args()

    node_id_to_idx, edges, link_id_to_idx, _ = parse_network(args.network)
    idx_to_link_id = {v: k for k, v in link_id_to_idx.items()}
    print(f"Network: {len(node_id_to_idx)} nodes, {len(link_id_to_idx)} car links")

    fr = parse_fixed_route_population(args.population, link_id_to_idx)
    num_agents = len(fr.departure_times)
    print(f"Fixed-route population: {num_agents} agents, {fr.paths_flat.shape[0]} total path edges")

    edge_static, edge_endpoints = build_edge_tensors(edges)
    departure_times = torch.tensor(fr.departure_times, dtype=torch.int32)
    act_end_times = torch.tensor(fr.act_end_times, dtype=torch.int32)
    act_durations = torch.tensor(fr.act_durations, dtype=torch.int32)
    num_legs = torch.tensor(fr.num_legs, dtype=torch.int32)
    paths_flat = torch.tensor(fr.paths_flat, dtype=torch.int32)
    path_offsets = torch.tensor(fr.path_offsets, dtype=torch.long)

    os.makedirs(args.output_dir, exist_ok=True)

    for seed in args.seeds:
        t0 = time.time()
        events, dynamic_tt, metrics, leg_travel_times = run_one_seed(
            seed,
            edge_static,
            edge_endpoints,
            departure_times,
            act_end_times,
            act_durations,
            num_legs,
            paths_flat,
            path_offsets,
            args.max_steps,
            args.device,
            args.link_tt_interval,
        )
        elapsed = time.time() - t0

        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        write_events_csv(events, idx_to_link_id, os.path.join(seed_dir, "events.csv"))
        if dynamic_tt is not None:
            torch.save(dynamic_tt, os.path.join(seed_dir, "dynamic_link_tt.pt"))
        torch.save(leg_travel_times, os.path.join(seed_dir, "leg_travel_times.pt"))

        print(
            f"[seed {seed}] {elapsed:.1f}s, {len(events)} events, "
            f"arrived={metrics['arrived_count']}, en_route={metrics['en_route_count']}, "
            f"avg_tt={metrics['avg_travel_time']:.1f}s, n_imputed={metrics['n_imputed_legs']}"
        )


if __name__ == "__main__":
    main()
