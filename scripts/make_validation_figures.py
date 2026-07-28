"""Render AAAI-ready vector-PDF figures for the TorchDNL-vs-MATSim DNL-fidelity
validation appendix (Sioux Falls, fixed MATSim routes replayed through both
simulators across matched seeds — see scripts/validate_against_matsim.py for
the full experiment description and the summary.md this figure set visualizes).

This script reads the *same* per-seed raw data as validate_against_matsim.py
and reuses its metric helpers (tamarl.validation.geh / .distributional /
.matsim_io) — it does not recompute or redefine any metric, only presentation.
It is deliberately kept separate from validate_against_matsim.py so that
script's existing summary.md-generation behavior is untouched.

Usage:
    python scripts/make_validation_figures.py \\
        --network /path/to/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_network_PT.xml \\
        --matsim-root tamarl/../matsim_project/output/siouxfalls-2014/dnl_validation \\
        --torchdnl-root tamarl/data/scenarios/sioux_falls/dnl_validation \\
        --output-dir tamarl/data/scenarios/sioux_falls/dnl_validation/report/figures_aaai
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from tamarl.envs.scenario_loader import parse_network
from tamarl.validation import distributional, geh as geh_mod
from tamarl.validation.matsim_io import EVT_LEFT_LINK, parse_matsim_events, parse_torchdnl_events_csv
from tamarl.visualisation import validation_figures as vf
from tamarl.visualisation.plot_histogram import _compute_leg_histogram_data

BUCKET_SEC_DEFAULT = 900


def _find_common_seeds(matsim_root: str, torchdnl_root: str) -> list[int]:
    matsim_seeds = {
        int(os.path.basename(p).removeprefix("seed_"))
        for p in glob.glob(os.path.join(matsim_root, "seed_*"))
        if os.path.exists(os.path.join(p, "output_events.xml.gz"))
    }
    torchdnl_seeds = {
        int(os.path.basename(p).removeprefix("seed_"))
        for p in glob.glob(os.path.join(torchdnl_root, "seed_*"))
        if os.path.exists(os.path.join(p, "events.csv"))
    }
    common = sorted(matsim_seeds & torchdnl_seeds)
    if not common:
        raise FileNotFoundError(
            f"No seeds with both a MATSim and a TorchDNL run found under "
            f"{matsim_root} / {torchdnl_root}."
        )
    return common


def _en_route_series(events: list[tuple], max_steps: int, bucket_sec: int) -> dict[str, np.ndarray]:
    bins, dep, arr, stuck, en_route = _compute_leg_histogram_data(
        events, max_steps, dt=1.0, bucket_size_sec=bucket_sec
    )
    return {"bins": bins, "departures": dep, "arrivals": arr, "stuck": stuck, "en_route": en_route}


def _link_volumes(events: list[tuple], num_links: int) -> np.ndarray:
    counts = np.zeros(num_links, dtype=np.int64)
    for _, evt_type, _agent, edge_idx in events:
        if evt_type == EVT_LEFT_LINK and 0 <= edge_idx < num_links:
            counts[edge_idx] += 1
    return counts


def _matsim_trip_travel_times_sec(matsim_seed_dir: str) -> np.ndarray:
    trips_path = os.path.join(matsim_seed_dir, "output_trips.csv.gz")
    df = pd.read_csv(trips_path, sep=";")

    def hms_to_sec(s: str) -> float:
        h, m, sec = map(int, s.split(":"))
        return h * 3600 + m * 60 + sec

    return df["trav_time"].map(hms_to_sec).to_numpy(dtype=np.float64)


def _torchdnl_trip_travel_times_sec(torchdnl_seed_dir: str) -> np.ndarray:
    leg_tt = torch.load(os.path.join(torchdnl_seed_dir, "leg_travel_times.pt"))
    leg_tt = leg_tt.numpy()
    valid = leg_tt > 0
    return leg_tt[valid]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--matsim-root", required=True, help="Directory containing seed_N/output_events.xml.gz etc.")
    parser.add_argument("--torchdnl-root", required=True, help="Directory containing seed_N/events.csv etc.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--bucket-sec", type=int, default=BUCKET_SEC_DEFAULT)
    parser.add_argument(
        "--cache", default=None,
        help="Optional .npz path to cache parsed per-seed arrays (XML/CSV parsing is the "
        "slow step; reuse the cache across figure-styling iterations with --use-cache).",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Load arrays from --cache instead of re-parsing, if the cache file exists.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    usetex_ok = vf.setup_style()
    print(f"matplotlib text.usetex: {'enabled' if usetex_ok else 'disabled (STIX fallback)'}")

    if args.use_cache and args.cache and os.path.exists(args.cache):
        print(f"Loading cached arrays from {args.cache}")
        cached = np.load(args.cache)
        seeds = cached["seeds"].tolist()
        bins = cached["bins"]
        matsim_en_route = cached["matsim_en_route"]
        torchdnl_en_route = cached["torchdnl_en_route"]
        matsim_dep = cached["matsim_dep"]
        torchdnl_dep = cached["torchdnl_dep"]
        matsim_arr = cached["matsim_arr"]
        torchdnl_arr = cached["torchdnl_arr"]
        matsim_link_vol = cached["matsim_link_vol"]
        torchdnl_link_vol = cached["torchdnl_link_vol"]
        matsim_trip_tt = cached["matsim_trip_tt"]
        torchdnl_trip_tt = cached["torchdnl_trip_tt"]
    else:
        _, _, link_id_to_idx, _ = parse_network(args.network)
        num_links = len(link_id_to_idx)

        seeds = _find_common_seeds(args.matsim_root, args.torchdnl_root)
        print(f"Comparing {len(seeds)} seeds: {seeds}")

        matsim_en_route, torchdnl_en_route = [], []
        matsim_dep, torchdnl_dep = [], []
        matsim_arr, torchdnl_arr = [], []
        matsim_link_vol, torchdnl_link_vol = [], []
        matsim_trip_tt, torchdnl_trip_tt = [], []
        bins = None

        for seed in seeds:
            matsim_seed_dir = os.path.join(args.matsim_root, f"seed_{seed}")
            torchdnl_seed_dir = os.path.join(args.torchdnl_root, f"seed_{seed}")

            matsim_events, _ = parse_matsim_events(
                os.path.join(matsim_seed_dir, "output_events.xml.gz"), link_id_to_idx
            )
            torchdnl_events = parse_torchdnl_events_csv(
                os.path.join(torchdnl_seed_dir, "events.csv"), link_id_to_idx
            )

            m_series = _en_route_series(matsim_events, args.max_steps, args.bucket_sec)
            t_series = _en_route_series(torchdnl_events, args.max_steps, args.bucket_sec)
            bins = m_series["bins"]

            matsim_en_route.append(m_series["en_route"])
            torchdnl_en_route.append(t_series["en_route"])
            matsim_dep.append(m_series["departures"])
            torchdnl_dep.append(t_series["departures"])
            matsim_arr.append(m_series["arrivals"])
            torchdnl_arr.append(t_series["arrivals"])

            matsim_link_vol.append(_link_volumes(matsim_events, num_links))
            torchdnl_link_vol.append(_link_volumes(torchdnl_events, num_links))

            matsim_trip_tt.append(_matsim_trip_travel_times_sec(matsim_seed_dir))
            torchdnl_trip_tt.append(_torchdnl_trip_travel_times_sec(torchdnl_seed_dir))

            print(f"  seed {seed}: parsed ({len(matsim_events)} MATSim events, {len(torchdnl_events)} TorchDNL events)")

        matsim_en_route = np.stack(matsim_en_route)
        torchdnl_en_route = np.stack(torchdnl_en_route)
        matsim_dep = np.stack(matsim_dep)
        torchdnl_dep = np.stack(torchdnl_dep)
        matsim_arr = np.stack(matsim_arr)
        torchdnl_arr = np.stack(torchdnl_arr)
        matsim_link_vol = np.stack(matsim_link_vol)
        torchdnl_link_vol = np.stack(torchdnl_link_vol)
        matsim_trip_tt = np.concatenate(matsim_trip_tt)
        torchdnl_trip_tt = np.concatenate(torchdnl_trip_tt)

        if args.cache:
            np.savez(
                args.cache, seeds=np.array(seeds), bins=bins,
                matsim_en_route=matsim_en_route, torchdnl_en_route=torchdnl_en_route,
                matsim_dep=matsim_dep, torchdnl_dep=torchdnl_dep,
                matsim_arr=matsim_arr, torchdnl_arr=torchdnl_arr,
                matsim_link_vol=matsim_link_vol, torchdnl_link_vol=torchdnl_link_vol,
                matsim_trip_tt=matsim_trip_tt, torchdnl_trip_tt=torchdnl_trip_tt,
            )
            print(f"Cached parsed arrays to {args.cache}")

    # ---- GEH per series (raw per-observation values, for the ECDF figure) ----
    geh_values_by_series = {}
    for name, model_seeds, ref_seeds in [
        ("en_route", torchdnl_en_route, matsim_en_route),
        ("departures", torchdnl_dep, matsim_dep),
        ("arrivals", torchdnl_arr, matsim_arr),
    ]:
        geh_values_by_series[name] = geh_mod.geh_series_from_counts(model_seeds, ref_seeds).ravel()
    geh_values_by_series["link_volume"] = geh_mod.geh_series_from_counts(
        torchdnl_link_vol, matsim_link_vol
    ).ravel()

    # ---- distributional stats for the ECDF figure's annotation ----
    tt_wasserstein = distributional.pooled_wasserstein(torchdnl_trip_tt, matsim_trip_tt)
    tt_ks_stat, tt_ks_pvalue = distributional.ks_test(torchdnl_trip_tt, matsim_trip_tt)

    # ---- Figures ----
    fig1_path = os.path.join(args.output_dir, "en_route_envelope.pdf")
    vf.plot_en_route_envelope(bins, matsim_en_route, torchdnl_en_route, fig1_path)
    print(f"Wrote {fig1_path}")

    fig2_path = os.path.join(args.output_dir, "travel_time_ecdf.pdf")
    vf.plot_travel_time_ecdf(
        matsim_trip_tt, torchdnl_trip_tt, tt_wasserstein, tt_ks_stat, tt_ks_pvalue, fig2_path
    )
    print(f"Wrote {fig2_path}")

    fig3_path = os.path.join(args.output_dir, "link_volume_parity.pdf")
    vf.plot_link_volume_parity(matsim_link_vol, torchdnl_link_vol, fig3_path)
    print(f"Wrote {fig3_path}")

    fig4_path = os.path.join(args.output_dir, "geh_distribution.pdf")
    vf.plot_geh_distribution(geh_values_by_series, fig4_path)
    print(f"Wrote {fig4_path}")


if __name__ == "__main__":
    main()
