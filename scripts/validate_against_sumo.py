"""Compare TorchDNL against SUMO's mesoscopic model (forced via --mesosim) on
the same fixed Sioux Falls routes used for the MATSim comparison.

Framed differently from validate_against_matsim.py: SUMO's mesoscopic model is
a different (though queue-like) traffic flow theory than MATSim/TorchDNL's, and
getting the MATSim network to import into SUMO at all required non-trivial
fixes (see convert_fixed_routes_to_sumo.py / netconvert connection-file
workaround for two separate netconvert MATSim-import bugs). Treat this as a
qualitative-to-moderate plausibility check, not as tight a claim as the MATSim
comparison -- and note the known SUMO-side "waiting" vs "loaded" bookkeeping
gap (a scheduled departure that hasn't been inserted onto a congested origin
edge yet has no MATSim/TorchDNL equivalent) directly in the report so it isn't
silently conflated with the departures-per-bin GEH computation.

Usage:
    python scripts/validate_against_sumo.py \\
        --sumo-root /path/to/matsim_project/scenarios/siouxfalls-2014/sumo \\
        --sumo-seed-prefix seed \\
        --torchdnl-root tamarl/data/scenarios/sioux_falls/dnl_validation \\
        --network /path/to/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_network_PT.xml \\
        --output-dir tamarl/data/scenarios/sioux_falls/dnl_validation/report_sumo
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from tamarl.envs.scenario_loader import parse_network
from tamarl.validation import distributional, geh as geh_mod
from tamarl.validation.matsim_io import parse_torchdnl_events_csv
from tamarl.validation.sumo_io import parse_summary, parse_tripinfo, rebin_summary
from tamarl.visualisation.plot_histogram import _compute_leg_histogram_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumo-root", required=True, help="Directory containing summary_seedN.xml / tripinfo_seedN.xml")
    parser.add_argument("--sumo-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--torchdnl-root", required=True)
    parser.add_argument("--torchdnl-seeds", type=int, nargs="+", required=True, help="TorchDNL seeds to pool for comparison (need not match SUMO seed numbers 1:1).")
    parser.add_argument("--network", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--bucket-sec", type=int, default=900)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _, _, link_id_to_idx, _ = parse_network(args.network)

    # ---- TorchDNL side (reuse events.csv from the MATSim comparison run) ----
    torchdnl_en_route, torchdnl_dep, torchdnl_arr, torchdnl_trip_tt = [], [], [], []
    for seed in args.torchdnl_seeds:
        events = parse_torchdnl_events_csv(
            os.path.join(args.torchdnl_root, f"seed_{seed}", "events.csv"), link_id_to_idx
        )
        bins, dep, arr, stuck, en_route = _compute_leg_histogram_data(
            events, args.max_steps, dt=1.0, bucket_size_sec=args.bucket_sec
        )
        torchdnl_en_route.append(en_route)
        torchdnl_dep.append(dep)
        torchdnl_arr.append(arr)
        leg_tt = torch.load(os.path.join(args.torchdnl_root, f"seed_{seed}", "leg_travel_times.pt")).numpy()
        torchdnl_trip_tt.append(leg_tt[leg_tt > 0])

    torchdnl_en_route = np.stack(torchdnl_en_route)
    torchdnl_dep = np.stack(torchdnl_dep)
    torchdnl_arr = np.stack(torchdnl_arr)
    torchdnl_trip_tt = np.concatenate(torchdnl_trip_tt)

    # ---- SUMO side ----
    sumo_en_route, sumo_loaded, sumo_inserted, sumo_arr, sumo_waiting = [], [], [], [], []
    sumo_trip_tt = []
    for seed in args.sumo_seeds:
        summary_df = parse_summary(os.path.join(args.sumo_root, f"summary_seed{seed}.xml"))
        rebinned = rebin_summary(summary_df, args.max_steps, args.bucket_sec)
        sumo_en_route.append(rebinned["en_route"])
        sumo_loaded.append(rebinned["loaded_per_bin"])
        sumo_inserted.append(rebinned["inserted_per_bin"])
        sumo_arr.append(rebinned["arrivals_per_bin"])
        sumo_waiting.append(rebinned["waiting"])

        trip_df = parse_tripinfo(os.path.join(args.sumo_root, f"tripinfo_seed{seed}.xml"))
        sumo_trip_tt.append(trip_df["duration"].to_numpy())

    sumo_en_route = np.stack(sumo_en_route)
    sumo_loaded = np.stack(sumo_loaded)
    sumo_inserted = np.stack(sumo_inserted)
    sumo_arr = np.stack(sumo_arr)
    sumo_waiting = np.stack(sumo_waiting)
    sumo_trip_tt = np.concatenate(sumo_trip_tt)
    n_bins = min(torchdnl_en_route.shape[1], sumo_en_route.shape[1])
    torchdnl_en_route, sumo_en_route = torchdnl_en_route[:, :n_bins], sumo_en_route[:, :n_bins]
    torchdnl_dep, sumo_loaded = torchdnl_dep[:, :n_bins], sumo_loaded[:, :n_bins]
    torchdnl_arr, sumo_arr = torchdnl_arr[:, :n_bins], sumo_arr[:, :n_bins]

    # ---- GEH (loaded/scheduled departures = fair comparison to MATSim/TorchDNL's departure event) ----
    geh_summaries = {
        "en_route": geh_mod.summarize(geh_mod.geh_series_from_counts(sumo_en_route, torchdnl_en_route)),
        "scheduled_departures": geh_mod.summarize(geh_mod.geh_series_from_counts(sumo_loaded, torchdnl_dep)),
        "arrivals": geh_mod.summarize(geh_mod.geh_series_from_counts(sumo_arr, torchdnl_arr)),
    }

    tt_wasserstein = distributional.pooled_wasserstein(sumo_trip_tt, torchdnl_trip_tt)
    tt_ks_stat, tt_ks_pvalue = distributional.ks_test(sumo_trip_tt, torchdnl_trip_tt)

    total_trips = torchdnl_trip_tt.shape[0] / len(args.torchdnl_seeds)
    sumo_arrival_fraction = sumo_trip_tt.shape[0] / len(args.sumo_seeds) / total_trips
    mean_waiting_fraction = float(np.mean(sumo_waiting) / np.mean(sumo_loaded.cumsum(axis=1)[:, -1]))

    report_path = os.path.join(args.output_dir, "summary.md")
    with open(report_path, "w") as f:
        f.write("# TorchDNL vs SUMO (mesoscopic) — plausibility check\n\n")
        f.write(
            "SUMO's mesoscopic model is a structurally different traffic flow model "
            "than MATSim/TorchDNL's queue model -- treat this as a plausibility check, "
            "not a tight quantitative validation like the MATSim comparison.\n\n"
        )
        f.write(f"SUMO seeds: {args.sumo_seeds} (fewer than the MATSim/TorchDNL N=10 -- "
                f"mesoscopic SUMO runs are far slower per seed than MATSim or TorchDNL here).\n\n")
        f.write("## GEH statistic\n\n")
        f.write("| series | n | mean | median | max | % GEH<5 | % GEH<10 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, s in geh_summaries.items():
            f.write(
                f"| {name} | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | "
                f"{s['max']:.2f} | {s['pct_geh_lt_5']:.1%} | {s['pct_geh_lt_10']:.1%} |\n"
            )
        f.write("\n## Known bookkeeping gap\n\n")
        f.write(
            f"- Fraction of scheduled departures still `waiting` (not yet physically "
            f"inserted due to origin-edge congestion), averaged over the day: "
            f"**{mean_waiting_fraction:.1%}**. These vehicles have \"departed\" in the "
            f"MATSim/TorchDNL sense but aren't yet `running` in SUMO -- a modeling "
            f"difference in how departure congestion is represented, not necessarily an error.\n"
        )
        f.write(
            f"- Fraction of trips that completed within the {args.max_steps}s horizon in SUMO: "
            f"**{sumo_arrival_fraction:.1%}** (TorchDNL/MATSim: 100%).\n"
        )
        f.write("\n## Pooled trip travel-time distribution (completed trips only)\n\n")
        f.write(f"- Wasserstein distance: **{tt_wasserstein:.2f}s**\n")
        f.write(f"- KS statistic: {tt_ks_stat:.4f} (p={tt_ks_pvalue:.3g})\n")
    print(f"Wrote {report_path}")

    x = np.arange(n_bins) * args.bucket_sec / 3600.0
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step(x, np.median(torchdnl_en_route, axis=0), where="post", color="tab:orange", linewidth=2, label="TorchDNL median")
    for i in range(sumo_en_route.shape[0]):
        ax.step(x, sumo_en_route[i], where="post", color="tab:green", alpha=0.6, linewidth=1,
                label="SUMO meso (per seed)" if i == 0 else None)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Agents en route / running")
    ax.set_title("En-route agents per bin: TorchDNL vs SUMO mesoscopic")
    ax.legend()
    fig_path = os.path.join(args.output_dir, "en_route_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
