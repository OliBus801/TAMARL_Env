"""Render the single cross-scenario DNL-fidelity summary figure (3 fidelity
signals x 3 scenarios) for the AAAI supplementary material, from the per-
scenario `.npz` caches written by `scripts/make_validation_figures.py
--cache ...`.

Usage:
    python scripts/make_combined_validation_figure.py \\
        --cache ingolstadt=/tmp/validation_cache_ingolstadt.npz \\
        --cache "grid_world/32x32=/tmp/validation_cache_grid32x32.npz" \\
        --cache "sioux_falls=/tmp/validation_figures_cache.npz" \\
        --output tamarl/data/scenarios/combined_dnl_validation_grid.pdf

Scenarios are plotted in the order given on the command line — pass them
smallest-to-largest so the grid reads left-to-right as a scale progression.
"""

from __future__ import annotations

import argparse

import numpy as np

from tamarl.visualisation import validation_figures as vf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", action="append", required=True,
        help="label=path.npz, repeatable, in left-to-right column order.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    usetex_ok = vf.setup_style()
    print(f"matplotlib text.usetex: {'enabled' if usetex_ok else 'disabled (STIX fallback)'}")

    scenarios = []
    for spec in args.cache:
        label, path = spec.split("=", 1)
        cached = np.load(path)
        n_trips = int(len(cached["matsim_trip_tt"]) / len(cached["seeds"].tolist()))
        scenarios.append(
            {
                "label": label,
                "n_trips": n_trips,
                "bins": cached["bins"],
                "matsim_en_route": cached["matsim_en_route"],
                "torchdnl_en_route": cached["torchdnl_en_route"],
                "matsim_link_vol": cached["matsim_link_vol"],
                "torchdnl_link_vol": cached["torchdnl_link_vol"],
                "matsim_trip_tt": cached["matsim_trip_tt"],
                "torchdnl_trip_tt": cached["torchdnl_trip_tt"],
            }
        )
        print(f"  loaded {label} from {path}: n_trips/seed={n_trips}")

    vf.plot_combined_scenario_grid(scenarios, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
