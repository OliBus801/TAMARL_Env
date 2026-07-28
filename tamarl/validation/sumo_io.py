"""Parse SUMO's own native output formats (summary-output, tripinfo-output)
for the TorchDNL/MATSim-vs-SUMO comparison.

Unlike the MATSim side, SUMO already produces exactly the aggregate series we
need directly -- no event-log reconstruction required:
  - `--summary-output`: per-second (loaded, inserted, running, waiting,
    ended, arrived, teleports, ...) counts, written by the simulation itself.
  - `--tripinfo-output`: one row per completed vehicle with `duration`
    (travel time) and `depart`/`arrival` directly.

One important semantic gap to keep in mind when comparing to MATSim/TorchDNL:
SUMO's "inserted" only increments once a vehicle physically enters the
network, which can lag its scheduled departure time if the origin edge is
congested (a `waiting` vehicle has "departed" in the MATSim/TorchDNL sense --
its activity ended -- but is not yet `inserted`/`running` in SUMO's
bookkeeping). Comparing "departures per bin" 1:1 against MATSim can therefore
be misleading under heavy congestion; report `loaded` (scheduled, matches
MATSim's departure event) alongside `inserted`/`running` so this distinction
is visible rather than silently conflated.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def parse_summary(summary_path: str) -> pd.DataFrame:
    """Parse a SUMO --summary-output XML into a per-second DataFrame."""
    rows = []
    for _, elem in ET.iterparse(summary_path, events=("end",)):
        if elem.tag == "step":
            rows.append({k: float(v) for k, v in elem.attrib.items()})
            elem.clear()
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


def parse_tripinfo(tripinfo_path: str) -> pd.DataFrame:
    """Parse a SUMO --tripinfo-output XML into a per-trip DataFrame."""
    rows = []
    for _, elem in ET.iterparse(tripinfo_path, events=("end",)):
        if elem.tag == "tripinfo":
            rows.append(dict(elem.attrib))
            elem.clear()
    df = pd.DataFrame(rows)
    for col in ("depart", "arrival", "duration", "routeLength", "waitingTime", "timeLoss"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def rebin_summary(summary_df: pd.DataFrame, max_steps: int, bucket_sec: int) -> dict[str, np.ndarray]:
    """Resample SUMO's per-second summary into fixed-size bins matching the
    MATSim/TorchDNL histogram convention (`_compute_leg_histogram_data`):
    per-bin departures/arrivals as the increase in the relevant cumulative
    counter, and en-route as the "running" gauge sampled at the bin's right edge.

    Returns bins, loaded (cumulative "scheduled" departures, matches MATSim's
    departure event regardless of SUMO insertion delay), inserted (cumulative,
    only actually-on-network departures), arrivals (cumulative "arrived"),
    en_route ("running" instantaneous), waiting (instantaneous, vehicles
    whose activity ended but haven't been inserted onto a congested origin
    edge yet -- has no MATSim/TorchDNL equivalent).
    """
    bins = np.arange(0, max_steps + bucket_sec, bucket_sec)
    time = summary_df["time"].to_numpy()

    def sample_at_bin_edges(col: str) -> np.ndarray:
        idx = np.searchsorted(time, bins, side="right") - 1
        idx = np.clip(idx, 0, len(summary_df) - 1)
        return summary_df[col].to_numpy()[idx]

    loaded_cum = sample_at_bin_edges("loaded")
    inserted_cum = sample_at_bin_edges("inserted")
    arrived_cum = sample_at_bin_edges("arrived")
    running = sample_at_bin_edges("running")
    waiting = sample_at_bin_edges("waiting")

    return {
        "bins": bins,
        "loaded_per_bin": np.diff(loaded_cum, prepend=0.0),
        "inserted_per_bin": np.diff(inserted_cum, prepend=0.0),
        "arrivals_per_bin": np.diff(arrived_cum, prepend=0.0),
        "en_route": running[1:] if len(running) == len(bins) else running,
        "waiting": waiting[1:] if len(waiting) == len(bins) else waiting,
    }
