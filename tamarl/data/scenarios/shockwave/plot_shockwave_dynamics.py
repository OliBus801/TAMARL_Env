import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Analyze shockwave queue dynamics and travel time hysteresis from MATSim events."
    )
    parser.add_argument(
        "--events_file", type=str, required=True, help="Path to the events.csv file."
    )
    parser.add_argument(
        "--link_id",
        type=str,
        required=True,
        help="Link ID of the bottleneck/target to analyze (e.g. L2).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Temporal granularity in seconds (default: 1.0).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output path for the plot PNG file."
    )

    args = parser.parse_args()

    if not os.path.exists(args.events_file):
        print(f"Error: Events file not found at '{args.events_file}'")
        sys.exit(1)

    print(f"Reading events from: {args.events_file}")
    df = pd.read_csv(args.events_file)

    # 2. Extract departures from the target link
    df_link = df[df["link"] == args.link_id]
    df_left = df_link[df_link["type"] == "left_link"]

    if len(df_left) > 0:
        df_departures = df_left.copy()
        dep_type_str = "left_link"
    else:
        df_departures = df_link[df_link["type"].isin({"leaves_traffic", "arrival"})].copy()
        # Prevent double counting if both leaves_traffic and arrival are present
        df_departures = df_departures.drop_duplicates(subset=["person"], keep="first")
        dep_type_str = "leaves_traffic/arrival"

    if len(df_departures) == 0:
        print(f"Error: No exit events found on link '{args.link_id}'.")
        print("Available links in events:", df["link"].dropna().unique())
        sys.exit(1)

    target_agents = set(df_departures["person"])

    # 1. Extract arrivals (enters_traffic events) ONLY for agents that belong to the target flux
    df_arrivals = df[(df["type"] == "enters_traffic") & (df["person"].isin(target_agents))].copy()

    if len(df_arrivals) == 0:
        print("Error: No entry events ('enters_traffic') found for the target agents.")
        sys.exit(1)

    df_arrivals = df_arrivals.sort_values(by="time")
    df_departures = df_departures.sort_values(by="time")

    arrival_times = df_arrivals["time"].values
    departure_times = df_departures["time"].values

    print(f"Target link: {args.link_id} | Departure event type used: {dep_type_str}")

    # 3. Calculate dynamic accumulation x(t) of the specific flux
    A_at_arrival = np.searchsorted(arrival_times, arrival_times, side="right")
    D_at_arrival = np.searchsorted(departure_times, arrival_times, side="right")

    df_arrivals["acc_at_entry"] = A_at_arrival - D_at_arrival

    # 4. Match entry (enters_traffic) and exit times per agent
    arrivals_by_person = {
        person: group.to_dict("records") for person, group in df_arrivals.groupby("person")
    }
    departures_by_person = {
        person: group.to_dict("records") for person, group in df_departures.groupby("person")
    }

    matched_trips = []
    for person, arrivals in arrivals_by_person.items():
        departures = departures_by_person.get(person, [])
        # Ensure chronological ordering per agent
        sorted_arrivals = sorted(arrivals, key=lambda x: x["time"])
        sorted_departures = sorted(departures, key=lambda x: x["time"])

        for arr_row, dep_row in zip(sorted_arrivals, sorted_departures):
            matched_trips.append(
                {
                    "person": person,
                    "time_in": arr_row["time"],
                    "time_out": dep_row["time"],
                    "travel_time": dep_row["time"] - arr_row["time"],
                    "acc_at_entry": arr_row["acc_at_entry"],
                }
            )

    trips_df = pd.DataFrame(matched_trips)

    if len(trips_df) == 0:
        print("Error: Could not match any entries and exits for travel time calculation.")
        sys.exit(1)

    print(f"Processed {len(df_arrivals)} arrivals, {len(df_departures)} departures.")
    print(f"Successfully matched {len(trips_df)} completed trips.")

    # Sort trips chronologically by entry time
    trips_df = trips_df.sort_values(by="time_in")

    # Set up output filename
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.events_file), f"shockwave_dynamics_{args.link_id}.png"
        )

    # --- PLOTTING ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Left Plot: Cumulative Flow (Newell's Curves)
    min_t = 0.0
    # Add a dynamic 15% padding to the maximum time for better visibility of queue dissipation
    max_t = df["time"].max()
    max_t = max_t * 1.15

    time_grid = np.arange(min_t, max_t + args.timestep, args.timestep)
    A_grid = np.searchsorted(arrival_times, time_grid, side="right")
    D_grid = np.searchsorted(departure_times, time_grid, side="right")

    # Convert time grid to minutes for easier reading
    time_grid_min = time_grid / 60.0

    # Plot curves
    ax1.plot(
        time_grid_min, A_grid, color="#0e668b", linewidth=2.5, label="Arrivals (Enters Traffic)"
    )
    ax1.plot(
        time_grid_min,
        D_grid,
        color="#d65a31",
        linewidth=2.5,
        label=f"Departures (Exits {args.link_id})",
    )

    # Shade area between curves (representing queue accumulation)
    ax1.fill_between(
        time_grid_min, A_grid, D_grid, color="#0e668b", alpha=0.15, label="Accumulation (Queue)"
    )

    ax1.set_title(
        f"Cumulative Flow Curves (Newell) - Link {args.link_id}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Simulation Time (min)", fontsize=12, labelpad=10)
    ax1.set_ylabel("Cumulative Vehicle Count", fontsize=12, labelpad=10)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right Plot: Travel Time vs Accumulation (Hysteresis Loop)
    scatter = ax2.scatter(
        trips_df["acc_at_entry"],
        trips_df["travel_time"],
        c=trips_df["time_in"] / 60.0,
        cmap="plasma",
        s=20,
        alpha=0.8,
        edgecolors="none",
        zorder=3,
    )

    # Connect points in chronological order to show loop direction
    ax2.plot(
        trips_df["acc_at_entry"],
        trips_df["travel_time"],
        color="#7f8c8d",
        linewidth=0.8,
        alpha=0.6,
        zorder=2,
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Vehicle Entry Time (min)", fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    ax2.set_title(
        f"Travel Time vs Accumulation (Hysteresis) - Link {args.link_id}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Accumulation at Entry $x(t_{in})$ (veh)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Experienced Travel Time $tt$ (s)", fontsize=12, labelpad=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved successfully to: {args.output}")


if __name__ == "__main__":
    main()
