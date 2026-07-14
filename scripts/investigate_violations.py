import argparse
import os
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def format_time(seconds):
    if pd.isna(seconds) or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_network(network_file):
    print(f"Loading network from {network_file}...")
    sys.path.append(os.path.abspath("."))
    try:
        from tamarl.envs.scenario_loader import parse_network
    except ImportError:
        print(
            "Error: Could not import tamarl.envs.scenario_loader. Please run from the TrafficGym root."
        )
        sys.exit(1)

    node_id_to_idx, edges, link_id_to_idx = parse_network(network_file, 1.0, 1.0)
    return node_id_to_idx, edges, link_id_to_idx


def main():
    parser = argparse.ArgumentParser(
        description="Investigate Free-Flow violations in sanity check CSV using Top-K paths."
    )
    parser.add_argument("csv_path", type=str, help="Path to the sanity_01_tt_vs_fftt.csv file")
    parser.add_argument("--network", type=str, default=None, help="Path to the network XML file")
    parser.add_argument(
        "--population", type=str, default=None, help="Path to the population XML file"
    )
    parser.add_argument("--paths", type=str, default=None, help="Path to the top-K paths .pkl file")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        sys.exit(1)

    print(f"Loading data from {args.csv_path}...\n")
    df = pd.read_csv(args.csv_path)

    finite_mask = np.isfinite(df["fftt_sec"]) & np.isfinite(df["realized_tt_sec"])
    valid_df = df[finite_mask].copy()

    valid_df["violation_sec"] = valid_df["fftt_sec"] - valid_df["realized_tt_sec"]
    violations_df = valid_df[valid_df["violation_sec"] > 1e-4].copy()

    total_violations = len(violations_df)
    if total_violations == 0:
        print("🎉 Great news! No free-flow travel time violations found.")
        sys.exit(0)

    print(f"⚠ Found {total_violations} legs with Free-Flow violations.")

    worst_row = violations_df.loc[violations_df["violation_sec"].idxmax()]

    if "leg_idx" in worst_row:
        worst_idx = int(worst_row["leg_idx"])
    elif "agent_idx" in worst_row:
        worst_idx = int(worst_row["agent_idx"])
    else:
        worst_idx = int(worst_row.name)

    worst_fftt = worst_row["fftt_sec"]
    worst_realized = worst_row["realized_tt_sec"]
    worst_violation = worst_row["violation_sec"]

    planned_dep = worst_row.get("planned_departure", -1)
    actual_dep = worst_row.get("actual_departure", -1)
    arrival = worst_row.get("arrival", -1)
    chosen_path_idx = worst_row.get("chosen_path_idx", -1)

    print("\n--- WORST OFFENDER SUMMARY ---")
    print(f"Global Leg Index    : {worst_idx}")
    print(f"Calculated FFTT     : {worst_fftt:.2f} seconds")
    print(f"Realized Travel Time: {worst_realized:.2f} seconds")
    print(f"Violation Amount    : {worst_violation:.2f} seconds (arrived too early)")
    print(f"Planned Departure   : {format_time(planned_dep)}")
    print(f"Actual Departure    : {format_time(actual_dep)}")
    print(f"Arrival             : {format_time(arrival)}")
    if not pd.isna(chosen_path_idx) and chosen_path_idx >= 0:
        print(f"Chosen Path Index   : {int(chosen_path_idx)}")
    print("------------------------------")

    # Infer paths
    target_dir = os.path.dirname(os.path.abspath(args.csv_path))
    parts = target_dir.split(os.sep)
    try:
        scen_idx = parts.index("scenarios")
        scenario_path = os.sep.join(parts[: scen_idx + 2])
    except ValueError:
        scenario_path = target_dir

    network_file = args.network
    pop_file = args.population
    paths_file = args.paths

    if not network_file or not pop_file or not paths_file:
        files = os.listdir(scenario_path) if os.path.exists(scenario_path) else []
        if not network_file:
            nets = [f for f in files if "network" in f.lower() and f.endswith(".xml")]
            network_file = os.path.join(scenario_path, nets[0]) if nets else None
        if not pop_file:
            pops = [
                f
                for f in files
                if ("population" in f.lower() or "plans" in f.lower()) and f.endswith(".xml")
            ]
            routed = [p for p in pops if "routed" in p.lower()]
            pop_file = (
                os.path.join(scenario_path, routed[0] if routed else pops[0]) if pops else None
            )
        if not paths_file:
            pkls = [f for f in files if f.endswith(".pkl")]
            paths_file = os.path.join(scenario_path, pkls[0]) if pkls else None

    if not network_file or not os.path.exists(network_file):
        print("\nError: Could not locate network XML file.")
        sys.exit(1)
    if not pop_file or not os.path.exists(pop_file):
        print("\nError: Could not locate population XML file.")
        sys.exit(1)
    if not paths_file or not os.path.exists(paths_file):
        print("\nError: Could not locate top-K paths PKL file.")
        sys.exit(1)

    node_id_to_idx, edges, link_id_to_idx = load_network(network_file)

    print(f"\nTracing Global Leg Index {worst_idx} in {pop_file}...")
    context = ET.iterparse(pop_file, events=("end",))

    current_leg_idx = 0
    target_orig_node = None
    target_dest_node = None

    for event, elem in context:
        if elem.tag == "person":
            selected_plan = None
            for child in elem:
                if child.tag == "plan" and child.get("selected") == "yes":
                    selected_plan = child
                    break
                elif child.tag == "plan" and selected_plan is None:
                    selected_plan = child

            if selected_plan is not None:
                for el in selected_plan:
                    if el.tag == "leg" and el.get("mode") == "car":
                        route_tag = el.find("route")
                        if route_tag is not None and route_tag.text:
                            route_str = route_tag.text.strip()
                            link_ids = route_str.split(" ")
                            first_link_id = link_ids[0]
                            last_link_id = link_ids[-1]

                            if first_link_id in link_id_to_idx and last_link_id in link_id_to_idx:
                                if current_leg_idx == worst_idx:
                                    target_orig_node = edges[link_id_to_idx[first_link_id]]["v"]
                                    target_dest_node = edges[link_id_to_idx[last_link_id]]["v"]
                                    break
                                current_leg_idx += 1

            if target_orig_node is not None:
                break
            elem.clear()

    if target_orig_node is None:
        print(f"\n⚠ Global Leg Index {worst_idx} could not be found in the population file.")
        sys.exit(0)

    print(f"\nOrigin Node Index: {target_orig_node} | Destination Node Index: {target_dest_node}")
    print(f"Loading Top-K Paths from {paths_file}...")

    with open(paths_file, "rb") as fp:
        top_k_data = pickle.load(fp)

    od_key = (target_orig_node, target_dest_node)
    if od_key not in top_k_data:
        print(f"⚠ OD Pair {od_key} not found in Top-K paths dictionary!")
        sys.exit(0)

    paths = top_k_data[od_key]

    if not pd.isna(chosen_path_idx) and chosen_path_idx >= 0:
        chosen_path_idx = int(chosen_path_idx)
        print(f"\n--- CHOSEN PATH DETAILS (Index {chosen_path_idx}) ---")
        if chosen_path_idx < len(paths):
            path_list = [paths[chosen_path_idx]]
            offset = chosen_path_idx
        else:
            print(
                f"⚠ Chosen path index {chosen_path_idx} is out of bounds for top-{len(paths)} paths!"
            )
            path_list = paths
            offset = 0
    else:
        print(f"\n--- TOP-{len(paths)} PATHS FOR OD PAIR ---")
        path_list = paths
        offset = 0

    for k, path in enumerate(path_list):
        real_idx = k + offset
        # path is a list of edge indices
        fftt_total = sum(edges[e]["attr"][4] for e in path)
        # Verify connectivity
        connected = True
        for i in range(len(path) - 1):
            u_edge = edges[path[i]]
            v_edge = edges[path[i + 1]]
            if u_edge["v"] != v_edge["u"]:
                connected = False
                break

        conn_emoji = "✅ Valid" if connected else "❌ Disconnected!"

        print(f"\nPath {real_idx} (Length: {len(path)} edges) [{conn_emoji}]")
        print(f"  Full FFTT      : {fftt_total:.2f} seconds")

        # Print path details
        edge_ids = [edges[e]["id"] for e in path]
        print(f"  Edge Sequence  : {' -> '.join(edge_ids)}")


if __name__ == "__main__":
    main()
