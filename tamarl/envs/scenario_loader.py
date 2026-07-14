"""Scenario loader for traffic simulation XML network and population files.

Parses network and population XML files into tensors ready for the TorchDNL engine
and the RL environment wrappers.
"""

import os
import pickle
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ScenarioData:
    """All tensors needed to instantiate TorchDNL in RL mode."""

    edge_static: torch.Tensor  # [E, 5]
    edge_endpoints: torch.Tensor  # [E, 2]
    node_coords: torch.Tensor  # [N, 2]
    departure_times: torch.Tensor  # [A]
    first_edges: torch.Tensor  # [A, MaxLegs]
    destinations: torch.Tensor  # [A, MaxLegs] destination node indices
    act_end_times: torch.Tensor  # [A, MaxActs] absolute end time (-1 if missing)
    act_durations: torch.Tensor  # [A, MaxActs] duration (-1 if missing)
    num_legs: torch.Tensor  # [A] total number of legs per agent
    num_nodes: int
    num_edges: int
    num_agents: int
    node_id_to_idx: dict[str, int]
    link_id_to_idx: dict[str, int]


def parse_network(network_file: str, scale_factor: float = 1.0, timestep: float = 1.0):
    """Parse a traffic network XML file.

    Returns:
        node_id_to_idx: mapping from node string ID to integer index
        edges: list of edge dicts with keys (u, v, id, attr)
        link_id_to_idx: mapping from link string ID to integer index
    """
    node_id_to_idx = {}
    edges = []
    link_id_to_idx = {}
    node_coords = []
    valid_links = 0
    eff_cell_size = 7.5

    context = ET.iterparse(network_file, events=("end",))
    for event, elem in context:
        if elem.tag == "node":
            nid = elem.get("id")
            node_id_to_idx[nid] = len(node_id_to_idx)
            x = float(elem.get("x", 0.0))
            y = float(elem.get("y", 0.0))
            node_coords.append([x, y])
            elem.clear()
        elif elem.tag == "link":
            modes = elem.get("modes")
            if "car" in modes:
                u_id = elem.get("from")
                v_id = elem.get("to")
                link_id = elem.get("id")
                if u_id in node_id_to_idx and v_id in node_id_to_idx:
                    u = node_id_to_idx[u_id]
                    v = node_id_to_idx[v_id]
                    length = float(elem.get("length"))
                    freespeed = float(elem.get("freespeed"))
                    capacity_h = float(elem.get("capacity"))
                    lanes = float(elem.get("permlanes"))

                    unscaledFlowCapacity_s = capacity_h / 3600
                    D_e = unscaledFlowCapacity_s * timestep * scale_factor
                    c_e = ((length * lanes) / eff_cell_size) * scale_factor
                    c_e = max(c_e, D_e)
                    ff_time = length / freespeed
                    temp_spaceCap = ff_time * unscaledFlowCapacity_s * scale_factor
                    c_e = max(c_e, temp_spaceCap)

                    attr = [length, freespeed, c_e, D_e, ff_time]
                    edges.append({"u": u, "v": v, "id": link_id, "attr": attr})
                    link_id_to_idx[link_id] = valid_links
                    valid_links += 1
            elem.clear()

    return node_id_to_idx, edges, link_id_to_idx, node_coords


def parse_population(pop_file: str, link_id_to_idx: dict[str, int], node_id_to_idx: dict[str, int]):
    """Parse a traffic population XML file for RL mode.

    Returns:
        agents: list of dicts per person with keys:
            - dep_time (int)
            - legs (list of dicts: {'first_edge': int, 'dest_link_id': str})
            - act_end_times (list of int, per intermediate activity)
            - act_durations (list of int, per intermediate activity)
    """
    agents = []

    def time_to_sec(t_str):
        h, m, s = map(int, t_str.split(":"))
        return h * 3600 + m * 60 + s

    context = ET.iterparse(pop_file, events=("end",))
    for event, elem in context:
        if elem.tag == "person":
            selected_plan = None
            for child in elem:
                if child.tag == "plan":
                    if child.get("selected") == "yes":
                        selected_plan = child
                        break
                    if selected_plan is None:
                        selected_plan = child

            if selected_plan is not None:
                elements = list(selected_plan)

                first_dep_time = 0
                person_legs = []
                act_end_times = []
                act_durations = []

                first_act = True

                for el in elements:
                    if el.tag in ["act", "activity"]:
                        end_time_str = el.get("end_time")
                        duration_str = el.get("duration")

                        act_end = -1
                        if end_time_str:
                            act_end = time_to_sec(end_time_str)

                        act_dur = -1
                        if duration_str:
                            act_dur = time_to_sec(duration_str)

                        if first_act:
                            if act_end >= 0:
                                first_dep_time = act_end
                            elif act_dur >= 0:
                                first_dep_time = act_dur
                            first_act = False
                        else:
                            act_end_times.append(act_end)
                            act_durations.append(act_dur)

                    elif el.tag == "leg":
                        mode = el.get("mode")
                        if mode == "car":
                            route_tag = el.find("route")
                            route_str = (
                                route_tag.text.strip()
                                if (route_tag is not None and route_tag.text)
                                else None
                            )
                            if route_str:
                                link_ids = route_str.split(" ")
                                first_link_id = link_ids[0]
                                last_link_id = link_ids[-1]

                                if (
                                    first_link_id in link_id_to_idx
                                    and last_link_id in link_id_to_idx
                                ):
                                    person_legs.append(
                                        {
                                            "first_edge": link_id_to_idx[first_link_id],
                                            "dest_link_id": last_link_id,
                                        }
                                    )

                if len(person_legs) > 0:
                    num_boundaries = len(person_legs) - 1
                    act_end_times = act_end_times[:num_boundaries]
                    act_durations = act_durations[:num_boundaries]

                    agents.append(
                        {
                            "dep_time": first_dep_time,
                            "legs": person_legs,
                            "act_end_times": act_end_times,
                            "act_durations": act_durations,
                        }
                    )

            elem.clear()

    return agents


def load_scenario(
    root_folder: str,
    population_filter: str | None = None,
    timestep: float = 1.0,
    scale_factor: float = 1.0,
    save_pickle: bool = False,
) -> ScenarioData:
    """Load a traffic scenario (network + population) for RL mode.

    Args:
        root_folder: directory containing network.xml and population.xml
        population_filter: substring to match population file (e.g. '100')
        timestep: simulation timestep in seconds
        scale_factor: scale factor for network capacities

    Returns:
        ScenarioData with all tensors ready for TorchDNL RL mode
    """
    cache_name = "scenario_data"
    if population_filter:
        cache_name += f"_{population_filter}"
    cache_name += ".pkl"
    pickle_path = os.path.join(root_folder, cache_name)

    # Check if pickle exists
    if os.path.exists(pickle_path):
        print(f"Loading scenario data from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    # Locate files
    files = [f for f in os.listdir(root_folder) if f.endswith(".xml")]
    network_file = None
    population_file = None

    net_candidates = [f for f in files if "network" in f.lower()]
    pop_candidates = [f for f in files if "population" in f.lower() or "plans" in f.lower()]

    if net_candidates:
        network_file = os.path.join(root_folder, net_candidates[0])
    if pop_candidates:
        if population_filter:
            # Token-based matching first
            filtered = []
            for p in pop_candidates:
                tokens = p.replace("-", "_").replace(".", "_").split("_")
                if population_filter in tokens:
                    filtered.append(p)
            if filtered:
                pop_candidates = filtered
            else:
                pop_candidates = [p for p in pop_candidates if population_filter in p]
        if pop_candidates:
            # Prioritize 'routed' files if available
            routed_candidates = [p for p in pop_candidates if "routed" in p.lower()]
            if routed_candidates:
                population_file = os.path.join(root_folder, routed_candidates[0])
            else:
                population_file = os.path.join(root_folder, pop_candidates[0])

    if not network_file:
        raise FileNotFoundError(f"No network file found in {root_folder}")
    if not population_file:
        raise FileNotFoundError(f"No population file found in {root_folder}")

    # Parse
    node_id_to_idx, edges_data, link_id_to_idx, node_coords_list = parse_network(
        network_file, scale_factor, timestep
    )
    agents = parse_population(population_file, link_id_to_idx, node_id_to_idx)

    if len(agents) == 0:
        raise ValueError(f"No valid trips found in population file {population_file}.")

    # Build edge info
    # edges_data[i]['v'] is the to_node index
    edge_to_node = {edges_data[i]["id"]: edges_data[i]["v"] for i in range(len(edges_data))}

    # Build tensors
    edge_static = torch.tensor([e["attr"] for e in edges_data], dtype=torch.float32)
    edge_endpoints = torch.tensor([[e["u"], e["v"]] for e in edges_data], dtype=torch.int32)
    node_coords = torch.tensor(node_coords_list, dtype=torch.float32)

    departure_times = torch.tensor([a["dep_time"] for a in agents], dtype=torch.int32)

    num_agents = len(agents)
    max_legs = 0
    max_acts = 0
    for a in agents:
        max_legs = max(max_legs, len(a["legs"]))
        max_acts = max(max_acts, len(a["act_end_times"]))

    first_edges = torch.full((num_agents, max_legs), -1, dtype=torch.long)
    destinations = torch.full((num_agents, max_legs), -1, dtype=torch.long)
    act_end_times = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    act_durations = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    num_legs = torch.zeros(num_agents, dtype=torch.int32)

    for i, a in enumerate(agents):
        legs = a["legs"]
        num_legs[i] = len(legs)
        for j, leg in enumerate(legs):
            first_edges[i, j] = leg["first_edge"]
            destinations[i, j] = edge_to_node[leg["dest_link_id"]]

        n_acts = len(a["act_end_times"])
        if n_acts > 0:
            act_end_times[i, :n_acts] = torch.tensor(a["act_end_times"], dtype=torch.int32)
            act_durations[i, :n_acts] = torch.tensor(a["act_durations"], dtype=torch.int32)

    num_nodes = len(node_id_to_idx)
    num_edges = len(edges_data)

    scenario_data = ScenarioData(
        edge_static=edge_static,
        edge_endpoints=edge_endpoints,
        node_coords=node_coords,
        departure_times=departure_times,
        first_edges=first_edges,
        destinations=destinations,
        act_end_times=act_end_times,
        act_durations=act_durations,
        num_legs=num_legs,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_agents=num_agents,
        node_id_to_idx=node_id_to_idx,
        link_id_to_idx=link_id_to_idx,
    )

    if save_pickle:
        print(f"Saving scenario data to {pickle_path}...")
        with open(pickle_path, "wb") as f:
            pickle.dump(scenario_data, f)

    return scenario_data
