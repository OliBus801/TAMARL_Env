"""Scenario loader for MATSim XML network + population files.

Extracts the XML parsing logic from benchmark_matsim_dnl.py into a reusable module
for the RL environment.
"""

import os
import xml.etree.ElementTree as ET
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ScenarioData:
    """All tensors needed to instantiate TorchDNLMATSim in RL mode."""
    edge_static: torch.Tensor        # [E, 5]
    edge_endpoints: torch.Tensor     # [E, 2]
    departure_times: torch.Tensor    # [A]
    first_edges: torch.Tensor        # [A]
    destinations: torch.Tensor       # [A] destination node indices
    num_nodes: int
    num_edges: int
    num_agents: int
    node_id_to_idx: Dict[str, int]
    link_id_to_idx: Dict[str, int]


def parse_network(network_file: str, scale_factor: float = 1.0, timestep: float = 1.0):
    """Parse a MATSim network XML file.
    
    Returns:
        node_id_to_idx: mapping from node string ID to integer index
        edges: list of edge dicts with keys (u, v, id, attr)
        link_id_to_idx: mapping from link string ID to integer index
    """
    node_id_to_idx = {}
    edges = []
    link_id_to_idx = {}
    valid_links = 0
    eff_cell_size = 7.5

    context = ET.iterparse(network_file, events=("end",))
    for event, elem in context:
        if elem.tag == "node":
            nid = elem.get('id')
            node_id_to_idx[nid] = len(node_id_to_idx)
            elem.clear()
        elif elem.tag == "link":
            modes = elem.get('modes')
            if 'car' in modes:
                u_id = elem.get('from')
                v_id = elem.get('to')
                link_id = elem.get('id')
                if u_id in node_id_to_idx and v_id in node_id_to_idx:
                    u = node_id_to_idx[u_id]
                    v = node_id_to_idx[v_id]
                    length = float(elem.get('length'))
                    freespeed = float(elem.get('freespeed'))
                    capacity_h = float(elem.get('capacity'))
                    lanes = float(elem.get('permlanes'))

                    unscaledFlowCapacity_s = capacity_h / 3600
                    D_e = unscaledFlowCapacity_s * timestep * scale_factor
                    c_e = ((length * lanes) / eff_cell_size) * scale_factor
                    c_e = max(c_e, D_e)
                    ff_time = length / freespeed
                    temp_spaceCap = ff_time * unscaledFlowCapacity_s * scale_factor
                    c_e = max(c_e, temp_spaceCap)

                    attr = [length, freespeed, c_e, D_e, ff_time]
                    edges.append({'u': u, 'v': v, 'id': link_id, 'attr': attr})
                    link_id_to_idx[link_id] = valid_links
                    valid_links += 1
            elem.clear()

    return node_id_to_idx, edges, link_id_to_idx


def parse_population(pop_file: str, link_id_to_idx: Dict[str, int], node_id_to_idx: Dict[str, int]):
    """Parse a MATSim population XML file.
    
    Returns:
        trips: list of dicts with keys (dep_time, first_edge, dest_node)
    """
    trips = []

    def time_to_sec(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s

    context = ET.iterparse(pop_file, events=("end",))
    for event, elem in context:
        if elem.tag == "person":
            selected_plan = None
            for child in elem:
                if child.tag == 'plan':
                    if child.get('selected') == 'yes':
                        selected_plan = child
                        break
                    if selected_plan is None:
                        selected_plan = child

            if selected_plan is not None:
                current_act_end_time = 0
                # Find the destination from the last activity
                dest_link_id = None
                activities = [el for el in selected_plan if el.tag in ['act', 'activity']]
                if len(activities) >= 2:
                    dest_link_id = activities[-1].get('link')

                for el in selected_plan:
                    if el.tag in ['act', 'activity']:
                        end_time_str = el.get('end_time')
                        if end_time_str:
                            current_act_end_time = time_to_sec(end_time_str)
                        else:
                            max_dur = el.get('max_dur')
                            if max_dur:
                                current_act_end_time += time_to_sec(max_dur)
                    elif el.tag == 'leg':
                        mode = el.get('mode')
                        if mode == 'car':
                            route_tag = el.find('route')
                            route_str = route_tag.text.strip() if (route_tag is not None and route_tag.text) else None
                            if route_str:
                                link_ids = route_str.split(' ')
                                # First edge
                                first_link_id = link_ids[0]
                                if first_link_id not in link_id_to_idx:
                                    continue
                                first_edge = link_id_to_idx[first_link_id]

                                # Destination node: to_node of the last link in the route
                                last_link_id = link_ids[-1]
                                if last_link_id not in link_id_to_idx:
                                    continue

                                # Find destination node from dest_link's to_node
                                # We'll resolve this to node idx later
                                trips.append({
                                    'dep_time': current_act_end_time,
                                    'first_edge': first_edge,
                                    'dest_link_id': last_link_id,
                                })

            elem.clear()

    return trips


def load_scenario(
    root_folder: str,
    population_filter: Optional[str] = None,
    timestep: float = 1.0,
    scale_factor: float = 1.0,
) -> ScenarioData:
    """Load a MATSim scenario (network + population) for RL mode.
    
    Args:
        root_folder: directory containing network.xml and population.xml
        population_filter: substring to match population file (e.g. '100')
        timestep: simulation timestep in seconds
        scale_factor: scale factor for network capacities
        
    Returns:
        ScenarioData with all tensors ready for TorchDNLMATSim RL mode
    """
    # Locate files
    files = [f for f in os.listdir(root_folder) if f.endswith('.xml')]
    network_file = None
    population_file = None
    
    net_candidates = [f for f in files if 'network' in f.lower()]
    pop_candidates = [f for f in files if 'population' in f.lower() or 'plans' in f.lower()]

    if net_candidates:
        network_file = os.path.join(root_folder, net_candidates[0])
    if pop_candidates:
        if population_filter:
            # Token-based matching first
            filtered = []
            for p in pop_candidates:
                tokens = p.replace('-', '_').replace('.', '_').split('_')
                if population_filter in tokens:
                    filtered.append(p)
            if filtered:
                pop_candidates = filtered
            else:
                pop_candidates = [p for p in pop_candidates if population_filter in p]
        if pop_candidates:
            population_file = os.path.join(root_folder, pop_candidates[0])

    if not network_file:
        raise FileNotFoundError(f"No network file found in {root_folder}")
    if not population_file:
        raise FileNotFoundError(f"No population file found in {root_folder}")

    # Parse
    node_id_to_idx, edges_data, link_id_to_idx = parse_network(network_file, scale_factor, timestep)
    trips = parse_population(population_file, link_id_to_idx, node_id_to_idx)

    if len(trips) == 0:
        raise ValueError("No valid trips found in population file.")

    # Build edge info
    # edges_data[i]['v'] is the to_node index
    edge_to_node = {edges_data[i]['id']: edges_data[i]['v'] for i in range(len(edges_data))}

    # Build tensors
    edge_static = torch.tensor([e['attr'] for e in edges_data], dtype=torch.float32)
    edge_endpoints = torch.tensor([[e['u'], e['v']] for e in edges_data], dtype=torch.int32)
    
    departure_times = torch.tensor([t['dep_time'] for t in trips], dtype=torch.int32)
    first_edges = torch.tensor([t['first_edge'] for t in trips], dtype=torch.long)
    destinations = torch.tensor(
        [edge_to_node[t['dest_link_id']] for t in trips], dtype=torch.long
    )

    num_nodes = len(node_id_to_idx)
    num_edges = len(edges_data)
    num_agents = len(trips)

    return ScenarioData(
        edge_static=edge_static,
        edge_endpoints=edge_endpoints,
        departure_times=departure_times,
        first_edges=first_edges,
        destinations=destinations,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_agents=num_agents,
        node_id_to_idx=node_id_to_idx,
        link_id_to_idx=link_id_to_idx,
    )
