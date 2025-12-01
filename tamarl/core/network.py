"""Network utilities for traffic assignment environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class NetworkMetadata:
    """Helper container for network adjacency information."""

    out_edges_per_node: List[List[int]]
    max_out_degree: int


def build_out_edges_per_node(data: Data) -> NetworkMetadata:
    """Build outgoing edge lists per node.

    Args:
        data: Graph data with ``edge_index`` shaped [2, num_edges].

    Returns:
        Metadata containing outgoing edges per node and the maximum out degree.
    """

    num_nodes = int(data.num_nodes)
    out_edges_per_node: List[List[int]] = [[] for _ in range(num_nodes)]
    sources = data.edge_index[0].tolist()

    for edge_id, src in enumerate(sources):
        out_edges_per_node[int(src)].append(edge_id)

    max_out = max((len(edges) for edges in out_edges_per_node), default=0)
    return NetworkMetadata(out_edges_per_node=out_edges_per_node, max_out_degree=max_out)

# TODO : Make this function vectorized. It takes a vector of node_ids and action_indices and returns a vector of edge_ids.
def map_action_to_edge(
    node_id: int, action_index: int, out_edges_per_node: List[List[int]]
) -> int:
    """Map a local action index to a global edge id.

    Args:
        node_id: Current node id of the agent.
        action_index: Local outgoing-edge index selected by the agent.
        out_edges_per_node: Pre-computed outgoing edges per node.

    Returns:
        The global edge id corresponding to the chosen action.

    Raises:
        ValueError: If the action_index is invalid for the node.
    """

    edges = out_edges_per_node[node_id]
    if action_index < 0 or action_index >= len(edges):
        raise ValueError(f"Invalid action {action_index} for node {node_id}")
    return edges[action_index]


def edge_id_to_nodes(edge_id: int, edge_index: torch.Tensor) -> Tuple[int, int]:
    """Return source and target nodes for the given edge id."""

    src = int(edge_index[0, edge_id])
    tgt = int(edge_index[1, edge_id])
    return src, tgt

def load_scenario_from_json(path: str | Path) -> Tuple[Data, np.ndarray]:
    """Load a directed network and OD demand matrix from a JSON file.

    The file must contain ``nodes`` and ``edges`` lists with ``source``, ``target``,
    ``capacity`` and ``freeflow_travel_time`` fields, as well as a ``demand`` dictionary
    with keys formatted as ``"(o,d)"``.

    Returns:
        A tuple of (network_data, od_matrix).
    """

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    network = payload.get("network", {})
    demand = payload.get("demand", {})

    network_data = load_network(network)
    od_matrix = load_demand(demand)

    return network_data, od_matrix


def load_network(dict: Dict) -> Data:
    """Load a directed network from a dictionary object.

    The dict must contain ``nodes`` and ``edges`` lists with ``source``, ``target``,
    ``capacity`` and ``freeflow_travel_time`` fields.
    """

    nodes = dict.get("nodes", [])
    edges = dict.get("edges", [])

    node_ids = [n["id"] for n in nodes]
    num_nodes = max(node_ids) + 1 if node_ids else 0

    sources: List[int] = []
    targets: List[int] = []
    capacities: List[float] = []
    ff_times: List[float] = []

    for edge in edges:
        sources.append(int(edge["source"]))
        targets.append(int(edge["target"]))
        capacities.append(float(edge["capacity"]))
        ff_times.append(float(edge["freeflow_travel_time"]))

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(np.stack([ff_times, capacities], axis=1), dtype=torch.float)

    return Data(num_nodes=num_nodes, edge_index=edge_index, edge_attr=edge_attr)


def load_demand(dict: Dict) -> np.ndarray:
    """Load the OD demand from a dictionary object.
    Creates a list of agents based on the demand matrix.

    The JSON must have a ``demand`` dictionary with keys formatted as ``"o,d"`` and values representing demand.
    Returns a tensor of shape (2, n) with n equal to the total demand. 
    The first row contains origins and the second row contains destinations.
    """

    origins: List[int] = []
    destinations: List[int] = []

    for key, value in dict.items():
        o_str, d_str = key.split(",")
        origin = int(o_str)
        destination = int(d_str)
        demand = int(value)

        origins.extend([origin] * demand)
        destinations.extend([destination] * demand)
    
    return torch.tensor([origins, destinations], dtype=torch.long)

__all__ = [
    "NetworkMetadata",
    "build_out_edges_per_node",
    "map_action_to_edge",
    "edge_id_to_nodes",
    "load_network_from_json",
    "load_demand_from_json",
]
