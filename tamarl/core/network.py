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


def load_network_from_json(path: str | Path) -> Data:
    """Load a directed network from a JSON file.

    The file must contain ``nodes`` and ``edges`` lists with ``source``, ``target``,
    ``capacity`` and ``freeflow_travel_time`` fields.
    """

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])

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


def load_demand_from_json(path: str | Path) -> np.ndarray:
    """Load an OD demand matrix from JSON.

    The JSON must have a ``demand`` dictionary with keys formatted as ``"(o,d)"``.
    Returns a dense numpy array ``od_matrix[o, d]``.
    """

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    demand_dict: Dict[str, float] = payload.get("demand", {})
    max_node = -1
    for key in demand_dict.keys():
        key_clean = key.strip("()")
        origin_str, dest_str = key_clean.split(",")
        max_node = max(max_node, int(origin_str), int(dest_str))

    size = max_node + 1 if max_node >= 0 else 0
    od_matrix = np.zeros((size, size), dtype=float)

    for key, value in demand_dict.items():
        key_clean = key.strip("()")
        origin_str, dest_str = key_clean.split(",")
        od_matrix[int(origin_str), int(dest_str)] = float(value)

    return od_matrix


__all__ = [
    "NetworkMetadata",
    "build_out_edges_per_node",
    "map_action_to_edge",
    "edge_id_to_nodes",
    "load_network_from_json",
    "load_demand_from_json",
]
