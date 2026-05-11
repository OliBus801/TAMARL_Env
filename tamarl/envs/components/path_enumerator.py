"""Top-k loopless path enumeration using Yen's algorithm.

Provides utilities to enumerate the k-shortest loopless paths between
origin-destination pairs in a directed graph, using free-flow travel
times as edge weights.
"""

from __future__ import annotations

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def enumerate_top_k_paths(
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Enumerate top-k loopless paths for each unique OD pair.

    Args:
        num_nodes: number of nodes in the graph
        edge_endpoints: [E, 2] array (from_node, to_node)
        ff_times: [E] array of free-flow travel times (edge weights)
        od_pairs: [M, 2] array of unique (origin, destination) pairs
        k: number of shortest paths per OD pair

    Returns:
        Dict mapping (origin, dest) → list of k paths (each path is a
        list of edge indices). May have fewer than k entries if fewer
        loopless paths exist.
    """
    # Build NetworkX DiGraph
    G = nx.DiGraph()
    edge_map: Dict[Tuple[int, int], int] = {}

    for e in range(edge_endpoints.shape[0]):
        u = int(edge_endpoints[e, 0])
        v = int(edge_endpoints[e, 1])
        w = float(ff_times[e])

        # If there are multiple edges between u and v, keep the shortest one
        if G.has_edge(u, v):
            if w < G[u][v]["weight"]:
                G[u][v]["weight"] = w
                edge_map[(u, v)] = e
        else:
            G.add_edge(u, v, weight=w)
            edge_map[(u, v)] = e

    result: Dict[Tuple[int, int], List[List[int]]] = {}

    for i in range(od_pairs.shape[0]):
        o = int(od_pairs[i, 0])
        d = int(od_pairs[i, 1])
        od_key = (o, d)

        if od_key in result:
            continue  # already computed

        if o == d:
            result[od_key] = [[]]
            continue

        if not G.has_node(o) or not G.has_node(d):
            result[od_key] = []
            continue

        try:
            paths_gen = nx.shortest_simple_paths(
                G, source=o, target=d, weight="weight"
            )

            paths = []
            for path_nodes in paths_gen:
                edge_path = []
                for j in range(len(path_nodes) - 1):
                    u = path_nodes[j]
                    v = path_nodes[j + 1]
                    edge_path.append(edge_map[(u, v)])
                paths.append(edge_path)

                if len(paths) >= k:
                    break

            result[od_key] = paths
        except nx.NetworkXNoPath:
            result[od_key] = []

    return result
