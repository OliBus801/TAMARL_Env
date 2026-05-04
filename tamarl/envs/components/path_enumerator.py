"""Top-k loopless path enumeration using Yen's algorithm.

Provides utilities to enumerate the k-shortest loopless paths between
origin-destination pairs in a directed graph, using free-flow travel
times as edge weights.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Set, Tuple

import numpy as np


def _dijkstra_with_exclusions(
    adj: List[List[Tuple[int, int]]],
    edge_weights: np.ndarray,
    origin: int,
    destination: int,
    excluded_nodes: Set[int],
    excluded_edges: Set[int],
) -> Tuple[float, List[int]]:
    """Single-source shortest path avoiding excluded nodes and edges.

    Args:
        adj: adjacency list, adj[u] = [(v, edge_id), ...]
        edge_weights: [E] array of edge weights (travel times)
        origin: source node
        destination: target node
        excluded_nodes: nodes to avoid (except origin/destination)
        excluded_edges: edges to avoid

    Returns:
        (cost, path) where path is a list of edge_ids. Returns (inf, []) if
        no path exists.
    """
    dist = {origin: 0.0}
    parent: Dict[int, Tuple[int, int]] = {}  # node -> (prev_node, edge_id)
    pq = [(0.0, origin)]

    while pq:
        d, u = heapq.heappop(pq)

        if u == destination:
            # Reconstruct path
            path = []
            curr = destination
            while curr in parent:
                p, eid = parent[curr]
                path.append(eid)
                curr = p
            path.reverse()
            return d, path

        if d > dist.get(u, float("inf")):
            continue

        for v, eid in adj[u]:
            if eid in excluded_edges:
                continue
            if v in excluded_nodes:
                continue
            w = float(edge_weights[eid])
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = (u, eid)
                heapq.heappush(pq, (nd, v))

    return float("inf"), []


def yen_k_shortest_paths(
    adj: List[List[Tuple[int, int]]],
    edge_endpoints: np.ndarray,
    edge_weights: np.ndarray,
    origin: int,
    destination: int,
    k: int,
) -> List[Tuple[float, List[int]]]:
    """Yen's algorithm for k-shortest loopless paths.

    Args:
        adj: adjacency list, adj[u] = [(v, edge_id), ...]
        edge_endpoints: [E, 2] array (from_node, to_node)
        edge_weights: [E] array of edge weights
        origin: source node
        destination: target node
        k: number of shortest paths to find

    Returns:
        List of (cost, path) tuples, sorted by cost. Each path is a list
        of edge indices. May return fewer than k paths if fewer exist.
    """
    if origin == destination:
        return [(0.0, [])]

    # Find the shortest path
    cost0, path0 = _dijkstra_with_exclusions(
        adj, edge_weights, origin, destination, set(), set()
    )
    if not path0:
        return []

    A = [(cost0, path0)]  # k-shortest paths found so far
    B: List[Tuple[float, List[int]]] = []  # candidates

    for i in range(1, k):
        prev_path = A[i - 1][1]  # edge list of (i-1)-th shortest path

        # Convert edge path to node path for spur operations
        if prev_path:
            node_path = [int(edge_endpoints[prev_path[0], 0])]
            for eid in prev_path:
                node_path.append(int(edge_endpoints[eid, 1]))
        else:
            node_path = [origin]

        for j in range(len(node_path) - 1):
            spur_node = node_path[j]
            root_edges = prev_path[:j]  # edges from origin to spur_node

            # Compute root cost
            root_cost = sum(float(edge_weights[eid]) for eid in root_edges)

            # Exclude edges from spur_node that are shared by existing
            # shortest paths with the same root
            excluded_edges: Set[int] = set()
            for _, p in A:
                # Convert p to node path
                if len(p) >= j + 1:
                    p_nodes = [int(edge_endpoints[p[0], 0])]
                    for eid in p:
                        p_nodes.append(int(edge_endpoints[eid, 1]))
                    # Check if the root portion matches
                    if p_nodes[: j + 1] == node_path[: j + 1]:
                        # Exclude the edge leaving spur_node in this path
                        if j < len(p):
                            excluded_edges.add(p[j])

            # Exclude nodes in root path (except spur_node) to ensure loopless
            excluded_nodes: Set[int] = set()
            for idx in range(j):
                excluded_nodes.add(node_path[idx])

            # Find spur path
            spur_cost, spur_path = _dijkstra_with_exclusions(
                adj, edge_weights, spur_node, destination,
                excluded_nodes, excluded_edges,
            )

            if spur_path:
                total_path = root_edges + spur_path
                total_cost = root_cost + spur_cost

                # Check this path isn't already in B
                is_dup = False
                for bc, bp in B:
                    if bp == total_path:
                        is_dup = True
                        break
                if not is_dup:
                    heapq.heappush(B, (total_cost, total_path))

        if not B:
            break

        # Pop the best candidate
        best_cost, best_path = heapq.heappop(B)
        A.append((best_cost, best_path))

    return A


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
    # Build adjacency list
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for e in range(edge_endpoints.shape[0]):
        u = int(edge_endpoints[e, 0])
        v = int(edge_endpoints[e, 1])
        adj[u].append((v, e))

    result: Dict[Tuple[int, int], List[List[int]]] = {}

    for i in range(od_pairs.shape[0]):
        o = int(od_pairs[i, 0])
        d = int(od_pairs[i, 1])
        od_key = (o, d)

        if od_key in result:
            continue  # already computed

        paths_with_costs = yen_k_shortest_paths(
            adj, edge_endpoints, ff_times, o, d, k
        )
        result[od_key] = [path for _, path in paths_with_costs]

    return result
