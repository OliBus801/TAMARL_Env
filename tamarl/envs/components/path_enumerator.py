"""Top-k path enumeration supporting Yen (igraph), Sidetrack (rustworkx), and Penalty methods.

Provides utilities to enumerate the k-shortest loopless paths between
origin-destination pairs in a directed graph, using free-flow travel
times as edge weights.

Algorithms
----------
- ``yen``       : Exact k-shortest loopless paths via igraph (Yen's algorithm).
                  Complexity: O(k · |V| · (|E| + |V| log |V|)).
                  Best for small networks or reference computations.
- ``sidetrack``  : Exact k-shortest simple paths via rustworkx (Rust backend).
                  Internally uses an Eppstein/Yen hybrid, significantly faster
                  than igraph on large graphs.
                  Complexity: O(|E| + |V| log |V| + k log k) per OD pair.
- ``penalty``   : Approximate k spatially-diverse paths via iterative Dijkstra
                  with multiplicative edge penalization.
                  Complexity: O(k · (|E| + |V| log |V|)) per OD pair.
                  Best for DTA/MARL where spatial diversity matters more than
                  strict optimality.
"""
from __future__ import annotations

import heapq
import os
import pickle
from typing import Dict, List, Optional, Tuple

import igraph as ig
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Worker-process globals – each subprocess gets its own copy via initializer
# ---------------------------------------------------------------------------
_WORKER_IG_GRAPH: Optional[ig.Graph] = None           # yen
_WORKER_RX_GRAPH: Optional[tuple] = None              # sidetrack  (graph, node_edge_map)
_WORKER_ADJ: Optional[Dict] = None                    # penalty    adjacency list
_WORKER_BASE_WEIGHTS: Optional[np.ndarray] = None     # penalty    base travel times


# ===========================================================================
#  YEN  –  igraph (exact, multiprocess)
# ===========================================================================

def _init_yen_worker(graph: ig.Graph) -> None:
    global _WORKER_IG_GRAPH
    _WORKER_IG_GRAPH = graph


def _compute_yen_chunk(
    od_chunk: np.ndarray, k: int
) -> Dict[Tuple[int, int], List[List[int]]]:
    global _WORKER_IG_GRAPH
    result: Dict[Tuple[int, int], List[List[int]]] = {}
    for i in range(od_chunk.shape[0]):
        o, d = int(od_chunk[i, 0]), int(od_chunk[i, 1])
        od_key = (o, d)
        if o == d:
            result[od_key] = [[]]
            continue
        try:
            paths_edges = _WORKER_IG_GRAPH.get_k_shortest_paths(
                o, to=d, k=k, weights="weight", output="epath"
            )
            paths = []
            for path in paths_edges:
                if path:
                    paths.append(
                        [_WORKER_IG_GRAPH.es[e]["original_id"] for e in path]
                    )
            result[od_key] = paths
        except Exception:
            result[od_key] = []
    return result


def _enumerate_yen_paths(
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Exact k-shortest loopless paths via igraph's Yen implementation."""
    # Filter parallel edges – keep only the fastest per (u, v)
    edge_dict: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for e in range(edge_endpoints.shape[0]):
        u, v = int(edge_endpoints[e, 0]), int(edge_endpoints[e, 1])
        w = float(ff_times[e])
        if (u, v) not in edge_dict or w < edge_dict[(u, v)][1]:
            edge_dict[(u, v)] = (e, w)

    filtered_edges = list(edge_dict.keys())
    original_ids = [edge_dict[uv][0] for uv in filtered_edges]
    weights = [edge_dict[uv][1] for uv in filtered_edges]

    G = ig.Graph(n=num_nodes, directed=True)
    G.add_edges(filtered_edges)
    G.es["weight"] = weights
    G.es["original_id"] = original_ids

    num_workers = os.cpu_count() or 4
    chunks = np.array_split(od_pairs, max(1, num_workers * 4))

    result: Dict[Tuple[int, int], List[List[int]]] = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_yen_worker,
        initargs=(G,),
    ) as executor:
        futures = [
            executor.submit(_compute_yen_chunk, chunk, k)
            for chunk in chunks
            if len(chunk) > 0
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Yen k-paths (OD chunks)"
        ):
            result.update(future.result())
    return result


# ===========================================================================
#  SIDETRACK  –  rustworkx (exact, Rust backend, much faster on large graphs)
# ===========================================================================

def _init_sidetrack_worker(
    nx_graph, node_edge_map: Dict[Tuple[int, int], int]
) -> None:
    global _WORKER_NX_GRAPH
    _WORKER_NX_GRAPH = (nx_graph, node_edge_map)

def _compute_sidetrack_chunk(
    od_chunk: np.ndarray, k: int
) -> Dict[Tuple[int, int], List[List[int]]]:
    import networkx as nx
    from itertools import islice

    global _WORKER_NX_GRAPH
    G, node_edge_map = _WORKER_NX_GRAPH
    result: Dict[Tuple[int, int], List[List[int]]] = {}

    for i in range(od_chunk.shape[0]):
        o, d = int(od_chunk[i, 0]), int(od_chunk[i, 1])
        od_key = (o, d)
        if o == d:
            result[od_key] = [[]]
            continue
        try:
            paths_nodes = list(islice(nx.shortest_simple_paths(G, o, d, weight="weight"), k))
            paths = []
            for node_path in paths_nodes:
                if len(node_path) < 2:
                    continue
                edge_path: List[int] = []
                valid = True
                for j in range(len(node_path) - 1):
                    uv = (int(node_path[j]), int(node_path[j + 1]))
                    eid = node_edge_map.get(uv)
                    if eid is None:
                        valid = False
                        break
                    edge_path.append(eid)
                if valid and edge_path:
                    paths.append(edge_path)
            result[od_key] = paths
        except Exception:
            result[od_key] = []
    return result


def _enumerate_sidetrack_paths(
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Exact k-shortest loopless paths via rustworkx (Rust backend)."""
    import networkx as nx

    # Filter parallel edges – keep only the fastest per (u, v)
    edge_dict: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for e in range(edge_endpoints.shape[0]):
        u, v = int(edge_endpoints[e, 0]), int(edge_endpoints[e, 1])
        w = float(ff_times[e])
        if (u, v) not in edge_dict or w < edge_dict[(u, v)][1]:
            edge_dict[(u, v)] = (e, w)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    node_edge_map: Dict[Tuple[int, int], int] = {}
    for (u, v), (orig_id, w) in edge_dict.items():
        G.add_edge(u, v, weight=w)
        node_edge_map[(u, v)] = orig_id

    num_workers = os.cpu_count() or 4
    chunks = np.array_split(od_pairs, max(1, num_workers * 4))

    result: Dict[Tuple[int, int], List[List[int]]] = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_sidetrack_worker,
        initargs=(G, node_edge_map),
    ) as executor:
        futures = [
            executor.submit(_compute_sidetrack_chunk, chunk, k)
            for chunk in chunks
            if len(chunk) > 0
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Sidetrack k-paths (OD chunks)",
        ):
            result.update(future.result())
    return result


# ===========================================================================
#  PENALTY  –  iterative Dijkstra with multiplicative edge penalization
# ===========================================================================

def _init_penalty_worker(
    adj: Dict[int, List[Tuple[int, int]]],
    base_weights: np.ndarray,
) -> None:
    global _WORKER_ADJ, _WORKER_BASE_WEIGHTS
    _WORKER_ADJ = adj
    _WORKER_BASE_WEIGHTS = base_weights


def _dijkstra_penalty(
    adj: Dict[int, List[Tuple[int, int]]],
    weights: np.ndarray,
    src: int,
    dst: int,
) -> Optional[List[int]]:
    """Run Dijkstra with given weights; return edge-path or None if unreachable."""
    dist: Dict[int, float] = {src: 0.0}
    prev_edge: Dict[int, Tuple[int, int]] = {}  # node -> (edge_id, parent_node)
    heap = [(0.0, src)]
    visited: set = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            path: List[int] = []
            cur = dst
            while cur != src:
                eid, par = prev_edge[cur]
                path.append(eid)
                cur = par
            path.reverse()
            return path
        for v, eid in adj.get(u, []):
            nd = d + weights[eid]
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev_edge[v] = (eid, u)
                heapq.heappush(heap, (nd, v))
    return None


def _compute_penalty_chunk(
    od_chunk: np.ndarray,
    k: int,
    penalty_factor: float,
) -> Dict[Tuple[int, int], List[List[int]]]:
    global _WORKER_ADJ, _WORKER_BASE_WEIGHTS
    result: Dict[Tuple[int, int], List[List[int]]] = {}

    for i in range(od_chunk.shape[0]):
        o, d = int(od_chunk[i, 0]), int(od_chunk[i, 1])
        od_key = (o, d)
        if o == d:
            result[od_key] = [[]]
            continue

        # Fresh copy of weights for each (o, d)
        weights = _WORKER_BASE_WEIGHTS.copy()
        paths: List[List[int]] = []
        
        max_attempts = k * 10
        attempts = 0

        while len(paths) < k and attempts < max_attempts:
            path = _dijkstra_penalty(_WORKER_ADJ, weights, o, d)
            attempts += 1
            if path is None:
                break
                
            # Penalize all edges used by this path to force exploration
            for eid in path:
                weights[eid] *= penalty_factor
                
            # Only append if strictly distinct
            if path not in paths:
                paths.append(path)

        result[od_key] = paths
    return result


def _enumerate_penalty_paths(
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
    penalty_factor: float = 1.5,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Approximate k spatially-diverse paths via iterative penalized Dijkstra."""
    # Filter parallel edges – keep only the fastest per (u, v)
    edge_dict: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for e in range(edge_endpoints.shape[0]):
        u, v = int(edge_endpoints[e, 0]), int(edge_endpoints[e, 1])
        w = float(ff_times[e])
        if (u, v) not in edge_dict or w < edge_dict[(u, v)][1]:
            edge_dict[(u, v)] = (e, w)

    adj: Dict[int, List[Tuple[int, int]]] = {}
    for (u, v), (e, _) in edge_dict.items():
        adj.setdefault(u, []).append((v, e))

    base_weights = ff_times.copy()

    num_workers = os.cpu_count() or 4
    chunks = np.array_split(od_pairs, max(1, num_workers * 4))

    result: Dict[Tuple[int, int], List[List[int]]] = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_penalty_worker,
        initargs=(adj, base_weights),
    ) as executor:
        futures = [
            executor.submit(_compute_penalty_chunk, chunk, k, penalty_factor)
            for chunk in chunks
            if len(chunk) > 0
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Penalty k-paths (OD chunks)",
        ):
            result.update(future.result())
    return result


# ===========================================================================
#  Public API
# ===========================================================================

# Kept for backward compatibility (was the original public entry point)
def enumerate_top_k_paths(
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
    method: str = "yen",
    penalty_factor: float = 1.5,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Enumerate top-k paths using the selected algorithm.

    Parameters
    ----------
    method : {"yen", "sidetrack", "penalty"}
        Algorithm to use (see module docstring for details).
    penalty_factor : float
        Multiplicative factor applied to penalized edges (``method="penalty"`` only).
    """
    if method == "yen":
        return _enumerate_yen_paths(num_nodes, edge_endpoints, ff_times, od_pairs, k)
    elif method == "sidetrack":
        return _enumerate_sidetrack_paths(
            num_nodes, edge_endpoints, ff_times, od_pairs, k
        )
    elif method == "penalty":
        return _enumerate_penalty_paths(
            num_nodes, edge_endpoints, ff_times, od_pairs, k, penalty_factor
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: yen, sidetrack, penalty.")


def get_or_compute_top_k_paths(
    scenario_dir: str,
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
    method: str = "yen",
    penalty_factor: float = 1.5,
    force_recompute: bool = False,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Get top-k paths from cache, or compute and cache them.

    Each (method, k, penalty_factor) combination uses a distinct cache file so
    that switching algorithms never produces stale results.

    Cache file naming
    -----------------
    - yen       →  ``top_k_paths_k{k}_yen.pkl``
      (also accepts legacy ``top_k_paths_k{k}.pkl`` for backward compat.)
    - sidetrack →  ``top_k_paths_k{k}_sidetrack.pkl``
    - penalty   →  ``top_k_paths_k{k}_penalty_f{penalty_factor}.pkl``
    """
    if method == "yen":
        cache_path = os.path.join(scenario_dir, f"top_k_paths_k{k}_yen.pkl")
        legacy_path = os.path.join(scenario_dir, f"top_k_paths_k{k}.pkl")
    elif method == "sidetrack":
        cache_path = os.path.join(scenario_dir, f"top_k_paths_k{k}_sidetrack.pkl")
        legacy_path = None
    elif method == "penalty":
        pf_str = f"{penalty_factor:.2f}".rstrip("0").rstrip(".")
        cache_path = os.path.join(
            scenario_dir, f"top_k_paths_k{k}_penalty_f{pf_str}.pkl"
        )
        legacy_path = None
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: yen, sidetrack, penalty."
        )

    paths_dict: Dict[Tuple[int, int], List[List[int]]] = {}

    # Try primary cache
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached top-{k} paths ({method}) from {cache_path}")
        with open(cache_path, "rb") as f:
            paths_dict = pickle.load(f)
    # Fallback to legacy cache for yen
    elif not force_recompute and legacy_path and os.path.exists(legacy_path):
        print(f"Loading legacy cached top-{k} paths from {legacy_path}")
        with open(legacy_path, "rb") as f:
            paths_dict = pickle.load(f)

    # Check for missing OD pairs
    missing_od_pairs = [
        [int(od_pairs[i, 0]), int(od_pairs[i, 1])]
        for i in range(od_pairs.shape[0])
        if (int(od_pairs[i, 0]), int(od_pairs[i, 1])) not in paths_dict
    ]

    if missing_od_pairs:
        if paths_dict:
            print(
                f"Found {len(missing_od_pairs)} missing OD pairs in cache. "
                f"Computing with '{method}'..."
            )
        else:
            print(
                f"Computing top-{k} paths with '{method}' "
                f"(this may take a while for large networks)..."
            )

        missing_np = np.array(missing_od_pairs, dtype=np.int32)
        new_paths = enumerate_top_k_paths(
            num_nodes=num_nodes,
            edge_endpoints=edge_endpoints,
            ff_times=ff_times,
            od_pairs=missing_np,
            k=k,
            method=method,
            penalty_factor=penalty_factor,
        )
        paths_dict.update(new_paths)

        os.makedirs(scenario_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(paths_dict, f)
        print(f"Saved updated paths to {cache_path}")

    return paths_dict