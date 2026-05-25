"""Shared utility for building CSR-format candidate route tables.

Replaces the dense padded tensor [NumUniqueOD, K, MaxRouteLen] with a compact
CSR representation:
  - routes_flat:    [TotalActualEdges] int32  – edge IDs concatenated, no padding
  - routes_offsets: [NumUniqueOD * K + 1] int64 – start offset of each route

Lookup for route (od_idx, k_idx):
    row   = od_idx * K + k_idx
    start = routes_offsets[row]
    end   = routes_offsets[row + 1]
    edges = routes_flat[start:end]          # variable length, no -1 padding

Memory comparison for Berlin 1% (32 314 OD, K=9, MaxRouteLen=844):
    Dense:  32314 × 9 × 844 × 4 B =  936 MB
    CSR:    32314 × 9 × avg_len × 4 B + 32314 × 9 × 8 B ≈  50-120 MB (typ.)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def build_routes_csr(
    paths_dict: Dict[Tuple[int, int], List[List[int]]],
    unique_od: np.ndarray,   # [NumUniqueOD, 2] int array
    top_k: int,
    edge_static_np: np.ndarray | None = None,  # [E, 5] for FFTT computation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Build CSR candidate routes from a paths dictionary.

    Args:
        paths_dict:    {(origin, dest): [[edge_id, ...], ...]} – top-k paths per OD.
        unique_od:     [NumUniqueOD, 2] int32 array of unique OD pairs.
        top_k:         Number of candidate routes per OD.
        edge_static_np: Optional [E, 5] float array; if provided, computes fftt_matrix.

    Returns:
        routes_flat_np:    [TotalActualEdges] int32 – all edges concatenated (no -1).
        routes_offsets_np: [NumUniqueOD * K + 1] int64 – CSR offsets.
        masks_np:          [NumUniqueOD, K] bool – valid action mask.
        fftt_matrix_np:    [NumUniqueOD, K] float32 or None – free-flow travel times.
    """
    num_unique_od = len(unique_od)
    num_routes = num_unique_od * top_k

    # Pre-allocate offset array; offsets[row+1] will hold the route length
    routes_offsets_np = np.zeros(num_routes + 1, dtype=np.int64)
    masks_np = np.zeros((num_unique_od, top_k), dtype=bool)

    # First pass: determine lengths and validity
    route_arrays: List[np.ndarray] = []
    for od_idx in range(num_unique_od):
        od_key = (int(unique_od[od_idx, 0]), int(unique_od[od_idx, 1]))
        paths_list = paths_dict.get(od_key, [])

        for k_idx in range(top_k):
            row = od_idx * top_k + k_idx
            if paths_list:
                p_idx = min(k_idx, len(paths_list) - 1)
                path = np.array(paths_list[p_idx], dtype=np.int32)
                route_arrays.append(path)
                routes_offsets_np[row + 1] = len(path)
                masks_np[od_idx, k_idx] = (k_idx < len(paths_list))
            else:
                route_arrays.append(np.array([], dtype=np.int32))
                routes_offsets_np[row + 1] = 0

    # Convert lengths to offsets (cumsum in-place)
    np.cumsum(routes_offsets_np, out=routes_offsets_np)

    # Concatenate all route arrays into one flat buffer
    if route_arrays:
        routes_flat_np = np.concatenate(route_arrays).astype(np.int32)
    else:
        routes_flat_np = np.array([], dtype=np.int32)

    # Optional: compute free-flow travel time matrix
    fftt_matrix_np: np.ndarray | None = None
    if edge_static_np is not None:
        fftt_matrix_np = np.zeros((num_unique_od, top_k), dtype=np.float32)
        for od_idx in range(num_unique_od):
            for k_idx in range(top_k):
                row = od_idx * top_k + k_idx
                start = int(routes_offsets_np[row])
                end   = int(routes_offsets_np[row + 1])
                if masks_np[od_idx, k_idx] and end > start:
                    path_edges = routes_flat_np[start:end]
                    fftt_matrix_np[od_idx, k_idx] = edge_static_np[path_edges, 4].sum()
                else:
                    fftt_matrix_np[od_idx, k_idx] = np.inf

    return routes_flat_np, routes_offsets_np, masks_np, fftt_matrix_np
