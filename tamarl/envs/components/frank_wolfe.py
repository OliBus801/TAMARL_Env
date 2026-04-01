"""Frank-Wolfe algorithm for Static Traffic Assignment (TAP).

Computes User Equilibrium flows using the BPR cost function
as a theoretical baseline for evaluating RL agents.

BPR cost: t_a(x_a) = t_a^0 * (1 + alpha * (x_a / c_a)^beta)

References:
    - Frank, M. & Wolfe, P. (1956). An algorithm for quadratic programming.
    - Sheffi, Y. (1985). Urban Transportation Networks.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from tamarl.envs.scenario_loader import ScenarioData


@dataclass
class TAPResult:
    """Result of the static traffic assignment."""
    tstt: float                     # Total System Travel Time at equilibrium
    flows: np.ndarray               # [E] equilibrium link flows
    travel_times: np.ndarray        # [E] equilibrium link travel times
    converged: bool                 # Whether the algorithm converged
    iterations: int                 # Number of FW iterations
    gap_history: List[float]        # Relative gap per iteration


def _bpr_cost(flows: np.ndarray, ff_times: np.ndarray, capacities: np.ndarray,
              alpha: float = 0.15, beta: float = 4.0) -> np.ndarray:
    """BPR travel time function: t_a(x_a) = t_a^0 * (1 + alpha * (x_a / c_a)^beta)."""
    ratio = np.divide(flows, capacities, out=np.zeros_like(flows), where=capacities > 0)
    return ff_times * (1.0 + alpha * np.power(ratio, beta))


def _bpr_integral(flows: np.ndarray, ff_times: np.ndarray, capacities: np.ndarray,
                  alpha: float = 0.15, beta: float = 4.0) -> float:
    """Beckmann objective: sum_a integral_0^{x_a} t_a(w) dw.
    
    = sum_a [ t_a^0 * x_a * (1 + alpha/(beta+1) * (x_a/c_a)^beta) ]
    """
    ratio = np.divide(flows, capacities, out=np.zeros_like(flows), where=capacities > 0)
    integrals = ff_times * flows * (1.0 + alpha / (beta + 1.0) * np.power(ratio, beta))
    return float(integrals.sum())


def _dijkstra(num_nodes: int, adj: Dict[int, List[Tuple[int, int, float]]],
              source: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dijkstra's shortest path from source.
    
    Args:
        num_nodes: total number of nodes
        adj: adjacency list: node -> [(neighbor, edge_id, cost), ...]
        source: source node
        
    Returns:
        dist: [N] shortest distances
        pred_edge: [N] predecessor edge index (-1 if no path)
    """
    dist = np.full(num_nodes, np.inf)
    pred_edge = np.full(num_nodes, -1, dtype=np.int64)
    dist[source] = 0.0
    
    # (distance, node)
    pq = [(0.0, source)]
    visited = np.zeros(num_nodes, dtype=bool)
    
    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        
        if u not in adj:
            continue
        for v, edge_id, cost in adj[u]:
            new_dist = d + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                pred_edge[v] = edge_id
                heapq.heappush(pq, (new_dist, v))
    
    return dist, pred_edge


def _build_adjacency(edge_endpoints: np.ndarray, costs: np.ndarray
                     ) -> Dict[int, List[Tuple[int, int, float]]]:
    """Build adjacency list from edge endpoints and costs."""
    adj: Dict[int, List[Tuple[int, int, float]]] = {}
    for e in range(edge_endpoints.shape[0]):
        u = int(edge_endpoints[e, 0])
        v = int(edge_endpoints[e, 1])
        cost = float(costs[e])
        if u not in adj:
            adj[u] = []
        adj[u].append((v, e, cost))
    return adj


def _all_or_nothing(num_nodes: int, num_edges: int,
                    adj: Dict[int, List[Tuple[int, int, float]]],
                    edge_endpoints: np.ndarray,
                    od_demand: List[Tuple[int, int, float]]) -> np.ndarray:
    """All-or-nothing assignment: assign all demand to shortest paths.
    
    Args:
        num_nodes: number of nodes
        num_edges: number of edges  
        adj: adjacency list with current costs
        edge_endpoints: [E, 2] array
        od_demand: list of (origin, destination, demand) tuples
        
    Returns:
        flows: [E] link flows from all-or-nothing assignment
    """
    flows = np.zeros(num_edges, dtype=np.float64)
    
    # Group demand by origin for efficiency
    origin_demands: Dict[int, List[Tuple[int, float]]] = {}
    for o, d, demand in od_demand:
        if o not in origin_demands:
            origin_demands[o] = []
        origin_demands[o].append((d, demand))
    
    for origin, dest_list in origin_demands.items():
        dist, pred_edge = _dijkstra(num_nodes, adj, origin)
        
        for dest, demand in dest_list:
            if dist[dest] == np.inf:
                continue  # No path exists
            
            # Trace back path and add flows
            node = dest
            while pred_edge[node] != -1:
                e = pred_edge[node]
                flows[e] += demand
                node = int(edge_endpoints[e, 0])  # Go to predecessor node
    
    return flows


def _extract_od_demand(scenario: ScenarioData) -> List[Tuple[int, int, float]]:
    """Extract OD demand from scenario data.
    
    Aggregates per-agent OD pairs into (origin, destination, count) tuples.
    Origin = from_node of the agent's first_edge.
    Destination = destination node index.
    """
    edge_endpoints = scenario.edge_endpoints.numpy()
    first_edges = scenario.first_edges.numpy()
    destinations = scenario.destinations.numpy()
    
    # Aggregate OD pairs across all agents and all their legs
    od_counts: Dict[Tuple[int, int], float] = {}
    for i in range(first_edges.shape[0]):
        for leg in range(first_edges.shape[1]):
            fe = first_edges[i, leg]
            if fe >= 0:  # valid leg
                o = int(edge_endpoints[fe, 0])
                d = int(destinations[i, leg])
                key = (o, d)
                od_counts[key] = od_counts.get(key, 0.0) + 1.0
    
    return [(o, d, count) for (o, d), count in od_counts.items()]


def solve_static_tap(
    scenario: ScenarioData,
    alpha: float = 0.15,
    beta: float = 4.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    verbose: bool = False,
) -> TAPResult:
    """Solve the static Traffic Assignment Problem using Frank-Wolfe.
    
    Args:
        scenario: ScenarioData with network and demand information
        alpha: BPR alpha parameter
        beta: BPR beta parameter
        max_iter: maximum number of Frank-Wolfe iterations
        tol: convergence tolerance for the relative gap
        verbose: print iteration progress
        
    Returns:
        TAPResult with equilibrium flows, TSTT, and convergence info
    """
    # Extract network data
    edge_endpoints = scenario.edge_endpoints.numpy().astype(np.int64)
    ff_times = scenario.edge_static[:, 4].numpy().astype(np.float64)  # free-flow time (seconds)
    
    # Flow capacity: edge_static[:, 3] is in veh/timestep
    # For static assignment we need veh/hour (or consistent unit)
    # Since BPR uses ratio x/c, and demand is in total vehicles, we use
    # capacity as total vehicles that can flow during the assignment period.
    # For a simple approach: capacity (veh/h) = edge_static[:, 3] * 3600 / timestep
    # But edge_static[:, 3] is already D_e = (capacity_h / 3600) * timestep * scale
    # So capacity_h = D_e * 3600 / timestep
    # Since we're using total vehicles as demand, use capacity_h directly
    D_e = scenario.edge_static[:, 3].numpy().astype(np.float64)
    # D_e is veh/timestep. With timestep=1s, D_e = veh/s. Capacity in veh/h:
    capacities_h = D_e * 3600.0  # Convert back to veh/h
    
    num_nodes = scenario.num_nodes
    num_edges = scenario.num_edges
    
    # Extract OD demand
    od_demand = _extract_od_demand(scenario)
    total_demand = sum(d for _, _, d in od_demand)
    
    if verbose:
        print(f"Frank-Wolfe TAP: {num_nodes} nodes, {num_edges} edges, "
              f"{len(od_demand)} OD pairs, {total_demand:.0f} total demand")
    
    # Step 0: Initialize with all-or-nothing on free-flow times
    adj = _build_adjacency(edge_endpoints, ff_times)
    flows = _all_or_nothing(num_nodes, num_edges, adj, edge_endpoints, od_demand)
    
    gap_history = []
    converged = False
    
    for iteration in range(max_iter):
        # Compute current costs
        costs = _bpr_cost(flows, ff_times, capacities_h, alpha, beta)
        
        # All-or-nothing with current costs -> auxiliary flows
        adj = _build_adjacency(edge_endpoints, costs)
        aux_flows = _all_or_nothing(num_nodes, num_edges, adj, edge_endpoints, od_demand)
        
        # Compute relative gap = (costs · flows - costs · aux_flows) / (costs · flows)
        # This is the Wardrop gap: measures how far we are from equilibrium
        numerator = float(np.dot(costs, flows) - np.dot(costs, aux_flows))
        denominator = float(np.dot(costs, flows))
        
        if denominator > 0:
            rel_gap = numerator / denominator
        else:
            rel_gap = 0.0
        
        gap_history.append(rel_gap)
        
        if verbose and (iteration < 5 or iteration % 50 == 0 or rel_gap < tol):
            obj = _bpr_integral(flows, ff_times, capacities_h, alpha, beta)
            tstt_cur = float(np.dot(costs, flows))
            print(f"  FW iter {iteration:4d} | gap={rel_gap:.6e} | "
                  f"TSTT={tstt_cur:.1f} | Beckmann={obj:.1f}")
        
        if rel_gap < tol:
            converged = True
            break
        
        # Line search: find optimal step size via bisection
        # Minimize f(λ) = Beckmann(flows + λ * (aux_flows - flows))
        direction = aux_flows - flows
        step = _bisection_line_search(flows, direction, ff_times, capacities_h, alpha, beta)
        
        # Update flows
        flows = flows + step * direction
    
    # Final costs and TSTT
    final_costs = _bpr_cost(flows, ff_times, capacities_h, alpha, beta)
    tstt = float(np.dot(final_costs, flows))
    
    if verbose:
        status = "CONVERGED" if converged else "MAX_ITER"
        print(f"  FW {status} after {iteration + 1} iterations | "
              f"TSTT = {tstt:.1f}s | Final gap = {gap_history[-1]:.6e}")
    
    return TAPResult(
        tstt=tstt,
        flows=flows,
        travel_times=final_costs,
        converged=converged,
        iterations=iteration + 1,
        gap_history=gap_history,
    )


def _bisection_line_search(
    flows: np.ndarray, direction: np.ndarray,
    ff_times: np.ndarray, capacities: np.ndarray,
    alpha: float, beta: float,
    tol: float = 1e-8, max_iter: int = 50,
) -> float:
    """Bisection line search on the Beckmann objective."""
    lo, hi = 0.0, 1.0
    
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        # Derivative of Beckmann at mid: sum_a t_a(x_a + mid*d_a) * d_a
        test_flows = flows + mid * direction
        costs = _bpr_cost(test_flows, ff_times, capacities, alpha, beta)
        deriv = float(np.dot(costs, direction))
        
        if deriv > 0:
            hi = mid
        else:
            lo = mid
    
    return (lo + hi) / 2.0
