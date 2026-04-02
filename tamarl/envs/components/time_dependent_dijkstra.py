"""Time-dependent shortest path algorithm."""

import heapq
import numpy as np

def build_adjacency_list(num_nodes: int, edge_endpoints: np.ndarray):
    """Builds adjacency list from endpoint array.
    Returns list of lists: adj[u] = [(v, edge_internal_idx), ...]
    """
    adj = [[] for _ in range(num_nodes)]
    for e in range(edge_endpoints.shape[0]):
        u, v = edge_endpoints[e]
        adj[u].append((v, e))
    return adj

def compute_td_shortest_paths(
    adj: list,
    start_times: np.ndarray,      # shape: [N], seconds
    origin_nodes: np.ndarray,     # shape: [N]
    destination_nodes: np.ndarray,# shape: [N]
    tt_matrix: np.ndarray,        # shape: [max_intervals, num_edges], seconds
    interval: float
) -> np.ndarray:
    """Computes pure time-dependent shortest paths using dynamic link travel times.
    
    Args:
        adj: adjacency list
        start_times: array of departure times for each query
        origin_nodes: array of origins
        destination_nodes: array of destinations
        tt_matrix: historic dynamically recorded link TT matrix
        interval: time per interval in seconds
        
    Returns:
        np.ndarray containing optimal travel times (t_SP) for each query.
    """
    N = len(start_times)
    t_sp = np.zeros(N, dtype=np.float32)
    num_intervals = tt_matrix.shape[0]
    
    for i in range(N):
        start_time = float(start_times[i])
        start_node = int(origin_nodes[i])
        dest_node = int(destination_nodes[i])
        
        # If agent hasn't started or is already at destination
        if start_node == dest_node:
            t_sp[i] = 0.0
            continue
            
        dist = {start_node: start_time}
        pq = [(start_time, start_node)]
        found = False
        
        while pq:
            curr_time, u = heapq.heappop(pq)
            
            if u == dest_node:
                t_sp[i] = curr_time - start_time
                found = True
                break
                
            if curr_time > dist.get(u, float('inf')):
                continue
                
            # Compute current interval
            interval_idx = int(curr_time // interval)
            if interval_idx >= num_intervals:
                interval_idx = num_intervals - 1
                
            for v, edge_id in adj[u]:
                tt = tt_matrix[interval_idx, edge_id]
                next_time = curr_time + float(tt)
                if next_time < dist.get(v, float('inf')):
                    dist[v] = next_time
                    heapq.heappush(pq, (next_time, v))
                    
        if not found:
            # If unreachable
            t_sp[i] = float('inf')
            
    return t_sp
