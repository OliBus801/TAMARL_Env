
import torch
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tamarl.core.dnl import TorchDNL
from tamarl.core.plot_histogram import plot_agent_status

def run_benchmark():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    num_nodes = 8827
    num_edges = 21396
    num_agents = 1000
    max_path_len = 50
    
    print(f"Simulating {num_agents} agents on {num_edges} edges.")

    # Synthetic Data
    # Edge Static: [len, ff_speed, cap_store, cap_flow, ff_time]
    # random values
    lengths = torch.randint(100, 1000, (num_edges, 1)).float()
    ff_speed = torch.randint(10, 30, (num_edges, 1)).float()
    cap_flow = torch.randint(1, 5, (num_edges, 1)).float() # cars per step
    cap_store = lengths / 5.0 # density ~ 1 car per 5m
    ff_time = torch.ceil(lengths / ff_speed)
    
    edge_static = torch.cat([lengths, ff_speed, cap_store, cap_flow, ff_time], dim=1)
    
    # --- Graph Topology & Path Generation ---
    print("Generating Graph Topology...")
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    # Generate random edges (u, v)
    # Ensure they are valid nodes 0...num_nodes-1
    # We want a connected graph usually, but random is fine for benchmark.
    # To ensure some connectivity, we can link i -> i+1 first, then random.
    
    sources = np.random.randint(0, num_nodes, num_edges)
    targets = np.random.randint(0, num_nodes, num_edges)
    
    # Avoid self-loops intuitively (though not critical)
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    
    # We might have lost some edges due to self-loops, assume num_edges is flexible or truncate
    actual_num_edges = len(sources)
    print(f"Generated {actual_num_edges} valid topological links.")
    
    # Weights
    weights = ff_time.flatten().numpy()[:actual_num_edges] # Use ff_time as weight
    
    # Create CSR matrix
    # Note: Duplicate edges will sum weights in csr_matrix constructor usually, 
    # but we want distinct edges.
    # We need to Map (u,v) -> edge_idx to reconstruct paths later.
    # Since we might have parallel edges (u,v same), we need to handle that.
    # For benchmark simplicity, we'll keep only unique (u,v) pairs for the graph routing,
    # but TorchDNL needs 'paths' as sequences of edge indices.
    # Let's clean up duplicates.
    
    edge_map = {} # (u,v) -> edge_idx
    
    clean_sources = []
    clean_targets = []
    clean_weights = []
    clean_edge_indices = []
    
    seen = set()
    for i in range(actual_num_edges):
        u, v = sources[i], targets[i]
        if (u, v) not in seen:
            seen.add((u,v))
            clean_sources.append(u)
            clean_targets.append(v)
            clean_weights.append(weights[i])
            clean_edge_indices.append(i) # Original index (roughly, if we ignore the filtered self-loops)
            # Wait, we need valid edge_static indices.
            # Let's just resize edge_static to match our clean graph to be safe and consistent.
            # Or simpler: Re-sample edge_static for the clean edges.
            
    # Update num_edges to clean count
    num_edges = len(clean_sources)
    
    # Re-build edge_static matches
    # We need to select the static attributes corresponding to these edges
    # We'll just slice the original edge_static
    edge_static = edge_static[:num_edges]
    
    # Edge Map
    edge_map = {(u,v): i for i, (u, v) in enumerate(zip(clean_sources, clean_targets))}
    
    # CSR
    graph_csr = csr_matrix((clean_weights, (clean_sources, clean_targets)), shape=(num_nodes, num_nodes))
    
    print("Calculating Optimal Dijkstra Paths for every agent...")
    
    # Strategy:
    # 100k agents. Computing all-pairs is impossible.
    # Computing 100k one-to-one is slow in Python loop.
    # We will pick a subset of nodes as 'Centroids' or 'Origins'.
    # Say 100 random origins. Assign agents to start at one of these 100.
    # Destinations can be distinct.
    # We run 100 one-to-all Dijkstras.
    
    num_origins = 50
    chosen_origins = np.random.choice(num_nodes, num_origins, replace=False)
    
    # Agent assignments
    agent_origins = np.random.choice(chosen_origins, num_agents)
    agent_destinations = np.random.randint(0, num_nodes, num_agents)
    
    # Run Dijkstra for each unique origin
    # predecessors[i, j] = predecessor of node j in path from origin i
    # We need to map `chosen_origins` index to the row in results
    origin_to_idx = {node: i for i, node in enumerate(chosen_origins)}
    
    print(f"Running Dijkstra from {num_origins} unique origins...")
    dist_matrix, predecessors = shortest_path(csgraph=graph_csr, directed=True, indices=chosen_origins, return_predecessors=True)
    
    # Reconstruct paths
    # This is the slow part in Python.
    print("Reconstructing paths...")
    
    path_list = []
    
    # Vectorization is hard for path reconstruction.
    # We'll do a simple loop, maybe optimized?
    # Actually, valid paths might not exist for all pairs (disconnected graph).
    # We need to handle that.
    
    processed_count = 0
    
    # Pre-compute edge_map lookup if possible? It's a dict.
    # (u,v) key.
    
    final_paths_tensor = torch.full((num_agents, max_path_len), -1, dtype=torch.long)
    
    for i in range(num_agents):
        u = agent_origins[i]
        v = agent_destinations[i]
        
        if u == v:
            continue
            
        row_idx = origin_to_idx[u]
        
        # Check reachability
        if predecessors[row_idx, v] == -9999: # Scipy uses -9999 for foundational nodes/unreachable
            continue
            
        # Backtrack
        curr = v
        node_path = [curr]
        while curr != u:
            prev = predecessors[row_idx, curr]
            if prev == -9999: # Should not happen if reachable
                break
            node_path.append(prev)
            curr = prev
            
        # Reverse to get u -> ... -> v
        node_path = node_path[::-1]
        
        # Convert to edges
        edge_path = []
        valid_path = True
        for k in range(len(node_path)-1):
            n1, n2 = node_path[k], node_path[k+1]
            if (n1, n2) in edge_map:
                edge_path.append(edge_map[(n1, n2)])
            else:
                valid_path = False
                break
        
        if valid_path and len(edge_path) > 0:
            # Truncate to max_path_len
            length = min(len(edge_path), max_path_len)
            final_paths_tensor[i, :length] = torch.tensor(edge_path[:length], dtype=torch.long)
            
        processed_count += 1
        if processed_count % 10000 == 0:
            print(f"Reconstructed {processed_count} paths...")

    paths = final_paths_tensor
    print(f"Finished generating paths. Valid paths found for {processed_count} agents.")

    
    # Init Engine
    dnl = TorchDNL(edge_static, paths, device=device, enable_profiling=True)
    
    # Init Agents
    dnl.agent_state[:, 0] = 0 # Waiting
    dnl.agent_state[:, 2] = paths[:, 0].to(device) # Target is first edge in path
    
    # Run loop
    print("Running simulation until all agents finish...")
    
    step = 0
    max_steps = 5000 # Safety break
    
    active_agents = True
    
    # Data collection
    history_arrived = []
    history_en_route = []
    history_steps = []
    
    start_time = time.time()
    
    while active_agents and step < max_steps:
        dnl.step()
        step += 1
        
        # Output metrics every N steps for progress
        if step % 100 == 0:
            metrics = dnl.get_metrics()
            print(f"Step {step}: Arrived={metrics['arrived_count']}, EnRoute={metrics['en_route_count']}, "
                  f"AvgTime={metrics['avg_travel_time']:.1f}, AvgDist={metrics['avg_travel_dist']:.1f}")
            
        # Collect Data for Plotting every 10 steps
        if step % 10 == 0:
            metrics = dnl.get_metrics()
            history_arrived.append(metrics['arrived_count'])
            history_en_route.append(metrics['en_route_count'])
            history_steps.append(step)

        # Check if every agent has finished every step (using last metrics)
        # We need fresh metrics if we didn't just call get_metrics ensure check is accurate?
        # dnl.get_metrics() is cheap, but let's reuse if calculated.
        # Actually, let's just check the en_route from the history if we just appended, 
        # or call it if we are at step % 100 but not % 10 (unlikely with %10).
        # To be safe, just call it for the check if needed or rely on the collection loop.
        
        if len(history_en_route) > 0 and history_en_route[-1] == 0:
             active_agents = False
             print(f"All agents have finished at step {step}.")

    total_sim_time = time.time() - start_time
    print(f"Simulation finished in {step} steps and {total_sim_time:.2f}s.")
    
    print("\n--- Profiling Stats ---")
    dnl.print_stats(limit=20)
    print("-----------------------")
    
    # Plotting
    metrics = dnl.agent_metrics.cpu().numpy()
    start_steps = metrics[:, 0]
    travel_times = metrics[:, 2]
    
    # Calculate arrival steps (start + travel)
    # Filter for those who finished (state 2) or have positive travel time
    state = dnl.agent_state[:, 0].cpu().numpy()
    finished = (state == 2)
    
    # It is possible some are Done but maybe metrics not fully synced if I grabbed metrics early? 
    # agent_metrics are updated in process_exit.
    
    arrival_steps = start_steps[finished] + travel_times[finished]
    start_steps = start_steps[state != -1] # Only started agents
    
    print("Generating Plot using plot_agent_status...")
    plot_agent_status(start_steps, arrival_steps, max_steps=step, bucket_size_sec=1, output_file='agent_status_plot.png')

    
if __name__ == "__main__":
    run_benchmark()
