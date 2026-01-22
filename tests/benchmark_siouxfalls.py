import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import time
from tamarl.core.dnl_matsim import TorchDNLMATSim
from tamarl.core.plot_histogram import plot_agent_status

def parse_network(network_file):
    print(f"Parsing Network: {network_file}")
    tree = ET.parse(network_file)
    root = tree.getroot()
    
    # Parse Nodes
    node_id_to_idx = {}
    node_coords = []
    
    nodes_xml = root.find('nodes')
    for idx, node in enumerate(nodes_xml.findall('node')):
        nid = node.get('id')
        x = float(node.get('x'))
        y = float(node.get('y'))
        node_id_to_idx[nid] = idx
        node_coords.append([x, y])
        
    node_coords = np.array(node_coords)
    num_nodes = len(node_id_to_idx)
    print(f"Parsed {num_nodes} nodes.")
    
    # Parse Links
    edges = []
    links_xml = root.find('links')
    valid_links = 0
    
    scale_factor = 1.0 # 100% sample
    eff_cell_size = 7.5
    
    for link in links_xml.findall('link'):
        modes = link.get('modes')
        if 'car' not in modes:
            continue
            
        u_id = link.get('from')
        v_id = link.get('to')
        u = node_id_to_idx[u_id]
        v = node_id_to_idx[v_id]
        
        length = float(link.get('length'))
        freespeed = float(link.get('freespeed')) # m/s
        capacity_h = float(link.get('capacity')) # veh/h
        lanes = float(link.get('permlanes'))
        
        # Derived
        D_e = (capacity_h * scale_factor) # Flow limit per HOUR (DNL converts internally)
        c_e = (length * lanes) / eff_cell_size  # Storage limit
        ff_time = length / freespeed
        
        # [length, free_flow_speed, c_e, D_e, ff_travel_time]
        attr = [length, freespeed, c_e, D_e, ff_time]
        
        edges.append({'u': u, 'v': v, 'attr': attr})
        valid_links += 1
        
    print(f"Parsed {valid_links} car links.")
    return node_id_to_idx, node_coords, edges

def parse_population(pop_file, node_id_to_idx, node_coords):
    print(f"Parsing Population: {pop_file}")
    
    tree = spatial.KDTree(node_coords)
    trips = []
    
    et = ET.parse(pop_file)
    root = et.getroot()
    
    count = 0
    skipped = 0
    
    def time_to_sec(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s
        
    for person in root.findall('person'):
        car_avail = person.get('car_avail')
        if car_avail == 'never':
            skipped += 1
            continue
            
        plan = person.find("plan[@selected='yes']")
        if plan is None:
            plan = person.find('plan')
        
        if plan is None:
            continue
            
        elements = list(plan)
        current_act = None
        
        for i, el in enumerate(elements):
            if el.tag == 'act':
                current_act = el
            elif el.tag == 'leg':
                mode = el.get('mode')
                if mode != 'car':
                    continue
                    
                if i + 1 < len(elements) and elements[i+1].tag == 'act':
                    next_act = elements[i+1]
                    ox = float(current_act.get('x'))
                    oy = float(current_act.get('y'))
                    dx = float(next_act.get('x'))
                    dy = float(next_act.get('y'))
                    
                    end_time_str = current_act.get('end_time')
                    if end_time_str:
                        dep_time = time_to_sec(end_time_str)
                    else:
                        dep_time = 0 
                        
                    trips.append({
                        'ox': ox, 'oy': oy,
                        'dx': dx, 'dy': dy,
                        'dep_time': dep_time
                    })
                    
        count += 1
        
    print(f"Parsed {count} persons. Skipped {skipped} (no car). Total trips: {len(trips)}")
    
    if not trips:
        return [], []
        
    trips_ox = [t['ox'] for t in trips]
    trips_oy = [t['oy'] for t in trips]
    trips_dx = [t['dx'] for t in trips]
    trips_dy = [t['dy'] for t in trips]
    
    origins_coords = np.column_stack((trips_ox, trips_oy))
    dests_coords = np.column_stack((trips_dx, trips_dy))
    
    print("Mapping coordinates to nearest nodes...")
    _, origin_indices = tree.query(origins_coords)
    _, dest_indices = tree.query(dests_coords)
    
    trip_data = []
    for i in range(len(trips)):
        trip_data.append({
            'u': origin_indices[i],
            'v': dest_indices[i],
            'dep_time': trips[i]['dep_time']
        })
        
    return trip_data

def run_benchmark():
    base_dir = "tamarl/data/scenarios/sioux_falls"
    net_file = os.path.join(base_dir, "Siouxfalls_network_PT.xml")
    pop_file = os.path.join(base_dir, "Siouxfalls_population.xml")
    
    # 1. Parse Data
    node_map, node_coords, edges_data = parse_network(net_file)
    trips = parse_population(pop_file, node_map, node_coords)
    
    if len(trips) == 0:
        print("No trips found. Exiting.")
        return
        
    num_nodes = len(node_map)
    
    # 2. Build Graph for Dijkstra
    uv_to_edge = {}
    g_data, g_row, g_col = [], [], []
    
    edge_static_list = []
    
    for i, e in enumerate(edges_data):
        u, v = e['u'], e['v']
        w = e['attr'][4] # ff_time
        edge_static_list.append(e['attr'])
        
        if (u,v) not in uv_to_edge:
            uv_to_edge[(u,v)] = i
            g_data.append(w)
            g_row.append(u)
            g_col.append(v)
        else:
            curr_best = uv_to_edge[(u,v)]
            if w < edges_data[curr_best]['attr'][4]:
                uv_to_edge[(u,v)] = i
                # Note: This simple overwrite in uv_to_edge logic 
                # doesn't update g_data if we appended already.
                # But typically Sioux Falls links are unique or parallel.
                # We assume standard graph for now.
                pass

    graph = sp.csr_matrix((g_data, (g_row, g_col)), shape=(num_nodes, num_nodes))
    
    # 3. Pathfinding
    print(f"Running Pathfinding for {len(trips)} trips...")
    origins = [t['u'] for t in trips]
    destinations = [t['v'] for t in trips]
    
    print("Calculating All-Pairs Shortest Path...")
    dist_matrix, predecessors = csgraph.shortest_path(graph, return_predecessors=True, directed=True)
    
    # Reconstruct paths
    paths_list = []
    valid_trip_indices = []
    max_len = 0
    
    for i, (u, v) in enumerate(zip(origins, destinations)):
        if u == v: continue
        if np.isinf(dist_matrix[u, v]): continue
             
        path_nodes = []
        curr = v
        while curr != u:
             path_nodes.append(curr)
             curr = predecessors[u, curr]
             if curr == -9999: break
        path_nodes.append(u)
        path_nodes.reverse() 
        
        edge_path = []
        possible = True
        for k in range(len(path_nodes)-1):
            n1, n2 = path_nodes[k], path_nodes[k+1]
            if (n1, n2) in uv_to_edge:
                edge_path.append(uv_to_edge[(n1, n2)])
            else:
                possible = False
                break
        
        if possible and len(edge_path) > 0:
            paths_list.append(edge_path)
            valid_trip_indices.append(i)
            max_len = max(max_len, len(edge_path))
            
    print(f"Found valid paths for {len(paths_list)} / {len(trips)} trips.")
    
    # 4. Prepare DNL
    num_agents = len(paths_list)
    paths_tensor = torch.full((num_agents, max_len), -1, dtype=torch.long)
    for i, p in enumerate(paths_list):
        paths_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long)
        
    departure_times = torch.tensor([trips[i]['dep_time'] for i in valid_trip_indices], dtype=torch.long)
    edge_static = torch.tensor(edge_static_list, dtype=torch.float32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Simulation on {device}...")
    
    # Instantiate TorchDNLMATSim
    dnl = TorchDNLMATSim(
        edge_static, 
        paths_tensor, 
        device=device, 
        departure_times=departure_times, 
        dt=1.0,
        stuck_threshold=15,
        enable_profiling=True
    )
    
    # 5. Simulation Loop
    max_steps = 86400 
    en_route_counts = []
    
    sim_start = time.time()
    step = 0
    active = True
    en_route = 0
    
    print("Starting simulation loop...")
    t0 = time.time()
    while active and step < max_steps:
        dnl.step()
        step += 1
        
        # Monitor status
        if step % 100 == 0:
            # Monitor status
            # agent_state[:, 0]: 0=Wait, 1=Travel, 2=Buffer, 3=Done
            state = dnl.status
            
            # Count (1 | 2)
            en_route = ((state == 1) | (state == 2)).sum().item()
            en_route_counts.append(en_route)
            
            # Check completion
            done_mask = (state == 3)
            if done_mask.all():
                print(f"All agents finished at step {step}")
                active = False
        else:
             # Just assume active or check less frequently? 
             # For correct termination we might need to check done_mask roughly often.
             # 100 steps is fine (100 seconds).
             pass
            
        if step % 3600 == 0:
            t1 = time.time()
            dt = t1 - t0
            ms_per_step = (dt / 3600) * 1000
            print(f"Step {step}, En Route: {en_route}, Speed: {ms_per_step:.2f} ms/step")
            t0 = time.time()

    sim_end = time.time()
    print(f"Simulation done in {step} steps, {sim_end - sim_start:.2f}s")
    print("\n--- Profiling Stats ---")
    dnl.print_stats(limit=20)
    print("-----------------------")
    
    # Post-process for Plotting
    # metrics = [accum_dist, final_travel_time]
    metrics = dnl.agent_metrics.cpu().numpy()
    
    # State: [status, curr, next, ptr, arrival, stuck, start_time]
    start_steps = dnl.start_time.cpu().numpy()
    status_cpu = dnl.status.cpu().numpy()
    
    # Arrival Steps
    # For finished agents: start + travel_time
    # travel_time in metrics col 1
    travel_times = metrics[:, 1]
    arrival_steps = start_steps + travel_times
    
    # Filter for plotting:
    # We pass all steps to plot_histogram, it handles filtering.
    # But arrival_steps for unfinished agents is invalid (travel time is 0).
    # We should set invalid arrivals to -1
    # We should set invalid arrivals to -1
    unfinished = (status_cpu != 3)
    arrival_steps[unfinished] = -1
    
    print("Generating Plot...")
    plot_agent_status(start_steps, arrival_steps, max_steps=step, bucket_size_sec=300, output_file='sioux_falls_status.png')

if __name__ == "__main__":
    run_benchmark()
