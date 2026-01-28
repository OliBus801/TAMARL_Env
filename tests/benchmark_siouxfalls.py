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
    
    link_id_to_idx = {}
    
    for idx, link in enumerate(links_xml.findall('link')):
        modes = link.get('modes')
        if 'car' not in modes:
            continue
            
        u_id = link.get('from')
        v_id = link.get('to')
        link_id = link.get('id')
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
        link_id_to_idx[link_id] = valid_links
        valid_links += 1
        
    print(f"Parsed {valid_links} car links.")
    return node_id_to_idx, node_coords, edges, link_id_to_idx

def parse_population(pop_file, link_id_to_idx):
    print(f"Parsing Population: {pop_file}")
    
    et = ET.parse(pop_file)
    root = et.getroot()
    
    trips = []
    count = 0
    skipped_no_car = 0
    skipped_no_path = 0
    
    def time_to_sec(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s
        
    for person in root.findall('person'):
        car_avail = person.get('car_avail')
        # Some scenarios don't set car_avail="never", check modes later
        
        plan = person.find("plan[@selected='yes']")
        if plan is None:
            plan = person.find('plan')
        
        if plan is None:
            continue
            
        elements = list(plan)
        current_act_end_time = 0
        
        for i, el in enumerate(elements):
            if el.tag == 'act':
                end_time_str = el.get('end_time')
                if end_time_str:
                    current_act_end_time = time_to_sec(end_time_str)
                # Else it keeps previous time or 0
                
            elif el.tag == 'leg':
                mode = el.get('mode')
                if mode != 'car':
                    continue
                
                route_tag = el.find('route')
                if route_tag is None or not route_tag.text:
                    skipped_no_path += 1
                    continue
                    
                route_str = route_tag.text.strip()
                link_ids = route_str.split(' ')
                
                # Convert to indices
                path_indices = []
                valid_path = True
                for lid in link_ids:
                    if lid in link_id_to_idx:
                        path_indices.append(link_id_to_idx[lid])
                    else:
                        print(f"Warning: Unknown link {lid} in route for agent {person.get('id')}")
                        valid_path = False
                        break
                
                if valid_path and len(path_indices) > 0:
                    trips.append({
                        'dep_time': current_act_end_time,
                        'path': path_indices
                    })
                else:
                    skipped_no_path += 1
                    
        count += 1
        
    print(f"Parsed {count} persons. Skipped {skipped_no_path} legs (no route/bad link). Total trips: {len(trips)}")
    return trips

def run_benchmark():
    base_dir = "tamarl/data/scenarios/sioux_falls"
    net_file = os.path.join(base_dir, "Siouxfalls_network_PT.xml")
    pop_file = os.path.join(base_dir, "Siouxfalls_route_population.xml")
    
    # 1. Parse Data
    node_map, node_coords, edges_data, link_id_to_idx = parse_network(net_file)
    trips = parse_population(pop_file, link_id_to_idx)
    
    if len(trips) == 0:
        print("No trips found. Exiting.")
        return
        
    # 2. Prepare Data for DNL (No Dijkstra needed)
    edge_static_list = [e['attr'] for e in edges_data]
    edge_static = torch.tensor(edge_static_list, dtype=torch.float32)
    
    departure_times = torch.tensor([t['dep_time'] for t in trips], dtype=torch.long)
    
    # Pack paths into tensor
    lengths = [len(t['path']) for t in trips]
    max_len = max(lengths)
    num_agents = len(trips)
    
    print(f"Packing {num_agents} paths (max len {max_len})...")
    paths_tensor = torch.full((num_agents, max_len), -1, dtype=torch.long)
    
    for i, t in enumerate(trips):
        p = t['path']
        paths_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Simulation on {device}...")
    
    # Instantiate TorchDNLMATSim
    dnl = TorchDNLMATSim(
        edge_static, 
        paths_tensor, 
        device=device, 
        departure_times=departure_times, 
        dt=1.0, 
        stuck_threshold=10,
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
            
        if step % 3600 == 0:
            t1 = time.time()
            dt = t1 - t0
            ms_per_step = (dt / 3600) * 1000
            print(f"Hour {step//3600} | En Route: {en_route} | Speed: {ms_per_step:.2f} ms/step | Elapsed time : {dt:.2f}s")
            t0 = time.time()

    sim_end = time.time()
    print(f"Simulation done in {step} steps, {sim_end - sim_start:.2f}s")
    print("\n--- Profiling Stats ---")
    dnl.print_stats(limit=20)
    print("-----------------------")
    
    # Post-process for Plotting
    # metrics = [accum_dist, final_travel_time]
    metrics = dnl.agent_metrics.cpu().numpy()

    print(f"--- Average Agents Metrics ---")
    print(f"Average Traveled Distance: {metrics[:, 0].mean():.2f}")
    print(f"Average Travel Time: {metrics[:, 1].mean():.2f}")
    print(f"Average Travel Speed: {metrics[:, 0].mean() / metrics[:, 1].mean():.2f}")
    print("-----------------------")
    
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
