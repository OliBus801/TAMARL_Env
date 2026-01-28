import os
import argparse
import xml.etree.ElementTree as ET
import torch
import numpy as np
import time
import datetime
import pandas as pd
from tamarl.core.dnl_matsim import TorchDNLMATSim
from tamarl.core.plot_histogram import plot_agent_status

def sec_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=int(seconds)))


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
        
    num_nodes = len(node_id_to_idx)
    print(f"Parsed {num_nodes} nodes.")
    
    # Parse Links
    edges = []
    links_xml = root.find('links')
    valid_links = 0
    
    scale_factor = 1.0 
    eff_cell_size = 7.5
    
    link_id_to_idx = {}
    
    for idx, link in enumerate(links_xml.findall('link')):
        modes = link.get('modes')
        if 'car' not in modes:
            continue
            
        u_id = link.get('from')
        v_id = link.get('to')
        link_id = link.get('id')
        
        # Ensure nodes exist
        if u_id not in node_id_to_idx or v_id not in node_id_to_idx:
            continue

        u = node_id_to_idx[u_id]
        v = node_id_to_idx[v_id]
        
        length = float(link.get('length'))
        freespeed = float(link.get('freespeed')) # m/s
        capacity_h = float(link.get('capacity')) # veh/h
        lanes = float(link.get('permlanes'))
        
        # Derived
        D_e = (capacity_h * scale_factor) 
        c_e = (length * lanes) / eff_cell_size 
        ff_time = length / freespeed
        
        # [length, free_flow_speed, c_e, D_e, ff_travel_time]
        attr = [length, freespeed, c_e, D_e, ff_time]
        
        edges.append({'u': u, 'v': v, 'id': link_id, 'attr': attr})
        link_id_to_idx[link_id] = valid_links
        valid_links += 1
        
    print(f"Parsed {valid_links} car links.")
    return node_id_to_idx, edges, link_id_to_idx

def parse_population(pop_file, link_id_to_idx):
    print(f"Parsing Population: {pop_file}")
    
    et = ET.parse(pop_file)
    root = et.getroot()
    
    trips = []
    trip_metadata = []
    
    count = 0
    skipped_no_path = 0
    
    def time_to_sec(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s
        
    for person in root.findall('person'):
        person_id = person.get('id')
        
        plan = person.find("plan[@selected='yes']")
        if plan is None:
            plan = person.find('plan')
        
        if plan is None:
            continue
            
        elements = list(plan)
        current_act_end_time = 0
        trip_counter = 0
        
        for i, el in enumerate(elements):
            if el.tag == 'act':
                end_time_str = el.get('end_time')
                if end_time_str:
                    current_act_end_time = time_to_sec(end_time_str)
                
            elif el.tag == 'leg':
                mode = el.get('mode')
                if mode != 'car':
                    continue
                
                route_tag = el.find('route')
                # Try to get text from route tag, handles None
                route_str = route_tag.text.strip() if (route_tag is not None and route_tag.text) else None
                
                if not route_str:
                    skipped_no_path += 1
                    continue
                    
                link_ids = route_str.split(' ')
                
                # Convert to indices
                path_indices = []
                valid_path = True
                
                # In basic MATSim, routes are link based.
                for lid in link_ids:
                    if lid in link_id_to_idx:
                        path_indices.append(link_id_to_idx[lid])
                    else:
                        # print(f"Warning: Unknown link {lid} in route")
                        valid_path = False
                        break
                
                if valid_path and len(path_indices) > 0:
                    trips.append({
                        'dep_time': current_act_end_time,
                        'path': path_indices
                    })
                    trip_metadata.append({
                        'agent_id': person_id,
                        'trip_number': trip_counter,
                        'path_str': route_str
                    })
                    trip_counter += 1
                else:
                    skipped_no_path += 1
                    
        count += 1
        
    print(f"Parsed {count} persons. Skipped {skipped_no_path} legs. Total trips: {len(trips)}")
    return trips, trip_metadata

def run_benchmark(root_folder):
    # 1. Locate files
    files = [f for f in os.listdir(root_folder) if f.endswith('.xml')]
    network_file = None
    population_file = None
    
    # Candidate lists
    net_candidates = []
    pop_candidates = []
    
    for f in files:
        lower_f = f.lower()
        if 'network' in lower_f:
            net_candidates.append(f)
        if 'population' in lower_f or 'plans' in lower_f:
            pop_candidates.append(f)
            
    # Select Network
    if net_candidates:
        # Prefer shortest name or specific logic? Just take first for now
        network_file = os.path.join(root_folder, net_candidates[0])
        
    # Select Population
    if pop_candidates:
        # Prefer one with "route" in name
        route_pops = [p for p in pop_candidates if 'route' in p.lower()]
        if route_pops:
            population_file = os.path.join(root_folder, route_pops[0])
        else:
            population_file = os.path.join(root_folder, pop_candidates[0])
            
    if not network_file:
        print(f"Error: Could not find network file (must contain 'network' and end with .xml) in {root_folder}")
        return
        
    if not population_file:
         print(f"Error: Could not find population file (must contain 'population' or 'plans') in {root_folder}")
         return
         
    print(f"Selected Network: {os.path.basename(network_file)}")
    print(f"Selected Population: {os.path.basename(population_file)}")

    # 2. Setup Output Directory
    output_dir = os.path.join(root_folder, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Parse Data
    node_map, edges_data, link_id_to_idx = parse_network(network_file)
    trips, trip_metadata = parse_population(population_file, link_id_to_idx)
    
    if len(trips) == 0:
        print("No trips found. Exiting.")
        return
        
    # 4. Prepare Data for DNL
    edge_static_list = [e['attr'] for e in edges_data]
    edge_static = torch.tensor(edge_static_list, dtype=torch.float32)
    
    departure_times = torch.tensor([t['dep_time'] for t in trips], dtype=torch.long)
    
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
    
    sim_start = time.time()
    step = 0
    active = True
    
    print("Starting simulation loop...")
    t0 = time.time()
    while active and step < max_steps:
        dnl.step()
        step += 1
            
        if step % 3600 == 0:
            t1 = time.time()
            dt = t1 - t0
            ms_per_step = (dt / 3600) * 1000
            
            # Use dnl status to get en_route count
            status = dnl.status # array on device
            en_route = ((status == 1) | (status == 2)).sum().item()
            
            print(f"Hour {step//3600} | En Route: {en_route} | Speed: {ms_per_step:.2f} ms/step | Elapsed time : {dt:.2f}s")
            t0 = time.time()
            
        # Check if all finished
        # Status 0: inactive, 1: enroute, 2: on link, 3: arrived, 4: stuck
        # Simulation is active if any agent is not arrived(3) or stuck(4) AND has started (status!=0) ?
        # Actually simplest check is if any encoded logic or manual check
        
        # Optimization: We can check less frequently or trust max_steps
        # But if we want to stop early:
        if step % 100 == 0:
             # Check if any agents are still running (status 1 or 2, or waiting to start 0)
             # Wait, status 0 means not started yet.
             # So we must continue until all agents are >= 3.
             # Note: torch.all(status >= 3)
             pass


    sim_end = time.time()
    print(f"Simulation done in {step} steps, {sim_end - sim_start:.2f}s")

    print("\n--- Profiling Stats ---")
    dnl.print_stats(limit=20)
    print("-----------------------")
    
    # 6. Post-process & Export
    
    # Metrics: [traveled_distance, travel_time]
    metrics = dnl.agent_metrics.cpu().numpy()

    print(f"--- Average Agents Metrics ---")
    print(f"Average Traveled Distance: {metrics[:, 0].mean():.2f}")
    print(f"Average Travel Time: {metrics[:, 1].mean():.2f}")
    print(f"Average Travel Speed: {metrics[:, 0].mean() / metrics[:, 1].mean():.2f}")
    print("-----------------------")
    
    # agent_metrics.csv --------------
    # "agent_id,trip_number,departure_time,travel_time,traveled_distance"
    
    dep_times_np = departure_times.cpu().numpy()
    
    df_metrics = pd.DataFrame(trip_metadata)
    
    # Format times as HH:MM:SS
    df_metrics['departure_time'] = [sec_to_hms(t) for t in dep_times_np]
    df_metrics['travel_time'] = [sec_to_hms(t) for t in metrics[:, 1]]
    
    df_metrics['traveled_distance'] = metrics[:, 0]
    
    # Reorder columns as requested
    df_metrics = df_metrics[['agent_id', 'trip_number', 'departure_time', 'travel_time', 'traveled_distance']]
    
    metrics_path = os.path.join(output_dir, "agent_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    # paths.csv --------------
    # "agent_id, trip_number, path"
    # We already have path_str in trip_metadata
    df_paths = pd.DataFrame(trip_metadata)
    df_paths.rename(columns={'path_str': 'path'}, inplace=True)
    df_paths = df_paths[['agent_id', 'trip_number', 'path']]
    
    paths_out_path = os.path.join(output_dir, "paths.csv")
    df_paths.to_csv(paths_out_path, index=False)
    print(f"Saved paths to {paths_out_path}")

    # average_metrics.csv --------------
    # "avg_trav_dist, avg_trav_time, avg_trav_speed"
    avg_metrics = pd.DataFrame({
        'avg_trav_dist': [metrics[:, 0].mean()],
        'avg_trav_time': [metrics[:, 1].mean()],
        'avg_trav_speed': [metrics[:, 0].mean() / metrics[:, 1].mean()]
    })
    
    avg_metrics.to_csv(os.path.join(output_dir, "average_metrics.csv"), index=False)
    print(f"Saved average metrics to {os.path.join(output_dir, "average_metrics.csv")}")
    
    # Plot --------------
    start_steps = dnl.start_time.cpu().numpy()
    status_cpu = dnl.status.cpu().numpy()
    
    travel_times = metrics[:, 1]
    arrival_steps = start_steps + travel_times
    
    unfinished = (status_cpu != 3)
    arrival_steps[unfinished] = -1
    
    plot_file = os.path.join(output_dir, "agent_status.png")
    print("Generating Plot...")
    plot_agent_status(start_steps, arrival_steps, max_steps=step, bucket_size_sec=300, output_file=plot_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", help="Root folder containing network.xml and population.xml")
    args = parser.parse_args()
    
    if not os.path.exists(args.root_folder):
        print(f"Root folder not found: {args.root_folder}")
    else:
        run_benchmark(args.root_folder)
