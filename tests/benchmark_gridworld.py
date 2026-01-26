import argparse
import xml.etree.ElementTree as ET
import torch
import numpy as np
import time
import tracemalloc
import sys
import psutil
import os
from tamarl.core.dnl_matsim import TorchDNLMATSim

def parse_network(network_file):
    print(f"Parsing Network: {network_file}")
    tree = ET.parse(network_file)
    root = tree.getroot()
    
    # Parse Nodes
    node_id_to_idx = {}
    node_coords = []
    
    nodes_xml = root.find('nodes')
    if nodes_xml is not None:
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
    
    eff_cell_size = 7.5
    
    if links_xml is not None:
        for link in links_xml.findall('link'):
            modes = link.get('modes')
            if 'car' not in modes:
                continue
                
            u_id = link.get('from')
            v_id = link.get('to')
            
            if u_id not in node_id_to_idx or v_id not in node_id_to_idx:
                continue

            u = node_id_to_idx[u_id]
            v = node_id_to_idx[v_id]
            
            length = float(link.get('length'))
            freespeed = float(link.get('freespeed')) # m/s
            capacity_h = float(link.get('capacity')) # veh/h
            lanes = float(link.get('permlanes'))
            
            # Derived
            D_e = capacity_h  # Flow capacity (veh/h)
            c_e = (length * lanes) / eff_cell_size  # Storage capacity (veh)
            ff_time = length / freespeed
            
            # [length, free_flow_speed, c_e, D_e, ff_travel_time]
            attr = [length, freespeed, c_e, D_e, ff_time]
            
            edges.append({'u': u, 'v': v, 'id': link.get('id'), 'attr': attr})
            valid_links += 1
        
    print(f"Parsed {valid_links} car links.")
    return node_id_to_idx, edges

def parse_population(pop_file, node_id_to_idx):
    print(f"Parsing Population: {pop_file}")
    
    trips = []
    et = ET.parse(pop_file)
    root = et.getroot()
    
    count = 0
    
    def time_to_sec(t_str):
        parts = list(map(int, t_str.split(':')))
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
        
    for person in root.findall('person'):
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
                    end_time_str = current_act.get('end_time')
                    if end_time_str:
                        dep_time = time_to_sec(end_time_str)
                    else:
                        dep_time = 0 
                        
                    route_text = None
                    route_elem = el.find('route')
                    if route_elem is not None:
                        route_text = route_elem.text
                        
                    trips.append({
                        'dep_time': dep_time,
                        'route_str': route_text
                    })
                    
        count += 1
        
    print(f"Parsed {count} persons. Total car trips: {len(trips)}")
    return trips

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 # MB

def run_simulation(network_file, population_file):
    tracemalloc.start()
    
    # 1. Parse Data
    node_map, edges_data = parse_network(network_file)
    trips = parse_population(population_file, node_map)
    
    if not trips:
        print("No trips. Exiting.")
        return

    num_nodes = len(node_map)
    
    # 2. Build Edge Lookup (Link ID -> Index)
    # edges_data is list of {'id': link_id, 'attr': ...}
    # We need to map link_id to index in edges_data list (which corresponds to tensor index)
    
    link_id_to_idx = {}
    edge_static_list = []
    
    for i, e in enumerate(edges_data):
        lid = e['id']
        link_id_to_idx[lid] = i
        edge_static_list.append(e['attr'])

    # 3. Process Paths (Direct Link ID Lookup)
    print("Processing Pre-calculated Paths (Link-Based)...")
    
    paths_list = []
    valid_trip_indices = []
    max_path_len = 0
    
    missing_routes = 0
    invalid_edges = 0
    
    t_process_start = time.time()
    
    for i, trip in enumerate(trips):
        route_str = trip.get('route_str')
        if not route_str:
            missing_routes += 1
            continue
            
        link_ids = route_str.strip().split(' ')
        if len(link_ids) == 0:
            continue
            
        edge_path = []
        possible = True
        
        try:
            edge_path = [link_id_to_idx[lid] for lid in link_ids]
        except KeyError:
            possible = False
            invalid_edges += 1
        
        if possible and len(edge_path) > 0:
            paths_list.append(edge_path)
            valid_trip_indices.append(i)
            max_path_len = max(max_path_len, len(edge_path))
            
    print(f"Path Processing Complete. Time: {time.time() - t_process_start:.2f}s")
    print(f"Valid Agents: {len(paths_list)} / {len(trips)}")
    
    if len(paths_list) == 0:
        print("No valid paths found.")
        return

    # 4. Init Simulation
    edge_static = torch.tensor(edge_static_list, dtype=torch.float32)
    departure_times = torch.tensor([trips[i]['dep_time'] for i in valid_trip_indices], dtype=torch.long)
    
    num_agents = len(paths_list)
    paths_tensor = torch.full((num_agents, max_path_len), -1, dtype=torch.long)
    for i, p in enumerate(paths_list):
        paths_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    tracemalloc.clear_traces()
    
    t_init_start = time.time()
    
    dnl = TorchDNLMATSim(
        edge_static,
        paths_tensor,
        departure_times=departure_times,
        device=device,
        enable_profiling=False
    )
    
    # Force initialize CUDA context if needed by running a tiny dummy op or step?
    # dnl init already does allocations.
    
    print(f"Initialization Time: {time.time() - t_init_start:.2f}s")
    
    # 5. Run Loop
    max_steps = 86400
    print(f"Starting Simulation for {max_steps} steps...")
    
    t_sim_start = time.time()
    step_fn = dnl.step
    
    for s in range(max_steps):
        step_fn()
        
    t_sim_end = time.time()
    total_time = t_sim_end - t_sim_start
    
    # Metrics
    ram_peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    sys_ram_peak = get_memory_usage()
    
    vram_peak = 0
    if device == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        
    print("\n--- RESULTS ---")
    print(f"Initialization Time: {t_sim_start - t_init_start:.4f} s")
    print(f"Simulation Time: {total_time:.4f} s")
    print(f"Total Wall-clock Time: {(time.time() - t_init_start):.4f} s")
    print(f"Steps: {max_steps}")
    print(f"Avg Time per Step: {(total_time/max_steps)*1000:.4f} ms")
    print(f"Peak RAM (tracemalloc): {ram_peak:.2f} MB")
    print(f"Peak RSS (System): {sys_ram_peak:.2f} MB")
    if device == 'cuda':
        print(f"Peak VRAM: {vram_peak:.2f} MB")
        
    print("---------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help="Path to network XML")
    parser.add_argument("population", help="Path to population XML")
    args = parser.parse_args()
    
    if not os.path.exists(args.network):
        print(f"Network file not found: {args.network}")
        sys.exit(1)
    if not os.path.exists(args.population):
        print(f"Population file not found: {args.population}")
        sys.exit(1)
        
    run_simulation(args.network, args.population)
