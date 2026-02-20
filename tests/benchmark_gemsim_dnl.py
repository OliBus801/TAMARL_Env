import os
import argparse
import xml.etree.ElementTree as ET
import torch
import numpy as np
import time
import sys
import psutil
import datetime
import pandas as pd
import pickle
import gc
from tamarl.core.dnl_gemsim import TorchDNLGEMSim
from tamarl.core.plot_histogram import plot_agent_status

def sec_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=int(seconds)))

def parse_network(network_file, scale_factor=1.0, timestep=1.0):
    print(f"Parsing Network: {network_file}")
    node_id_to_idx = {}
    edges = []
    link_id_to_idx = {}
    valid_links = 0
    
    # New: Track incoming links for each node
    # node_idx -> list of incoming link indices
    node_incoming = {} 
    
    context = ET.iterparse(network_file, events=("end",))
     
    eff_cell_size = 7.5
    
    for event, elem in context:
        if elem.tag == "node":
            nid = elem.get('id')
            if nid not in node_id_to_idx:
                idx = len(node_id_to_idx)
                node_id_to_idx[nid] = idx
                node_incoming[idx] = []
            elem.clear()
            
        elif elem.tag == "link":
            modes = elem.get('modes')
            if 'car' in modes:
                u_id = elem.get('from')
                v_id = elem.get('to')
                link_id = elem.get('id')
                
                # Ensure nodes exist (sometimes links define nodes not yet seen?)
                # Usually nodes come first in MATSim XML, but to be safe:
                if u_id not in node_id_to_idx:
                    node_id_to_idx[u_id] = len(node_id_to_idx)
                    node_incoming[node_id_to_idx[u_id]] = []
                if v_id not in node_id_to_idx:
                    node_id_to_idx[v_id] = len(node_id_to_idx)
                    node_incoming[node_id_to_idx[v_id]] = []

                if u_id in node_id_to_idx and v_id in node_id_to_idx:
                    u = node_id_to_idx[u_id]
                    v = node_id_to_idx[v_id]
                    
                    length = float(elem.get('length')) # meters
                    freespeed = float(elem.get('freespeed')) # m/s
                    capacity_h = float(elem.get('capacity')) # veh/h
                    lanes = float(elem.get('permlanes'))
                    
                    # Calculate Flow Capacity (maximum Inflow per timestep)
                    unscaledFlowCapacity_s = (capacity_h / 3600) 
                    D_e = unscaledFlowCapacity_s * timestep * scale_factor

                    # Calculate Storage Capacity
                    c_e = ((length * lanes) / eff_cell_size) * scale_factor 
                    c_e = max(c_e, D_e)

                    ff_time = length / freespeed 
                    temp_spaceCap = ff_time * unscaledFlowCapacity_s * scale_factor
                    c_e = max(c_e, temp_spaceCap)

                    attr = [length, freespeed, c_e, D_e, ff_time]
                    
                    edges.append({'u': u, 'v': v, 'id': link_id, 'attr': attr})
                    link_idx = valid_links
                    link_id_to_idx[link_id] = link_idx
                    valid_links += 1
                    
                    # Add to node incoming list
                    # This link 'link_idx' goes u -> v
                    # So it is incoming to 'v'
                    # Wait, for DNL_GEMSim "process nodes", we need UPSTREAM links of a node?
                    # Yes, to pull agents from their out_queue.
                    # Upstream of 'v' is 'link_idx'.
                    node_incoming[v].append(link_idx)
            
            elem.clear()
            
    print(f"Parsed {len(node_id_to_idx)} nodes and {valid_links} car links.")
    return node_id_to_idx, edges, link_id_to_idx, node_incoming

def parse_population(pop_file, link_id_to_idx):
    print(f"Parsing Population: {pop_file}")
    
    trips = []
    trip_metadata = []
    
    count = 0
    skipped_no_path = 0

    context = ET.iterparse(pop_file, events=("end",))
    
    def time_to_sec(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s
        
    for event, elem in context:
        if elem.tag == "person":
            person_id = elem.get('id')
            selected_plan = None
            for child in elem:
                if child.tag == 'plan':
                    if child.get('selected') == 'yes':
                        selected_plan = child
                        break
                    if selected_plan is None:
                        selected_plan = child
            
            if selected_plan is not None:
                current_act_end_time = 0
                trip_counter = 0
                
                for el in selected_plan:
                    if el.tag in ['act', 'activity']:
                        end_time_str = el.get('end_time')
                        if end_time_str:
                            current_act_end_time = time_to_sec(end_time_str)
                        else:
                            max_dur = el.get('max_dur')
                            if max_dur:
                                current_act_end_time += time_to_sec(max_dur)
                    
                    elif el.tag == 'leg':
                        mode = el.get('mode')
                        if mode == 'car':
                            route_tag = el.find('route')
                            route_str = route_tag.text.strip() if (route_tag is not None and route_tag.text) else None
                            
                            if route_str:
                                link_ids = route_str.split(' ')
                                path_indices = []
                                valid_path = True
                                
                                for lid in link_ids:
                                    if lid in link_id_to_idx:
                                        path_indices.append(link_id_to_idx[lid])
                                    else:
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
                            else:
                                skipped_no_path += 1
            
            count += 1
            elem.clear() 
            
    print(f"Parsed {count} persons. Skipped {skipped_no_path} legs. Total trips: {len(trips)}")
    return trips, trip_metadata

def run_benchmark(root_folder, population_filter=None, timestep=1.0, scale_factor=1.0, n_hours=24, save_pickle=False, load_pickle=True, save_paths=False, save_agents=False, output_folder="output_gemsim"):
    
    process = psutil.Process(os.getpid())
    def get_mem():
        return process.memory_info().rss / 1024 / 1024

    initial_mem = get_mem()

    # 1. Locate files
    files = [f for f in os.listdir(root_folder) if f.endswith('.xml')]
    network_file = None
    population_file = None
    
    net_candidates = []
    pop_candidates = []
    
    for f in files:
        lower_f = f.lower()
        if 'network' in lower_f:
            net_candidates.append(f)
        if 'population' in lower_f or 'plans' in lower_f:
            pop_candidates.append(f)
            
    if net_candidates:
        network_file = os.path.join(root_folder, net_candidates[0])
        
    if pop_candidates:
        if population_filter:
            tokens_candidates = []
            for p in pop_candidates:
                tokens = p.replace('.', '_').split('_')
                if population_filter in tokens:
                    tokens_candidates.append(p)
            
            if tokens_candidates:
                pop_candidates = tokens_candidates
            else:
                pop_candidates = [p for p in pop_candidates if population_filter in p]

            if not pop_candidates:
                print(f"Error: No population file found matching '{population_filter}' in {root_folder}")
                return

        route_pops = [p for p in pop_candidates if 'route' in p.lower()]
        if route_pops:
            population_file = os.path.join(root_folder, route_pops[0])
        else:
            population_file = os.path.join(root_folder, pop_candidates[0])
            
    if not network_file:
        print(f"Error: Could not find network file.")
        return
    if not population_file:
         print(f"Error: Could not find population file.")
         return
         
    print(f"Selected Network: {os.path.basename(network_file)}")
    print(f"Selected Population: {os.path.basename(population_file)}")

    output_dir = os.path.join(root_folder, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Parse/Load Data
    
    # --- Network ---
    network_pkl = network_file.replace('.xml', '_gemsim.pkl') # Use different pickle for gemsim (has topology)
    network_loaded = False
    
    if load_pickle and os.path.exists(network_pkl):
        print(f"Loading cached network from {network_pkl}...")
        try:
            with open(network_pkl, 'rb') as f:
                node_map, edges_data, link_id_to_idx, node_incoming = pickle.load(f)
            network_loaded = True
            print("Loaded network from pickle.")
        except Exception as e:
            print(f"Failed to load network pickle: {e}. Reparsing...")
            
    if not network_loaded:
        t_start_net = time.time()
        node_map, edges_data, link_id_to_idx, node_incoming = parse_network(network_file, scale_factor, timestep)
        if save_pickle:
             print(f"Saving network to {network_pkl}...")
             with open(network_pkl, 'wb') as f:
                 pickle.dump((node_map, edges_data, link_id_to_idx, node_incoming), f)
    
    # --- Population ---
    pop_basename = os.path.basename(population_file)
    pop_pkl = os.path.join(os.path.dirname(population_file), pop_basename.rsplit('.', 1)[0] + '.pkl')
    pop_loaded = False
    
    if load_pickle and os.path.exists(pop_pkl):
        print(f"Loading cached population from {pop_pkl}...")
        try:
            with open(pop_pkl, 'rb') as f:
                trips, trip_metadata = pickle.load(f)
            pop_loaded = True
            print("Loaded population from pickle.")
        except Exception as e:
            print(f"Failed to load population pickle.")
            
    if not pop_loaded:
        trips, trip_metadata = parse_population(population_file, link_id_to_idx)
        if save_pickle:
            print(f"Saving population to {pop_pkl}...")
            with open(pop_pkl, 'wb') as f:
                pickle.dump((trips, trip_metadata), f)

    if len(trips) == 0:
        print("No trips found. Exiting.")
        return
        
    del link_id_to_idx
    gc.collect()

    # 4. Prepare Data for DNL_GEMSim
    edge_static_list = [e['attr'] for e in edges_data]
    edge_static = torch.tensor(edge_static_list, dtype=torch.float32)
    
    del edges_data
    gc.collect()
    
    departure_times = torch.tensor([t['dep_time'] for t in trips], dtype=torch.int32)
    
    lengths = [len(t['path']) for t in trips]
    max_len = max(lengths)
    num_agents = len(trips)
    
    print(f"Packing {num_agents} paths (max len {max_len})...")
    paths_tensor = torch.full((num_agents, max_len), -1, dtype=torch.int32)
    
    for i, t in enumerate(trips):
        p = t['path']
        paths_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.int32)
    
    del trips
    gc.collect()

    # Create Topology Tensor for GEMSim
    # node_incoming is dict: node_idx -> [link_idx...]
    # We need flattened CSR: indices, offsets
    max_node_idx = max(node_incoming.keys()) if node_incoming else -1
    # Ensure all nodes up to max exist in dict
    # (Should be guaranteed by parse_network logic?)
    
    print("Building topology CSR...")
    
    # Sort keys to ensure ordering
    sorted_node_indices = list(range(max_node_idx + 1))
    
    flat_indices = []
    offsets = [0]
    
    for n_idx in sorted_node_indices:
        links = node_incoming.get(n_idx, [])
        flat_indices.extend(links)
        offsets.append(len(flat_indices))
        
    topology = (torch.tensor(flat_indices, dtype=torch.int32), torch.tensor(offsets, dtype=torch.int32))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Simulation on {device}...")
    
    dnl = TorchDNLGEMSim(
        edge_static, 
        paths_tensor, 
        net_topology=topology,
        device=device, 
        departure_times=departure_times, 
        dt=timestep, 
        stuck_threshold=10,
        enable_profiling=True
    )
    
    # 5. Simulation Loop
    max_steps = n_hours * 3600 
    
    sim_start = time.time()
    step = 0
    active = True
    
    print("Starting simulation loop (GEMSim)...")
    t0 = time.time()
    while active and step < max_steps:
        dnl.step()
        step += 1
            
        if step % 3600 == 0:
            t1 = time.time()
            dt = t1 - t0
            ms_per_step = (dt / 3600) * 1000
            
            status = dnl.status
            en_route = ((status == 1) | (status == 2)).sum().item()

            print(f"Hour {step//3600} | En Route: {en_route} | Speed: {ms_per_step:.2f} ms/step | Elapsed: {dt:.2f}s")
            t0 = time.time()

    sim_end = time.time()
    print(f"Simulation done in {step} steps, {sim_end - sim_start:.2f}s")

    compute_time, peak_mem, peak_vram = dnl.print_stats(limit=20)

    # 6. Post-process & Export
    metrics = dnl.agent_metrics.cpu().numpy()

    print(f"--- Average Agents Metrics ---")
    print(f"Average Traveled Distance: {metrics[:, 0].mean():.2f}")
    if metrics[:, 1].mean() > 0:
        print(f"Average Travel Time: {metrics[:, 1].mean():.2f}")
        print(f"Average Travel Speed: {metrics[:, 0].mean() / metrics[:, 1].mean():.2f}")
    else:
        print("Average Travel Time: 0.00")
    print("-----------------------")
    
    # Save results similar to matsim benchmark
    avg_trav_time = metrics[:, 1].mean()
    avg_speed = metrics[:, 0].mean() / avg_trav_time if avg_trav_time > 0 else 0
    
    avg_metrics = pd.DataFrame({
        'avg_trav_dist': [metrics[:, 0].mean()],
        'avg_trav_time': [avg_trav_time],
        'avg_trav_speed': [avg_speed],
        'compute_time': [compute_time],
        'peak_memory': [peak_mem],
        'peak_vram': [peak_vram],
    })
    
    avg_metrics.to_csv(os.path.join(output_dir, f"{population_filter}_average_metrics.csv"), index=False)
    print(f"Saved average metrics to {os.path.join(output_dir, f'{population_filter}_average_metrics.csv')}")

    # 7. Plot Agent Status
    try:
        start_steps = dnl.start_time.cpu().numpy()
        durations = metrics[:, 1] # already numpy
        status = dnl.status.cpu().numpy()
        
        arrival_steps = np.zeros_like(start_steps)
        mask_done = (status == 3)
        arrival_steps[mask_done] = start_steps[mask_done] + durations[mask_done]
        
        plot_file = os.path.join(output_dir, f"{population_filter}_agent_status.png")
        plot_agent_status(start_steps, arrival_steps, max_steps, bucket_size_sec=300, output_file=plot_file)
    except Exception as e:
        print(f"Failed to plot agent status: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", help="Root folder containing network.xml and population.xml")
    parser.add_argument("--population", help="Substring to match for population file (e.g. '100000' or '1pct')", default=None)
    parser.add_argument("--save_pickle", help="If set, saves parsed network and population to .pkl files for faster loading next time.", action="store_true")
    parser.add_argument("--no_load_pickle", help="Force reparsing from XML even if pickle exists.", action="store_true")
    parser.add_argument("--save_paths", help="If set, exports paths.csv containing agent paths.", action="store_true")
    parser.add_argument("--save_agents", help="If set, exports agent_metrics.csv containing individual agent statistics.", action="store_true")
    parser.add_argument("--n_hours", help="Sim hours", default=24)
    parser.add_argument("--scale_factor", help="Scale factor", default=1.0)
    parser.add_argument("--timestep", help="Time step size", default=1)
    parser.add_argument("--output_folder", help="Output folder", default="output_gemsim")

    args = parser.parse_args()
    
    if not os.path.exists(args.root_folder):
        print(f"Root folder not found: {args.root_folder}")
    else:
        run_benchmark(args.root_folder, args.population, timestep=float(args.timestep), scale_factor=float(args.scale_factor), n_hours=int(args.n_hours), save_pickle=args.save_pickle, load_pickle=not args.no_load_pickle, save_paths=args.save_paths, save_agents=args.save_agents, output_folder=args.output_folder)
