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
from tamarl.core.dnl_matsim import TorchDNLMATSim
from tamarl.core.plot_histogram import plot_agent_status

def sec_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=int(seconds)))


def parse_network(network_file):
    print(f"Parsing Network: {network_file}")
    
    # Parse Nodes
    node_id_to_idx = {}
    
    # Context iterator for streaming parsing
    context = ET.iterparse(network_file, events=("start", "end"))
    context = iter(context)
    event, root = next(context) # Get root element

    num_nodes = 0
    
    for event, elem in context:
        if event == "end" and elem.tag == "node":
            nid = elem.get('id')
            # x = float(elem.get('x')) # Not storing coords for now to save memory if not needed for simulation
            # y = float(elem.get('y'))
            node_id_to_idx[nid] = num_nodes
            num_nodes += 1
            root.clear() # Clear memory
        
        elif event == "end" and elem.tag == "nodes":
             # Finished nodes
             print(f"Parsed {num_nodes} nodes.")
             root.clear()
             
    # Reset iterator for links (iterparse assumes sequential, but 'links' comes after 'nodes' usually)
    # If the file structure allows, we can continue. MATSim XML has nodes then links.
    # We need to restart or ensure we didn't miss 'links' start if it was nested? 
    # Actually, in MATSim, <nodes> and <links> are siblings under <network>.
    # So the previous loop would exit after </nodes>. We continue to find <links>.
    
    # Re-open or continue? 
    # Iterparse consumes the file. If we broke out, we might have consumed start of links?
    # Actually, the above loop processes until end of file normally unless we break.
    # To process both strictly, we should just loop once.

    # Re-implementing as single pass
    return parse_network_single_pass(network_file)

def parse_network_single_pass(network_file):
    print(f"Parsing Network: {network_file}")
    node_id_to_idx = {}
    edges = []
    link_id_to_idx = {}
    valid_links = 0
    
    context = ET.iterparse(network_file, events=("end",))
    
    scale_factor = 1.0 
    eff_cell_size = 7.5
    
    for event, elem in context:
        if elem.tag == "node":
            nid = elem.get('id')
            node_id_to_idx[nid] = len(node_id_to_idx)
            elem.clear()
            
        elif elem.tag == "link":
            modes = elem.get('modes')
            if 'car' in modes:
                u_id = elem.get('from')
                v_id = elem.get('to')
                link_id = elem.get('id')
                
                if u_id in node_id_to_idx and v_id in node_id_to_idx:
                    u = node_id_to_idx[u_id]
                    v = node_id_to_idx[v_id]
                    
                    length = float(elem.get('length'))
                    freespeed = float(elem.get('freespeed')) # m/s
                    capacity_h = float(elem.get('capacity')) # veh/h
                    lanes = float(elem.get('permlanes'))
                    
                    D_e = (capacity_h * scale_factor) 
                    c_e = (length * lanes) / eff_cell_size 
                    ff_time = length / freespeed
                    
                    attr = [length, freespeed, c_e, D_e, ff_time]
                    
                    edges.append({'u': u, 'v': v, 'id': link_id, 'attr': attr})
                    link_id_to_idx[link_id] = valid_links
                    valid_links += 1
            
            elem.clear()
            
    print(f"Parsed {len(node_id_to_idx)} nodes and {valid_links} car links.")
    return node_id_to_idx, edges, link_id_to_idx

# Alias for compatibility
parse_network = parse_network_single_pass

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
            
            # Simple assumption: first selected plan or first plan
            selected_plan = None
            
            # Since we get "person" END event, all children (plan) are processed.
            # We iterate over children of 'person' element
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
                    if el.tag == 'act':
                        end_time_str = el.get('end_time')
                        if end_time_str:
                            current_act_end_time = time_to_sec(end_time_str)
                    
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
            elem.clear() # Clear person element from memory
            
    print(f"Parsed {count} persons. Skipped {skipped_no_path} legs. Total trips: {len(trips)}")
    return trips, trip_metadata

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 # MB

def run_benchmark(root_folder, population_filter=None, save_pickle=False, load_pickle=True, save_paths=False, save_agents=False):
    
    process = psutil.Process(os.getpid())
    def get_mem():
        return process.memory_info().rss / 1024 / 1024

    initial_mem = get_mem()

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
        network_file = os.path.join(root_folder, net_candidates[0])
        
    # Select Population
    if pop_candidates:
        if population_filter:
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
    
    # 3. Parse/Load Data
    
    # --- Network ---
    network_pkl = network_file.replace('.xml', '.pkl')
    network_loaded = False
    
    if load_pickle and os.path.exists(network_pkl):
        print(f"Loading cached network from {network_pkl}...")
        try:
            with open(network_pkl, 'rb') as f:
                node_map, edges_data, link_id_to_idx = pickle.load(f)
            network_loaded = True
            print("Loaded network from pickle.")
        except Exception as e:
            print(f"Failed to load network pickle: {e}. Reparsing...")

    if not network_loaded:
        t_start_net = time.time()
        node_map, edges_data, link_id_to_idx = parse_network(network_file)
        if save_pickle:
             print(f"Saving network to {network_pkl}...")
             with open(network_pkl, 'wb') as f:
                 pickle.dump((node_map, edges_data, link_id_to_idx), f)
    
    mem_net = get_mem()
    print(f"Peak Memory after Network Processing: {mem_net:.2f} MB")
    
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
            print(f"Failed to load population pickle: {e}. Reparsing...")
            
    if not pop_loaded:
        trips, trip_metadata = parse_population(population_file, link_id_to_idx)
        if save_pickle:
            print(f"Saving population to {pop_pkl}...")
            # We might want to avoid pickling trip_metadata if it's huge and not needed for simulation proper, 
            # but usually it is needed for output.
            with open(pop_pkl, 'wb') as f:
                pickle.dump((trips, trip_metadata), f)

    mem_pop = get_mem()
    print(f"Peak Memory after Population Processing: {mem_pop:.2f} MB")
    
    if len(trips) == 0:
        print("No trips found. Exiting.")
        return
        
    # Free memory of raw data if possible?
    # We need edges_data and trips for tensor creation.
    # link_id_to_idx is not needed afterwards if not used. 
    del link_id_to_idx
    gc.collect()

    # 4. Prepare Data for DNL
    edge_static_list = [e['attr'] for e in edges_data]
    edge_static = torch.tensor(edge_static_list, dtype=torch.float32)
    
    # Free edges_data
    del edges_data
    gc.collect()
    
    departure_times = torch.tensor([t['dep_time'] for t in trips], dtype=torch.long)
    
    lengths = [len(t['path']) for t in trips]
    max_len = max(lengths)
    num_agents = len(trips)
    
    print(f"Packing {num_agents} paths (max len {max_len})...")
    paths_tensor = torch.full((num_agents, max_len), -1, dtype=torch.long)
    
    for i, t in enumerate(trips):
        p = t['path']
        paths_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    
    # We can free 'trips' list now if we don't need it later?
    # We only need trip_metadata for final CSVs.
    del trips
    gc.collect()

    # Object Sizes
    edge_static_size = edge_static.element_size() * edge_static.nelement() / 1024 / 1024
    paths_tensor_size = paths_tensor.element_size() * paths_tensor.nelement() / 1024 / 1024
    
    # Estimate metadata size
    trips_size = sys.getsizeof(trip_metadata) / 1024 / 1024 

    print("\n--- 📦 Object Sizes ---")
    print(f"🛣️  Network (edge_static): {edge_static_size:.2f} MB")
    print(f"👥 Population (metadata): {trips_size:.2f} MB")
    print(f"🔢 Population (paths_tensor): {paths_tensor_size:.2f} MB")
    print("-----------------------\n")

    
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
            
        if step % 100 == 0:
             # Check completion logic if needed
             pass

    sim_end = time.time()
    print(f"Simulation done in {step} steps, {sim_end - sim_start:.2f}s")

    compute_time, peak_mem, peak_vram = dnl.print_stats(limit=20)

    # 6. Post-process & Export
    
    # Metrics: [traveled_distance, travel_time]
    metrics = dnl.agent_metrics.cpu().numpy()

    print(f"--- Average Agents Metrics ---")
    print(f"Average Traveled Distance: {metrics[:, 0].mean():.2f}")
    if metrics[:, 1].mean() > 0:
        print(f"Average Travel Time: {metrics[:, 1].mean():.2f}")
        print(f"Average Travel Speed: {metrics[:, 0].mean() / metrics[:, 1].mean():.2f}")
    else:
        print("Average Travel Time: 0.00")
        print("Average Travel Speed: N/A")
    print("-----------------------")
    
    dep_times_np = departure_times.cpu().numpy()
    
    df_metrics = pd.DataFrame(trip_metadata)
    
    # Format times as HH:MM:SS
    df_metrics['departure_time'] = [sec_to_hms(t) for t in dep_times_np]
    df_metrics['travel_time'] = [sec_to_hms(t) for t in metrics[:, 1]]
    
    df_metrics['traveled_distance'] = metrics[:, 0]
    
    # Reorder columns as requested
    df_metrics = df_metrics[['agent_id', 'trip_number', 'departure_time', 'travel_time', 'traveled_distance']]
    
    # agents_metrics.csv --------------
    if save_agents:
        metrics_path = os.path.join(output_dir, "agent_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
    
    # paths.csv --------------
    if save_paths:
        df_paths = pd.DataFrame(trip_metadata)
        df_paths.rename(columns={'path_str': 'path'}, inplace=True)
        df_paths = df_paths[['agent_id', 'trip_number', 'path']]
        
        paths_out_path = os.path.join(output_dir, "paths.csv")
        df_paths.to_csv(paths_out_path, index=False)
        print(f"Saved paths to {paths_out_path}")

    # average_metrics.csv --------------
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
    
    avg_metrics.to_csv(os.path.join(output_dir, "average_metrics.csv"), index=False)
    print(f"Saved average metrics to {os.path.join(output_dir, 'average_metrics.csv')}")
    
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
    parser.add_argument("--population", help="Substring to match for population file (e.g. '100000')", default=None)
    parser.add_argument("--save_pickle", help="If set, saves parsed network and population to .pkl files for faster loading next time.", action="store_true")
    # By default, we load pickle if it exists, unless user might want to force reparse?
    # Let's add --no_load_pickle to force reparse
    parser.add_argument("--no_load_pickle", help="Force reparsing from XML even if pickle exists.", action="store_true")
    parser.add_argument("--save_paths", help="If set, exports paths.csv containing agent paths.", action="store_true")
    parser.add_argument("--save_agents", help="If set, exports agent_metrics.csv containing individual agent statistics.", action="store_true")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root_folder):
        print(f"Root folder not found: {args.root_folder}")
    else:
        run_benchmark(args.root_folder, args.population, save_pickle=args.save_pickle, load_pickle=not args.no_load_pickle, save_paths=args.save_paths, save_agents=args.save_agents)
