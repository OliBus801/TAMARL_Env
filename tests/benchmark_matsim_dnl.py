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
from tamarl.core.dnl_matsim import TorchDNLMATSim, EVENT_TYPE_NAMES
from tamarl.visualisation.plot_histogram import plot_leg_histogram, export_leg_histogram_csv

def sec_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=int(seconds)))

def parse_network(network_file, scale_factor=1.0, timestep=1.0):
    print(f"Parsing Network: {network_file}")
    node_id_to_idx = {}
    edges = []
    link_id_to_idx = {}
    valid_links = 0
    
    context = ET.iterparse(network_file, events=("end",))
     
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
                    
                    length = float(elem.get('length')) # meters
                    freespeed = float(elem.get('freespeed')) # m/s
                    capacity_h = float(elem.get('capacity')) # veh/h
                    lanes = float(elem.get('permlanes'))
                    
                    # Calculate Flow Capacity (maximum Inflow per timestep)
                    unscaledFlowCapacity_s = (capacity_h / 3600) 
                    D_e = unscaledFlowCapacity_s * timestep * scale_factor

                    # Calculate Storage Capacity
                    # First guess
                    c_e = ((length * lanes) / eff_cell_size) * scale_factor # storageCapacity

                    # Storage Capacity needs to be at least enough to handle the flow per timestep
                    c_e = max(c_e, D_e)

                    # If speed on link is too slow, then we need more cells than above to handle the flowCap per timestep
                    # i.e. if freeFlowTravelTime is 2 seconds, then we need spaceCap = 2 * flowCap to handle it.
                    ff_time = length / freespeed # freeFlowTravelTime
                    temp_spaceCap = ff_time * unscaledFlowCapacity_s * scale_factor
                    # Adjust storageCapacity if too small, needs to accomodate at least free flow flux
                    c_e = max(c_e, temp_spaceCap)

                    attr = [length, freespeed, c_e, D_e, ff_time]
                    
                    edges.append({'u': u, 'v': v, 'id': link_id, 'attr': attr})
                    link_id_to_idx[link_id] = valid_links
                    valid_links += 1
            
            elem.clear()
            
    print(f"Parsed {len(node_id_to_idx)} nodes and {valid_links} car links.")
    return node_id_to_idx, edges, link_id_to_idx

def parse_population(pop_file, link_id_to_idx):
    """Parse population XML into one agent per person with multi-leg support.
    
    Returns:
        agents: list of dicts per person with keys:
            - agent_id, dep_time, legs (list of path index lists),
            - act_end_times, act_durations (per intermediate activity)
        agent_metadata: list of dicts with agent_id for export
    """
    print(f"Parsing Population: {pop_file}")
    
    agents = []
    agent_metadata = []
    
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
                # Collect all elements in order
                elements = list(selected_plan)
                
                # First pass: collect legs and intermediate activity attributes
                first_dep_time = 0
                person_legs = []       # list of path index lists
                act_end_times = []     # per intermediate activity
                act_durations = []     # per intermediate activity
                
                # Parse first activity end_time
                first_act = True
                pending_act_end = -1   # absolute end_time for next intermediate act
                pending_act_dur = -1   # duration for next intermediate act
                
                for el in elements:
                    if el.tag in ['act', 'activity']:
                        end_time_str = el.get('end_time')
                        dur_str = el.get('dur')
                        max_dur_str = el.get('max_dur')
                        
                        # Compute absolute end_time
                        act_end = -1
                        if end_time_str:
                            act_end = time_to_sec(end_time_str)
                        
                        # Compute duration: min(dur, max_dur) if both set
                        act_dur = -1
                        if dur_str and max_dur_str:
                            act_dur = min(time_to_sec(dur_str), time_to_sec(max_dur_str))
                        elif dur_str:
                            act_dur = time_to_sec(dur_str)
                        elif max_dur_str:
                            act_dur = time_to_sec(max_dur_str)
                        
                        if first_act:
                            # First activity: its end_time is the first departure time
                            if act_end >= 0:
                                first_dep_time = act_end
                            elif act_dur >= 0:
                                first_dep_time = act_dur
                            first_act = False
                        else:
                            # Intermediate or final activity
                            act_end_times.append(act_end)
                            act_durations.append(act_dur)
                    
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
                                    person_legs.append(path_indices)
                                else:
                                    skipped_no_path += 1
                            else:
                                skipped_no_path += 1
                
                if len(person_legs) > 0:
                    # Trim act_end_times/act_durations to match number of leg boundaries
                    # We have len(person_legs) - 1 intermediate activities
                    num_boundaries = len(person_legs) - 1
                    act_end_times = act_end_times[:num_boundaries]
                    act_durations = act_durations[:num_boundaries]
                    
                    agents.append({
                        'dep_time': first_dep_time,
                        'legs': person_legs,
                        'act_end_times': act_end_times,
                        'act_durations': act_durations,
                    })
                    agent_metadata.append({
                        'agent_id': person_id,
                    })
            
            count += 1
            elem.clear()
            
    total_legs = sum(len(a['legs']) for a in agents)
    print(f"Parsed {count} persons. Skipped {skipped_no_path} legs. Total agents: {len(agents)}, Total legs: {total_legs}")
    return agents, agent_metadata

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 # MB

def run_benchmark(root_folder, population_filter=None, timestep=1.0, scale_factor=1.0, start_hour=0, end_hour=24, save_pickle=False, load_pickle=True, save_paths=False, save_agents=False, save_events=False, track_events=True, output_folder="output", seed=None):
    
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
            # Try to find exact token matches first
            tokens_candidates = []
            for p in pop_candidates:
                # Split by common delimiters
                tokens = p.replace('-', '_').replace('.', '_').split('_')
                if population_filter in tokens:
                    tokens_candidates.append(p)
            
            if tokens_candidates:
                pop_candidates = tokens_candidates
            else:
                # Fallback to substring matching
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
    output_dir = os.path.join(root_folder, output_folder)
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
        node_map, edges_data, link_id_to_idx = parse_network(network_file, scale_factor, timestep)
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
    # Keep link_id_to_idx if we need it for event export
    if not save_events:
        del link_id_to_idx
        link_id_to_idx = None
    gc.collect()

    # 4. Prepare Data for DNL
    edge_static_list = [e['attr'] for e in edges_data]
    edge_static = torch.tensor(edge_static_list, dtype=torch.float32)
    
    # Edge connectivity for path validation
    edge_endpoints = torch.tensor([[e['u'], e['v']] for e in edges_data], dtype=torch.int32)
    
    # Free edges_data
    del edges_data
    gc.collect()
    
    # Note: `trips` in parsing is now `agents`
    agents_data = trips
    departure_times = torch.tensor([a['dep_time'] for a in agents_data], dtype=torch.int32)
    
    num_agents = len(agents_data)
    
    # Calculate max lengths
    max_path_len = 0
    max_acts = 0
    for a in agents_data:
        # lengths of individual legs + sentinels (-2) between them
        total_len = sum(len(leg) for leg in a['legs']) + len(a['legs']) - 1
        max_path_len = max(max_path_len, total_len)
        max_acts = max(max_acts, len(a['act_end_times']))

    print(f"Packing {num_agents} paths (max len {max_path_len}, max int_acts {max_acts})...")
    paths_tensor = torch.full((num_agents, max_path_len), -1, dtype=torch.int32)
    act_end_times_tensor = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    act_durations_tensor = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    num_legs_tensor = torch.zeros(num_agents, dtype=torch.int32)
    
    for i, a in enumerate(agents_data):
        legs = a['legs']
        num_legs_tensor[i] = len(legs)
        
        # Build concatenated path with -2 sentinels
        ptr = 0
        for leg_idx, leg in enumerate(legs):
            leg_len = len(leg)
            paths_tensor[i, ptr:ptr+leg_len] = torch.tensor(leg, dtype=torch.int32)
            ptr += leg_len
            if leg_idx < len(legs) - 1:
                paths_tensor[i, ptr] = -2  # sentinel
                ptr += 1
                
        # Fill act times
        n_acts = len(a['act_end_times'])
        if n_acts > 0:
            act_end_times_tensor[i, :n_acts] = torch.tensor(a['act_end_times'], dtype=torch.int32)
            act_durations_tensor[i, :n_acts] = torch.tensor(a['act_durations'], dtype=torch.int32)
    
    del trips
    agents_data = None
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
    
    # If save_events is requested, we must track events
    effective_track_events = track_events or save_events
    
    dnl = TorchDNLMATSim(
        edge_static, 
        paths_tensor, 
        device=device, 
        departure_times=departure_times, 
        edge_endpoints=edge_endpoints,
        act_end_times=act_end_times_tensor,
        act_durations=act_durations_tensor,
        num_legs=num_legs_tensor,
        dt=timestep,
        seed=seed, 
        stuck_threshold=10,
        enable_profiling=True,
        track_events=effective_track_events
    )

    
    # 5. Simulation Loop
    max_steps = int(end_hour * 3600)
    start_step = int(start_hour * 3600)
    
    dnl.current_step = start_step
    
    sim_start = time.time()
    step = start_step
    active = True
    
    print(f"Starting simulation loop... ({start_hour}h -> {end_hour}h)")
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

            # Calculate interactions and reset
            interactions = dnl.interactions
            dnl.interactions = 0

            print(f"Hour {step//3600} ({step//3600}h) | En Route: {en_route} | Speed: {ms_per_step:.2f} ms/step | Interactions: {interactions} | Elapsed time : {dt:.2f}s")
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
    
    # Note: trip_metadata was renamed to agent_metadata in parse_population
    # but the variable returned is still captured as trip_metadata in run_benchmark
    df_metrics = pd.DataFrame(trip_metadata)
    
    # Format times as HH:MM:SS
    df_metrics['departure_time'] = [sec_to_hms(t) for t in dep_times_np]
    df_metrics['travel_time'] = [sec_to_hms(t) for t in metrics[:, 1]]
    
    df_metrics['traveled_distance'] = metrics[:, 0]
    
    # Reorder columns as requested
    df_metrics = df_metrics[['agent_id', 'departure_time', 'travel_time', 'traveled_distance']]
    
    # agents_metrics.csv --------------
    if save_agents:
        metrics_path = os.path.join(output_dir, "agent_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
    
    # paths.csv --------------
    if save_paths:
        df_paths = pd.DataFrame(trip_metadata)
        # Note: 'path_str' is no longer stored since we have multiple legs per agent
        # We can just ignore the path export or export a simplified version
        df_paths = df_paths[['agent_id']]
        
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
    
    # Plot (only if events were tracked) --------------
    if effective_track_events:
        events = dnl.get_events()
        plot_file = os.path.join(output_dir, "leg_histogram.png")
        print("Generating Plot...")
        plot_leg_histogram(events, max_steps=step, dt=timestep, bucket_size_sec=300, output_file=plot_file)
        csv_file = os.path.join(output_dir, "leg_histogram.csv")
        export_leg_histogram_csv(events, max_steps=step, dt=timestep, bucket_size_sec=300, output_file=csv_file)

    # Events CSV --------------
    if save_events:
        # events is already fetched above (save_events implies effective_track_events)
        if not events:
            events = dnl.get_events()
        print(f"Collected {len(events)} events. Exporting to CSV...")

        # Build inverse mapping: edge_idx -> link_id
        idx_to_link_id = {v: k for k, v in link_id_to_idx.items()}

        events_rows = []
        for (t, evt_type, agent_id, edge_id) in events:
            evt_name = EVENT_TYPE_NAMES.get(evt_type, str(evt_type))
            link_name = idx_to_link_id.get(edge_id, str(edge_id))
            person_name = trip_metadata[agent_id]['agent_id'] if agent_id < len(trip_metadata) else f"agent_{agent_id}"
            
            # Extra info
            extra = ''
            if evt_type in (0, 7):  # actend / actstart
                extra = 'h' if evt_type == 0 else 'w'
            elif evt_type in (1, 6):  # departure / arrival
                extra = 'car'
            
            events_rows.append({
                'time': t * timestep,
                'type': evt_name,
                'person': person_name,
                'link': link_name,
                'extra': extra
            })

        df_events = pd.DataFrame(events_rows)
        events_path = os.path.join(output_dir, "events.csv")
        df_events.to_csv(events_path, index=False)
        print(f"Saved {len(events_rows)} events to {events_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", help="Root folder containing network.xml and population.xml")
    parser.add_argument("--population", help="Substring to match for population file (e.g. '100000')", default=None)
    parser.add_argument("--save_pickle", help="If set, saves parsed network and population to .pkl files for faster loading next time.", action="store_true")
    parser.add_argument("--no_load_pickle", help="Force reparsing from XML even if pickle exists.", action="store_true")
    parser.add_argument("--save_paths", help="If set, exports paths.csv containing agent paths.", action="store_true")
    parser.add_argument("--save_agents", help="If set, exports agent_metrics.csv containing individual agent statistics.", action="store_true")
    parser.add_argument("--save_events", help="If set, exports MATSim-style simulation events to events.csv.", action="store_true")
    parser.add_argument("--track_events", help="Track simulation events in memory (needed for leg_histogram plot). Default: True.", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hours", nargs=2, type=float, metavar=("START", "END"), help="Start and end hours for the simulation (e.g. --hours 7 7.5 for 7h00 to 7h30). Default: 0 24.", default=[0, 24])
    parser.add_argument("--scale_factor", help="If set, scales the storageCapacity and outflowCapacity of the network links. By default 1.0.", default=1.0)
    parser.add_argument("--timestep", help="Time step size of each simulation step, by default 1 second.", default=1)
    parser.add_argument("--output_folder", help="Output folder for results.", default="output")
    parser.add_argument("--seed", help="Seed for random number generator. Used in priority calculation", default=None)


    args = parser.parse_args()
    
    if not os.path.exists(args.root_folder):
        print(f"Root folder not found: {args.root_folder}")
    else:
        run_benchmark(args.root_folder, args.population, timestep=float(args.timestep), scale_factor=float(args.scale_factor), start_hour=args.hours[0], end_hour=args.hours[1], save_pickle=args.save_pickle, load_pickle=not args.no_load_pickle, save_paths=args.save_paths, save_agents=args.save_agents, save_events=args.save_events, track_events=args.track_events, output_folder=args.output_folder, seed=int(args.seed) if args.seed is not None else None)
