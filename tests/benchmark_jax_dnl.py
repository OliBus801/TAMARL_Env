"""
Benchmark script for JaxDNL (JAX-based DNL simulator).

Usage:
    PYTHONPATH=. python tests/benchmark_jax_dnl.py tamarl/data/scenarios/grid_world/3x3 --population 100
"""

import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import jax
import time
import sys
import psutil
import datetime
import gc
import pandas as pd

from tamarl.core.jax_dnl import JaxDNL, create_params


def sec_to_hms(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def parse_network(network_file, scale_factor=1.0, timestep=1.0):
    print(f"Parsing Network: {network_file}")
    node_id_to_idx = {}
    edges = []
    link_id_to_idx = {}
    valid_links = 0
    eff_cell_size = 7.5

    context = ET.iterparse(network_file, events=("end",))
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
                    freespeed = float(elem.get('freespeed'))
                    capacity_h = float(elem.get('capacity'))
                    lanes = float(elem.get('permlanes'))

                    unscaledFlowCapacity_s = capacity_h / 3600
                    D_e = unscaledFlowCapacity_s * timestep * scale_factor
                    c_e = ((length * lanes) / eff_cell_size) * scale_factor
                    c_e = max(c_e, D_e)
                    ff_time = length / freespeed
                    temp_spaceCap = ff_time * unscaledFlowCapacity_s * scale_factor
                    c_e = max(c_e, temp_spaceCap)

                    attr = [length, freespeed, c_e, D_e, ff_time]
                    edges.append({'u': u, 'v': v, 'id': link_id, 'attr': attr})
                    link_id_to_idx[link_id] = valid_links
                    valid_links += 1
            elem.clear()

    print(f"Parsed {len(node_id_to_idx)} nodes and {valid_links} car links.")
    return node_id_to_idx, edges, link_id_to_idx


def parse_population(pop_file, link_id_to_idx):
    print(f"Parsing Population: {pop_file}")
    agents = []
    agent_metadata = []
    count = 0
    skipped_no_path = 0

    def time_to_sec(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s

    context = ET.iterparse(pop_file, events=("end",))
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
                elements = list(selected_plan)
                first_dep_time = 0
                person_legs = []
                act_end_times = []
                act_durations = []
                first_act = True

                for el in elements:
                    if el.tag in ['act', 'activity']:
                        end_time_str = el.get('end_time')
                        duration_str = el.get('duration')
                        act_end = -1
                        if end_time_str:
                            act_end = time_to_sec(end_time_str)
                        act_dur = -1
                        if duration_str:
                            act_dur = time_to_sec(duration_str)

                        if first_act:
                            if act_end >= 0:
                                first_dep_time = act_end
                            elif act_dur >= 0:
                                first_dep_time = act_dur
                            first_act = False
                        else:
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
                    num_boundaries = len(person_legs) - 1
                    act_end_times = act_end_times[:num_boundaries]
                    act_durations = act_durations[:num_boundaries]
                    agents.append({
                        'dep_time': first_dep_time,
                        'legs': person_legs,
                        'act_end_times': act_end_times,
                        'act_durations': act_durations,
                    })
                    agent_metadata.append({'agent_id': person_id})

            count += 1
            elem.clear()

    total_legs = sum(len(a['legs']) for a in agents)
    print(f"Parsed {count} persons. Skipped {skipped_no_path} legs. Total agents: {len(agents)}, Total legs: {total_legs}")
    return agents, agent_metadata


def run_benchmark(root_folder, population_filter=None, timestep=1.0, scale_factor=1.0,
                  start_hour=0, end_hour=24, seed=None, track_events=True, output_folder="output"):
    process = psutil.Process(os.getpid())

    # 1. Locate files
    files = [f for f in os.listdir(root_folder) if f.endswith('.xml')]
    net_candidates = [f for f in files if 'network' in f.lower()]
    pop_candidates = [f for f in files if 'population' in f.lower() or 'plans' in f.lower()]

    if not net_candidates:
        print(f"Error: No network file found in {root_folder}")
        return
    network_file = os.path.join(root_folder, net_candidates[0])

    if pop_candidates and population_filter:
        filtered = [p for p in pop_candidates
                    if population_filter in p.replace('-', '_').replace('.', '_').split('_')]
        if filtered:
            pop_candidates = filtered
        else:
            pop_candidates = [p for p in pop_candidates if population_filter in p]

    if not pop_candidates:
        print(f"Error: No population file found in {root_folder}")
        return
    population_file = os.path.join(root_folder, pop_candidates[0])

    print(f"Selected Network: {os.path.basename(network_file)}")
    print(f"Selected Population: {os.path.basename(population_file)}")

    # 1b. Setup Output Directory
    output_dir = os.path.join(root_folder, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Parse
    node_map, edges_data, link_id_to_idx = parse_network(network_file, scale_factor, timestep)
    trips, trip_metadata = parse_population(population_file, link_id_to_idx)

    if len(trips) == 0:
        print("No trips found. Exiting.")
        return

    # 3. Prepare tensors (numpy)
    edge_static = np.array([e['attr'] for e in edges_data], dtype=np.float32)
    edge_endpoints = np.array([[e['u'], e['v']] for e in edges_data], dtype=np.int32)
    departure_times = np.array([a['dep_time'] for a in trips], dtype=np.int32)

    num_agents = len(trips)
    max_path_len = 0
    max_acts = 0
    for a in trips:
        total_len = sum(len(leg) for leg in a['legs']) + len(a['legs']) - 1
        max_path_len = max(max_path_len, total_len)
        max_acts = max(max_acts, len(a['act_end_times']))

    print(f"Packing {num_agents} paths (max len {max_path_len}, max acts {max_acts})...")
    paths_tensor = np.full((num_agents, max_path_len), -1, dtype=np.int32)
    act_end_times = np.full((num_agents, max(max_acts, 1)), -1, dtype=np.int32)
    act_durations = np.full((num_agents, max(max_acts, 1)), -1, dtype=np.int32)
    num_legs = np.zeros(num_agents, dtype=np.int32)

    for i, a in enumerate(trips):
        legs = a['legs']
        num_legs[i] = len(legs)
        ptr = 0
        for leg_idx, leg in enumerate(legs):
            leg_len = len(leg)
            paths_tensor[i, ptr:ptr+leg_len] = leg
            ptr += leg_len
            if leg_idx < len(legs) - 1:
                paths_tensor[i, ptr] = -2
                ptr += 1
        n_acts = len(a['act_end_times'])
        if n_acts > 0:
            act_end_times[i, :n_acts] = a['act_end_times']
            act_durations[i, :n_acts] = a['act_durations']

    del trips
    gc.collect()

    # 4. Create JAX params and state
    params = create_params(
        edge_static=edge_static,
        edge_endpoints=edge_endpoints,
        departure_times=departure_times,
        paths=paths_tensor,
        num_legs=num_legs,
        act_end_times=act_end_times,
        act_durations=act_durations,
        stuck_threshold=10,
        dt=timestep,
    )

    actual_seed = seed if seed is not None else 42
    dnl = JaxDNL(params, seed=actual_seed)

    # 5. Simulation Loop
    max_steps = int(end_hour * 3600)
    start_step = int(start_hour * 3600)
    dnl.current_step = start_step

    is_eval = track_events

    print(f"\nStarting JAX simulation loop... ({start_hour}h -> {end_hour}h)")
    print(f"  is_eval={is_eval} (trajectory tracking {'ON' if is_eval else 'OFF'})")
    print(f"  JAX backend: {jax.default_backend()}")

    # Warmup JIT compilation
    print("  JIT compiling (first step)...")
    t_jit_start = time.time()
    dnl.step(is_eval=is_eval)
    jax.block_until_ready(dnl.state.status)
    t_jit_end = time.time()
    print(f"  JIT compilation done in {t_jit_end - t_jit_start:.2f}s")

    sim_start = time.time()
    step = start_step + 1  # already did 1 step for warmup
    t0 = time.time()

    while step < max_steps:
        dnl.step(is_eval=is_eval)
        step += 1

        if step % 3600 == 0:
            jax.block_until_ready(dnl.state.status)
            t1 = time.time()
            dt = t1 - t0
            ms_per_step = (dt / 3600) * 1000

            status = np.asarray(dnl.state.status)
            en_route = int(((status == 1) | (status == 2)).sum())

            print(f"Hour {step//3600} ({step//3600}h) | En Route: {en_route} | Speed: {ms_per_step:.2f} ms/step | Elapsed: {dt:.2f}s")
            t0 = time.time()

    jax.block_until_ready(dnl.state.status)
    sim_end = time.time()
    print(f"\nSimulation done in {step} steps, {sim_end - sim_start:.2f}s")

    # 6. Metrics
    metrics = dnl.get_metrics()
    print(f"\n--- Average Legs Metrics ---")
    print(f"Arrived: {metrics['arrived_count']}")
    print(f"En Route: {metrics['en_route_count']}")
    print(f"Average Travel Time: {metrics['avg_travel_time']:.2f}")
    print(f"Average Travel Distance: {metrics['avg_travel_dist']:.2f}")
    if metrics['avg_travel_time'] > 0:
        print(f"Average Travel Speed: {metrics['avg_travel_dist'] / metrics['avg_travel_time']:.2f}")
    print("-----------------------")

    # Memory
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"\nPeak RAM: {mem_mb:.2f} MB")

    # Save metrics
    avg_speed = metrics['avg_travel_dist'] / metrics['avg_travel_time'] if metrics['avg_travel_time'] > 0 else 0.0
    avg_metrics = pd.DataFrame({
        'avg_trav_dist': [metrics['avg_travel_dist']],
        'avg_trav_time': [metrics['avg_travel_time']],
        'avg_trav_speed': [avg_speed],
        'compute_time': [sim_end - sim_start],
        'peak_memory': [mem_mb],
        'peak_vram': [0.0],  # JAX CPU doesn't have VRAM metric easily accessible here
    })
    
    avg_metrics.to_csv(os.path.join(output_dir, "average_metrics.csv"), index=False)
    print(f"Saved average metrics to {os.path.join(output_dir, 'average_metrics.csv')}")

    # Events (if tracked)
    if is_eval:
        events = dnl.get_events()
        print(f"Reconstructed {len(events)} events from trajectory matrices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder")
    parser.add_argument("--population", default=None)
    parser.add_argument("--hours", nargs=2, type=float, default=[0, 24], metavar=("START", "END"))
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--timestep", type=float, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--track_events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_folder", help="Output folder for results.", default="output")

    args = parser.parse_args()
    if not os.path.exists(args.root_folder):
        print(f"Root folder not found: {args.root_folder}")
    else:
        run_benchmark(
            args.root_folder, args.population,
            timestep=args.timestep, scale_factor=args.scale_factor,
            start_hour=args.hours[0], end_hour=args.hours[1],
            seed=args.seed, track_events=args.track_events,
            output_folder=args.output_folder,
        )
