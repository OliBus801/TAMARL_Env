import os
import argparse
import torch
import gc
import sys
import psutil
from tamarl.core.dnl_matsim import TorchDNLMATSim

# Reuse parsing from benchmark script to keep it simple and accurate
from tests.benchmark_matsim_dnl import parse_network, parse_population

def run_profiling(root_folder, population_filter=None, timestep=1.0, scale_factor=1.0, 
                  use_float16=False, expandable_segments=False, max_steps=1000):
    
    if expandable_segments:
        # Note: This env variable only works if set before python starts, 
        # so this is just informational if run from python.
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        print("Note: PYTORCH_ALLOC_CONF=expandable_segments:True should be set before running the script.")

    # 1. Locate files
    files = [f for f in os.listdir(root_folder) if f.endswith('.xml')]
    network_file = next((os.path.join(root_folder, f) for f in files if 'network' in f.lower()), None)
    
    pop_candidates = [f for f in files if 'population' in f.lower() or 'plans' in f.lower()]
    if population_filter:
        pop_candidates = [p for p in pop_candidates if population_filter in p]
    population_file = os.path.join(root_folder, pop_candidates[0]) if pop_candidates else None
    
    print(f"Network: {network_file}")
    print(f"Population: {population_file}")

    # 2. Parse Data
    print("\n--- Parsing Data ---")
    node_map, edges_data, link_id_to_idx = parse_network(network_file, scale_factor, timestep)
    trips, _ = parse_population(population_file, link_id_to_idx)
    
    edge_static = torch.tensor([e['attr'] for e in edges_data], dtype=torch.float32)
    edge_endpoints = torch.tensor([[e['u'], e['v']] for e in edges_data], dtype=torch.int32)
    departure_times = torch.tensor([a['dep_time'] for a in trips], dtype=torch.int32)
    
    num_agents = len(trips)
    max_path_len = max(sum(len(leg) for leg in a['legs']) + len(a['legs']) - 1 for a in trips)
    max_acts = max(len(a['act_end_times']) for a in trips)

    paths_tensor = torch.full((num_agents, max_path_len), -1, dtype=torch.int32)
    act_end_times_tensor = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    act_durations_tensor = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    num_legs_tensor = torch.zeros(num_agents, dtype=torch.int32)
    
    for i, a in enumerate(trips):
        legs = a['legs']
        num_legs_tensor[i] = len(legs)
        ptr = 0
        for leg_idx, leg in enumerate(legs):
            paths_tensor[i, ptr:ptr+len(leg)] = torch.tensor(leg, dtype=torch.int32)
            ptr += len(leg)
            if leg_idx < len(legs) - 1:
                paths_tensor[i, ptr] = -2
                ptr += 1
        
        n_acts = len(a['act_end_times'])
        if n_acts > 0:
            act_end_times_tensor[i, :n_acts] = torch.tensor(a['act_end_times'], dtype=torch.int32)
            act_durations_tensor[i, :n_acts] = torch.tensor(a['act_durations'], dtype=torch.int32)
            
    del trips, edges_data
    gc.collect()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("\n[WARNING] GPU not available! Script will run on CPU, but memory profiling relies on CUDA.")
        # Proceeding anyway just for syntax check
        
    torch.cuda.empty_cache()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"\nVRAM before instantiation: {mem_before:.2f} MB")

    print(f"\n--- Instantiating TorchDNLMATSim on {device} ---")
    
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
        collect_link_tt=True, # Critical to reproduce the interval_tt_count memory allocation
        link_tt_interval=300.0
    )

    if use_float16:
        print("Converting interval_tt tensors to float16/bfloat16...")
        # Simulating the change
        dnl.interval_tt_sum = dnl.interval_tt_sum.to(torch.bfloat16)
        dnl.interval_tt_count = dnl.interval_tt_count.to(torch.bfloat16)

    if device == 'cuda':
        mem_after = torch.cuda.memory_allocated() / 1024**2
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"VRAM after instantiation: {mem_after:.2f} MB")
        print(f"Peak VRAM during instantiation: {peak_mem:.2f} MB")
        
        with open("memory_summary_init.txt", "w") as f:
            f.write(torch.cuda.memory_summary(device=device))
        print("Saved detailed memory summary to memory_summary_init.txt")

    print(f"\n--- Running Profiler on step() for {max_steps} steps ---")
    dnl.current_step = 0
    
    # We will use PyTorch Profiler to analyze memory allocation during steps
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=50, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(max_steps):
            dnl.step()
            prof.step()

    print("PyTorch profiler logs saved to ./profiler_logs (view with tensorboard)")
    
    # Analyze data structures
    print("\n--- Data Structures Footprint ---")
    tensors = [
        ('edge_static', dnl.edge_static),
        ('paths', dnl.paths),
        ('interval_tt_sum', getattr(dnl, 'interval_tt_sum', None)),
        ('interval_tt_count', getattr(dnl, 'interval_tt_count', None)),
        ('status', dnl.status),
        ('_event_buffer', getattr(dnl, '_event_buffer', None)),
    ]
    
    for name, t in tensors:
        if t is not None:
            size_mb = t.element_size() * t.nelement() / 1024**2
            print(f"{name:20s}: {size_mb:8.2f} MB (dtype: {t.dtype}, shape: {list(t.shape)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", help="Root folder containing network and population xml files")
    parser.add_argument("--population", help="Population filter", default=None)
    parser.add_argument("--use_float16", action="store_true", help="Cast heavy arrays to float16 to test impact")
    parser.add_argument("--expandable_segments", action="store_true", help="Set PYTORCH_ALLOC_CONF=expandable_segments:True")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run in profiler")
    
    args = parser.parse_args()
    run_profiling(args.root_folder, args.population, use_float16=args.use_float16, 
                  expandable_segments=args.expandable_segments, max_steps=args.steps)
