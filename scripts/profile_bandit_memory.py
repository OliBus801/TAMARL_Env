import argparse
import os
import torch
import numpy as np
import time

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.rl.agents.random_agent import RandomAgent

def run_bandit_profiling(scenario_path, population_filter, top_k_paths=3, max_steps=1000):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM at start: {mem_start:.2f} MB")
    
    print("\n--- 1. Instantiating DTABanditEnv ---")
    bandit = DTABanditEnv(
        scenario_path=scenario_path,
        population_filter=population_filter,
        timestep=1.0,
        max_steps=max_steps,
        device=device,
        track_events=False
    )
    
    if device == 'cuda':
        mem_bandit = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM after DTABanditEnv: {mem_bandit:.2f} MB")

    print(f"\n--- 2. Instantiating AgentLevelWrapper (top_k={top_k_paths}) ---")
    env = AgentLevelWrapper(
        bandit=bandit,
        top_k=top_k_paths,
        feedback_type="full"
    )
    
    if device == 'cuda':
        mem_wrapper = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM after AgentLevelWrapper: {mem_wrapper:.2f} MB")
        
        # Breakdown wrapper tensors
        if hasattr(env, 'candidate_routes'):
            cr = env.candidate_routes
            size_mb = cr.element_size() * cr.nelement() / 1024**2
            print(f"  - candidate_routes: {size_mb:.2f} MB (dtype: {cr.dtype}, shape: {list(cr.shape)})")
            
    print("\n--- 3. Running 1 Episode (like train_bandit.py) ---")
    agent = RandomAgent(num_agents=env.num_envs, k=top_k_paths)
    obs, infos = env.reset()
    
    # Generate mock actions
    actions = agent.act()
    
    print("Running env.step(actions)... (This triggers path building & TorchDNLMATSim instantiation)")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        mem_before_step = torch.cuda.memory_allocated() / 1024**2
    
    # We profile the step function
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs_bandit'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        obs, rewards, terminated, truncated, infos = env.step(actions)
        prof.step()

    if device == 'cuda':
        mem_after_step = torch.cuda.memory_allocated() / 1024**2
        peak_step = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nVRAM after 1 episode: {mem_after_step:.2f} MB")
        print(f"Peak VRAM during episode: {peak_step:.2f} MB")
        
        with open("memory_summary_bandit.txt", "w") as f:
            f.write(torch.cuda.memory_summary(device=device))
        print("\nSaved detailed memory summary to memory_summary_bandit.txt")
        print("PyTorch profiler logs saved to ./profiler_logs_bandit")

    # Final breakdown of DNL inside bandit
    if bandit.dnl is not None:
        print("\n--- Final Breakdown of inner TorchDNLMATSim ---")
        tensors = [
            ('paths_flat', getattr(bandit.dnl, 'paths_flat', None)),
            ('interval_tt_sum', getattr(bandit.dnl, 'interval_tt_sum', None)),
        ]
        for name, t in tensors:
            if t is not None:
                size_mb = t.element_size() * t.nelement() / 1024**2
                print(f"{name:20s}: {size_mb:8.2f} MB (dtype: {t.dtype}, shape: {list(t.shape)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_path", help="Path to scenario folder")
    parser.add_argument("--population", help="Population filter", default=None)
    parser.add_argument("--top_k", type=int, default=3, help="Top K paths")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps for DNL")
    
    args = parser.parse_args()
    run_bandit_profiling(args.scenario_path, args.population, args.top_k, args.steps)
