import argparse
import inspect
import os
import time

import numpy as np
import torch

from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.envs.centralized_level_wrapper import CentralizedLevelWrapper
from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.od_level_wrapper import ODLevelWrapper
from tamarl.rl.agents.random_agent import RandomAgent
from tamarl.rl.train_bandit import _CLI_TO_KWARGS, _build_parser, load_config, run_episode, train


def run_bandit_profiling(kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM at start: {mem_start:.2f} MB")

    print("\n--- 1. Instantiating DTABanditEnv ---")
    bandit = DTABanditEnv(
        scenario_path=kwargs["scenario_path"],
        population_filter=kwargs.get("population_filter"),
        timestep=kwargs.get("timestep", 1.0),
        scale_factor=kwargs.get("scale_factor", 1.0),
        max_steps=kwargs.get("max_steps", 36000),
        device=device,
        track_events=False,
    )

    if device == "cuda":
        mem_bandit = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM after DTABanditEnv: {mem_bandit:.2f} MB")

    top_k_paths = kwargs.get("top_k", 3)
    formulation = kwargs.get("formulation", "agent")
    feedback_type = kwargs.get("feedback_type", "full")

    print(f"\n--- 2. Instantiating Wrapper (formulation={formulation}, top_k={top_k_paths}) ---")
    if formulation == "agent":
        env = AgentLevelWrapper(bandit=bandit, top_k=top_k_paths, feedback_type=feedback_type)
    elif formulation == "od":
        env = ODLevelWrapper(bandit=bandit, top_k=top_k_paths, feedback_type=feedback_type)
    elif formulation == "centralized":
        env = CentralizedLevelWrapper(bandit=bandit, top_k=top_k_paths, feedback_type=feedback_type)
    else:
        raise ValueError(f"Unknown formulation: {formulation}")

    if device == "cuda":
        mem_wrapper = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM after {env.__class__.__name__}: {mem_wrapper:.2f} MB")

        # Breakdown wrapper tensors
        if hasattr(env, "candidate_routes"):
            cr = env.candidate_routes
            size_mb = cr.element_size() * cr.nelement() / 1024**2
            print(
                f"  - candidate_routes: {size_mb:.2f} MB (dtype: {cr.dtype}, shape: {list(cr.shape)})"
            )

    print("\n--- 3. Running 1 Episode (like train_bandit.py) ---")
    agent = RandomAgent(num_agents=env.num_envs, k=top_k_paths)

    print(
        "Running run_episode(env, agent)... (This triggers reset, action generation, step & update)"
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    # Execute the full episode cycle manually tracking memory (Zero overhead)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    run_episode(env, agent)
    elapsed = time.time() - t0

    print(f"\nEpisode finished in {elapsed:.2f} seconds.")

    if device == "cuda":
        mem_after_step = torch.cuda.memory_allocated() / 1024**2
        peak_step = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nVRAM after 1 episode: {mem_after_step:.2f} MB")
        print(f"Peak VRAM during episode: {peak_step:.2f} MB")

        with open("memory_summary_bandit.txt", "w") as f:
            f.write(torch.cuda.memory_summary(device=device))
        print("\nSaved detailed memory summary to memory_summary_bandit.txt")

    # Final breakdown of DNL inside bandit
    if bandit.dnl is not None:
        print("\n--- Final Breakdown of inner TorchDNL ---")
        tensors = [
            ("paths_flat", getattr(bandit.dnl, "paths_flat", None)),
            ("interval_tt_sum", getattr(bandit.dnl, "interval_tt_sum", None)),
        ]
        for name, t in tensors:
            if t is not None:
                size_mb = t.element_size() * t.nelement() / 1024**2
                print(f"{name:20s}: {size_mb:8.2f} MB (dtype: {t.dtype}, shape: {list(t.shape)})")


def main():
    parser = _build_parser()
    parser.description = "Profile the One-Shot Bandit DTA memory footprint"

    args, unknown = parser.parse_known_args()

    # 1. Start with train() defaults
    sig = inspect.signature(train)
    kwargs = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }

    # 2. Override with JSON config if provided
    if args.config:
        config_path = args.config
        if not os.path.isfile(config_path):
            parser.error(f"Config file not found: {config_path}")
        json_kwargs = load_config(config_path)
        kwargs.update(json_kwargs)
        print(f"  [Config] Loaded from: {config_path}")

    # 3. Override with CLI arguments (only those explicitly set)
    args_dict = vars(args)
    for cli_name, kwarg_name in _CLI_TO_KWARGS.items():
        cli_val = args_dict.get(cli_name)
        if cli_val is not None:
            kwargs[kwarg_name] = cli_val

    # 4. Ensure scenario_path is set
    if "scenario_path" not in kwargs or kwargs["scenario_path"] is None:
        parser.error("--scenario is required (via CLI or config file)")

    run_bandit_profiling(kwargs)


if __name__ == "__main__":
    main()
