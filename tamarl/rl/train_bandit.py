"""RL training runner for the DTA One-Shot Bandit environment.

Runs episodes using the one-shot bandit paradigm, computes metrics,
and optionally logs to Weights & Biases for real-time monitoring.
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional

from tqdm import tqdm

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.vehicle_level_wrapper import VehicleLevelWrapper
from tamarl.envs.components.metrics import (
    compute_tstt, compute_mean_travel_time, compute_arrival_rate,
    compute_travel_time_stats, compute_relative_gap,
    compute_mean_episode_reward, compute_reward_stats,
)
from tamarl.envs.scenario_loader import load_scenario
from tamarl.rl.wandb_logger import WandbLogger
from tamarl.rl.render_helper import render_episode
from tamarl.rl.agents.random_agent import RandomAgent


def run_episode(env: VehicleLevelWrapper, agent, deterministic: bool = False):
    """Run a single episode using the bandit paradigm."""
    obs, infos = env.reset()

    if hasattr(agent, 'get_actions_batched'):
        # Vectorized agents (like EpsilonGreedyAgent)
        masks = infos.get("action_mask")
        device = getattr(env, "_device", "cpu")
        deciding_indices = torch.arange(env.num_envs, device=device)
        
        # Ensure obs/masks are tensors on the correct device
        obs_t = torch.from_numpy(obs).to(device)
        masks_t = torch.from_numpy(masks).to(device)
        
        actions = agent.get_actions_batched(obs_t, masks_t, deciding_indices)
        actions = actions.cpu().numpy()
    elif hasattr(agent, 'act'):
        # Basic agents (like RandomAgent)
        actions = agent.act()
    elif hasattr(agent, 'predict'):
        # For SB3 compatibility
        actions, _ = agent.predict(obs, deterministic=deterministic)
    else:
        # Fallback to random sampling if no recognizable method
        actions = env.action_space.sample()

    t0 = time.time()
    obs, rewards, terminated, truncated, infos = env.step(actions)
    wall_time = time.time() - t0

    # For SB3, if we were training in the loop we'd do it here, but SB3 natively
    # wraps the environment via agent.learn(). This runner handles manual steps.

    return _compute_stats(env, rewards, infos, wall_time, len(actions))


def _compute_stats(env: VehicleLevelWrapper, rewards: np.ndarray, infos: dict, wall_time: float, n_decisions: int) -> dict:
    """Compute metrics from a completed episode."""
    tt = infos["_episode"]["t"]
    mean_tt = infos["mean_travel_time"]
    tstt = tt.sum()
    
    # In bandit formulation, the simulation runs until all agents finish or max_steps is reached.
    # We can check arrival rate using the DNL's status
    arrival_rate = float((env.bandit.dnl.status >= 3).float().mean().cpu())

    stats = {
        'tstt': float(tstt),
        'mean_travel_time': mean_tt,
        'arrival_rate': arrival_rate,
        'mean_reward': float(rewards.mean()),
        'episode_length': 1,  # It's a bandit episode
        'n_decisions': n_decisions,
        'tt_std': float(np.std(tt)),
        'tt_min': float(np.min(tt)),
        'tt_max': float(np.max(tt)),
        'tt_median': float(np.median(tt)),
        'tt_p90': float(np.percentile(tt, 90)),
        'tt_p95': float(np.percentile(tt, 95)),
        'reward_std': float(np.std(rewards)),
        'reward_min': float(np.min(rewards)),
        'reward_max': float(np.max(rewards)),
        'wall_time': wall_time
    }
    
    # Compute relative gap if required and DNL collected it
    if env.bandit.dnl.collect_link_tt:
        # compute_relative_gap relies on DNL metrics
        rgap = compute_relative_gap(env.bandit.dnl, link_tt_interval=300.0)
        stats['relative_gap'] = rgap

    return stats


def _aggregate_stats(window: List[Dict]) -> Dict[str, float]:
    """Aggregate stats over a window of episodes."""
    keys = ['tstt', 'mean_travel_time', 'arrival_rate', 'mean_reward',
            'episode_length', 'n_decisions', 'tt_p90', 'tt_p95',
            'wall_time', 'relative_gap']
    agg = {}
    for k in keys:
        vals = [s[k] for s in window if k in s]
        if vals:
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
    return agg


def train(
    scenario_path: str,
    population_filter: Optional[str] = None,
    n_episodes: int = 5,
    max_steps: int = 86400,
    stuck_threshold: int = 10,
    timestep: float = 1.0,
    device: str = "cpu",
    seed: Optional[int] = None,
    # Agent
    agent_type: str = "random",
    # Formulation
    formulation: str = "path-based",
    top_k_paths: int = 3,
    # SB3
    sb3_lr: float = 3e-4,
    sb3_gamma: float = 1.0,
    sb3_net_arch: Optional[list] = None,
    sb3_batch_size: int = 64,
    sb3_buffer_size: int = 10_000,
    sb3_n_steps: int = 128,
    # Logging
    log_interval: int = 100,
    # Render
    render: Optional[str] = None,
    render_format: str = 'gif',
    render_fps: int = 5,
    render_hours: Optional[tuple] = None,
    render_speed: int = 1,
    # W&B
    wandb_enabled: bool = False,
    wandb_project: str = "tamarl",
    wandb_tags: Optional[list] = None,
    # Metrics
    relative_gap: bool = True,
    # Epsilon Greedy
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    # UCB
    ucb_c: float = 100.0,
):
    """Run the training loop for the One-Shot Bandit environment.

    Args:
        scenario_path: path to scenario folder
        population_filter: substring to match population file
        n_episodes: number of episodes to run
        max_steps: maximum ticks per episode
        timestep: simulation timestep
        device: 'cpu' or 'cuda'
        seed: random seed
        log_interval: compute & log detailed metrics every N episodes
    """
    scenario_name = os.path.basename(scenario_path.rstrip('/'))

    scenario_id = scenario_name
    if population_filter:
        scenario_id = f"{scenario_name}-{population_filter}"

    # ── Config ──
    config = {
        'scenario_path': scenario_path,
        'population_filter': population_filter,
        'n_episodes': n_episodes,
        'max_steps': max_steps,
        'stuck_threshold': stuck_threshold,
        'timestep': timestep,
        'device': device,
        'seed': seed,
        'log_interval': log_interval,
        'agent_type': agent_type,
        'formulation': formulation,
        'top_k_paths': top_k_paths,
        'relative_gap': relative_gap,
    }
    
    if agent_type in ('ppo', 'dqn', 'a2c'):
        config.update({
            'sb3_lr': sb3_lr,
            'sb3_gamma': sb3_gamma,
            'sb3_net_arch': sb3_net_arch or [64, 64],
            'sb3_batch_size': sb3_batch_size,
            'sb3_buffer_size': sb3_buffer_size,
            'sb3_n_steps': sb3_n_steps,
        })

    if formulation != 'path-based':
        print(f"  ⚠  Bandit runner only supports path-based formulation; overriding.")
        formulation = 'path-based'
        config['formulation'] = 'path-based'

    print(f"{'='*65}")
    print(f"  One-Shot Bandit DTA — Training Runner")
    print(f"{'='*65}")
    print(f"  Scenario:      {scenario_path}")
    print(f"  Population:    {population_filter}")
    print(f"  Agent:         {agent_type}")
    print(f"  Episodes:      {n_episodes} | Max steps: {max_steps} | Stuck threshold: {stuck_threshold}")
    print(f"  Log interval:  every {log_interval} episodes")
    print(f"  Device:        {device} | Seed: {seed}")
    print(f"  Formulation:   path-based (top-k={top_k_paths})")
    if agent_type in ('ppo', 'dqn', 'a2c'):
        print(f"  SB3 {agent_type.upper()}:    lr={sb3_lr}, γ={sb3_gamma}, net={sb3_net_arch or [64,64]}, batch={sb3_batch_size}")
    print(f"{'='*65}\n")

    # ── W&B init ──
    logger = WandbLogger(
        project=wandb_project,
        run_name=f"{agent_type}-bandit-seed{seed}",
        scenario_id=scenario_id,
        agent_type=agent_type,
        config=config,
        enabled=wandb_enabled,
        tags=wandb_tags,
    )

    # ── Environment ──
    need_events = render is not None
    
    # 1. Init DTABanditEnv
    bandit = DTABanditEnv(
        scenario_path=scenario_path,
        population_filter=population_filter,
        timestep=timestep,
        scale_factor=1.0,
        max_steps=max_steps,
        stuck_threshold=stuck_threshold,
        device=device,
        seed=seed,
        track_events=need_events,
    )
    
    # 2. Wrap it in VehicleLevelWrapper
    env = VehicleLevelWrapper(
        bandit=bandit,
        top_k=top_k_paths,
    )

    # ── Agent ──
    if agent_type in ('ppo', 'dqn', 'a2c'):
        agent = SB3Agent(
            algorithm=agent_type,
            env=env,
            learning_rate=sb3_lr,
            gamma=sb3_gamma,
            net_arch=sb3_net_arch,
            device=device,
            seed=seed,
            batch_size=sb3_batch_size,
            buffer_size=sb3_buffer_size,
            n_steps=sb3_n_steps,
        )
    elif agent_type == "epsilon_greedy":
        from tamarl.rl.agents.epsilon_greedy_agent import EpsilonGreedyAgent
        agent = EpsilonGreedyAgent(
            epsilon_start=kwargs.get("epsilon_start", 1.0),
            epsilon_end=kwargs.get("epsilon_end", 0.05),
            epsilon_decay=kwargs.get("epsilon_decay", 0.995),
            seed=seed
        )
    elif agent_type == "ucb":
        from tamarl.rl.agents.ucb_agent import UCBAgent
        agent = UCBAgent(
            num_agents=env.num_envs,
            k_paths=top_k_paths,
            c_exploration=ucb_c,
            device=device
        )
    else:
        from tamarl.rl.agents.random_agent import RandomAgent
        agent = RandomAgent(num_agents=env.num_envs, k=top_k_paths, seed=seed)

    print(f"Environment: {env.bandit.num_agents} agents, "
          f"{env.bandit.scenario.num_edges} edges, {env.bandit.scenario.num_nodes} nodes")
    print(f"Wrapper: VehicleLevelWrapper (top_k={top_k_paths})")
    print(f"Agent: {agent}\n")

    logger.log_config({
        'num_agents': env.bandit.num_agents,
        'num_edges': env.bandit.scenario.num_edges,
        'num_nodes': env.bandit.scenario.num_nodes,
        'bandit_top_k': top_k_paths,
    })

    # Load scenario data (required for rendering reverse mapping)
    idx_to_link_id = None
    if need_events:
        scenario_data = load_scenario(
            scenario_path, population_filter=population_filter, timestep=timestep
        )
        idx_to_link_id = {v: k for k, v in scenario_data.link_id_to_idx.items()}

    # ── Training loop ──
    all_stats = []
    window_stats = []
    
    # If using SB3, and we want to do proper RL training, we could call agent.learn().
    # But for parity with the manual loop of train.py, we run it manually for now.
    # Note: proper SB3 integration should use agent.model.learn(total_timesteps)

    pbar = tqdm(range(n_episodes), desc="Training", unit="ep", dynamic_ncols=True)

    for ep in pbar:
        # Determine if this episode requires full metric logging
        is_log_step = (ep == 0) or ((ep + 1) % log_interval == 0) or (ep == n_episodes - 1)
        
        # Configure env to collect metric if RGap is enabled
        needs_tt = (relative_gap and is_log_step)
        env.bandit.collect_link_tt = needs_tt

        # Decay epsilon if supported
        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
            
        stats = run_episode(env, agent)

        # Update agent state at end of episode if supported (e.g. UCB)
        if hasattr(agent, "end_episode"):
            agent.end_episode()

        if 'relative_gap' not in stats and relative_gap and is_log_step:
            pass # RGap should be inside stats if compute_relative_gap succeeded

        all_stats.append(stats)
        window_stats.append(stats)

        postfix = {
            'Mean TT': f"{stats['mean_travel_time']:.0f}",
            'Arr': f"{stats['arrival_rate']*100:.0f}%",
        }
        pbar.set_postfix(postfix)

        # ── Log at interval ──
        if is_log_step:
            agg = _aggregate_stats(window_stats)

            # Console output
            tqdm.write(
                f"\n  ┌─ Episodes {ep + 1 - len(window_stats) + 1}–{ep + 1} "
                f"({len(window_stats)} eps) ───────────────────────"
            )
            tqdm.write(f"  │ TSTT:       {agg['tstt_mean']:.0f} ± {agg['tstt_std']:.0f}")
            tqdm.write(f"  │ Mean TT:    {agg['mean_travel_time_mean']:.1f} ± {agg['mean_travel_time_std']:.1f}")
            if 'relative_gap_mean' in agg:
                tqdm.write(f"  │ RGap:       {agg['relative_gap_mean']:.4f} ± {agg['relative_gap_std']:.4f}")
            tqdm.write(f"  │ p95:        {agg.get('tt_p95_mean', 0):.1f} ± {agg.get('tt_p95_std', 0):.1f}")
            tqdm.write(f"  │ Arrival:    {agg['arrival_rate_mean']*100:.1f}%")
            tqdm.write(f"  │ Length:     {agg['episode_length_mean']:.1f} ± {agg['episode_length_std']:.1f}")
            tqdm.write(f"  │ Reward:     {agg['mean_reward_mean']:.1f} ± {agg['mean_reward_std']:.1f}")
            tqdm.write(f"  └─────────────────────────────────────────────")

            # W&B logging: log aggregated metrics into folders
            wandb_metrics = {
                'metrics/TSTT': agg['tstt_mean'],
                'metrics/Mean TT': agg['mean_travel_time_mean'],
                'metrics/Arrival Rate': agg['arrival_rate_mean'],
                'metrics/Mean Reward': agg['mean_reward_mean'],
            }
            if 'relative_gap_mean' in agg:
                wandb_metrics['metrics/Relative Gap'] = agg['relative_gap_mean']
            
            if hasattr(agent, "epsilon"):
                wandb_metrics['agent/epsilon'] = agent.epsilon

            for k, v in agg.items():
                if k not in ['tstt_mean', 'mean_travel_time_mean', 'relative_gap_mean',
                             'arrival_rate_mean', 'mean_reward_mean']:
                    nice_name = k.replace('_', ' ').title().replace('Tstt', 'TSTT').replace('Tt ', 'TT ')
                    wandb_metrics[f'charts/{nice_name}'] = v

            logger.log_episode(ep, wandb_metrics)

            # ── Render at interval ──
            if render == 'interval' and need_events:
                tqdm.write(f"  🎬 Rendering episode {ep + 1}...")
                render_episode(
                    scenario_path=scenario_path,
                    dnl=env.bandit.dnl,
                    idx_to_link_id=idx_to_link_id,
                    episode=ep + 1,
                    fmt=render_format,
                    render_fps=render_fps,
                    render_hours=render_hours,
                    render_speed=render_speed,
                    filename=f"{agent_type}-{scenario_id}-{ep + 1}",
                )

            window_stats = []

    # ── Final Summary ──
    print(f"\n{'='*65}")
    print(f"  Summary ({n_episodes} episodes)")
    print(f"{'='*65}")

    def _fmt(key):
        vals = [s[key] for s in all_stats]
        return f"{np.mean(vals):.1f} ± {np.std(vals):.1f}"

    print(f"  TSTT:             {_fmt('tstt')}")
    print(f"  Mean TT:          {_fmt('mean_travel_time')}")
    if relative_gap and any('relative_gap' in s for s in all_stats):
        gap_vals = [s['relative_gap'] for s in all_stats if 'relative_gap' in s]
        print(f"  Relative Gap:     {np.mean(gap_vals):.4f} ± {np.std(gap_vals):.4f}")
    print(f"  TT p95:           {_fmt('tt_p95')}")
    print(f"  Arrival:          {np.mean([s['arrival_rate'] for s in all_stats])*100:.1f}%")
    print(f"  Episode Length:   {_fmt('episode_length')}")
    print(f"  Mean Reward:      {_fmt('mean_reward')}")
    print(f"{'='*65}")

    # Log summary to W&B
    tstt_vals = [s['tstt'] for s in all_stats]
    summary = {
        'metrics/TSTT Mean': float(np.mean(tstt_vals)),
        'charts/TSTT Std': float(np.std(tstt_vals)),
        'metrics/Arrival Rate Mean': float(np.mean([s['arrival_rate'] for s in all_stats])),
        'metrics/Mean Reward Mean': float(np.mean([s['mean_reward'] for s in all_stats])),
    }
    
    if relative_gap and any('relative_gap' in s for s in all_stats):
        gap_vals = [s['relative_gap'] for s in all_stats if 'relative_gap' in s]
        summary['metrics/Relative Gap Mean'] = float(np.mean(gap_vals))
        
    logger.log_summary(summary)
    logger.finish()

    # ── Render at end ──
    if render == 'end' and need_events:
        print(f"\n🎬 Running and rendering final deterministic evaluation episode...")
        
        # Ensure events are tracked for this final evaluation episode
        env.bandit._track_events = True
        env.bandit.collect_link_tt = True
            
        stats = run_episode(env, agent, deterministic=True)
        
        # ── Evaluation Summary ──
        print(f"\n{'='*65}")
        print(f"  Evaluation Summary")
        print(f"{'='*65}")

        print(f"  TSTT:             {stats['tstt']:.0f}")
        print(f"  Mean TT:          {stats['mean_travel_time']:.1f}")
        print(f"  TT p95:           {stats['tt_p95']:.1f}")
        print(f"  Arrival:          {stats['arrival_rate']*100:.1f}%")
        print(f"  Episode Length:   {stats['episode_length']:.1f}")
        print(f"  Mean Reward:      {stats['mean_reward']:.1f}")
        print(f"{'='*65}")
        
        render_episode(
            scenario_path=scenario_path,
            dnl=env.bandit.dnl,
            idx_to_link_id=idx_to_link_id,
            episode='eval',
            fmt=render_format,
            render_fps=render_fps,
            render_hours=render_hours,
            render_speed=render_speed,
            filename=f"{agent_type}-{scenario_id}-bandit-eval",
        )

    env.close()
    return all_stats


def load_config(config_path: str) -> dict:
    """Load a JSON config file and map it to flat train() kwargs."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    kwargs: dict = {}

    # ── Scenario ──
    sc = cfg.get("scenario", {})
    if "path" in sc:
        kwargs["scenario_path"] = sc["path"]
    if "population_filter" in sc:
        kwargs["population_filter"] = sc["population_filter"]

    # ── Training ──
    tr = cfg.get("training", {})
    _map = {
        "agent": "agent_type",
        "episodes": "n_episodes",
        "max_steps": "max_steps",
        "stuck_threshold": "stuck_threshold",
        "timestep": "timestep",
        "device": "device",
        "seed": "seed",
        "formulation": "formulation",
        "top_k_paths": "top_k_paths",
    }
    for json_key, kwarg_key in _map.items():
        if json_key in tr:
            kwargs[kwarg_key] = tr[json_key]

    # ── SB3 ──
    sb3 = cfg.get("sb3", {})
    _sb3_map = {
        "learning_rate": "sb3_lr",
        "gamma": "sb3_gamma",
        "net_arch": "sb3_net_arch",
        "batch_size": "sb3_batch_size",
        "buffer_size": "sb3_buffer_size",
        "n_steps": "sb3_n_steps",
    }
    for json_key, kwarg_key in _sb3_map.items():
        if json_key in sb3:
            kwargs[kwarg_key] = sb3[json_key]

    # ── Logging ──
    log = cfg.get("logging", {})
    if "log_interval" in log:
        kwargs["log_interval"] = log["log_interval"]
    if "wandb_enabled" in log:
        kwargs["wandb_enabled"] = log["wandb_enabled"]
    if "wandb_project" in log:
        kwargs["wandb_project"] = log["wandb_project"]
    if "wandb_tags" in log:
        kwargs["wandb_tags"] = log["wandb_tags"]

    # ── Rendering ──
    rnd = cfg.get("rendering", {})
    if "render" in rnd:
        kwargs["render"] = rnd["render"]
    if "format" in rnd:
        kwargs["render_format"] = rnd["format"]
    if "fps" in rnd:
        kwargs["render_fps"] = rnd["fps"]
    if "hours" in rnd:
        kwargs["render_hours"] = tuple(rnd["hours"]) if rnd["hours"] else None
    if "speed" in rnd:
        kwargs["render_speed"] = rnd["speed"]

    # ── Metrics ──
    met = cfg.get("metrics", {})
    if "relative_gap" in met:
        kwargs["relative_gap"] = met["relative_gap"]

    return kwargs


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for CLI arguments."""
    parser = argparse.ArgumentParser(description="One-Shot Bandit DTA — Training Runner")

    # Config file
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file (overrides defaults, CLI overrides config)")

    # Scenario
    parser.add_argument("--scenario", default=None, help="Path to scenario folder")
    parser.add_argument("--population", default=None, help="Population file filter (e.g. '100')")

    # Agent
    parser.add_argument("--agent", default=None,
                        choices=["random", "ppo", "dqn", "a2c", "epsilon_greedy", "ucb"],
                        help="Agent type: 'random', 'ppo', 'dqn', 'a2c', 'epsilon_greedy', or 'ucb'")

    # Training
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=None, help="Max ticks per episode")
    parser.add_argument("--stuck_threshold", type=int, default=None, help="Steps before a stuck agent is removed")
    parser.add_argument("--timestep", type=float, default=None, help="Simulation timestep (s)")
    parser.add_argument("--device", default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging
    parser.add_argument("--log_interval", type=int, default=None,
                        help="Log metrics every N episodes (default: 100)")

    # Formulation
    parser.add_argument("--formulation", default=None, choices=["path-based"],
                        help="Formulation: 'path-based' only")
    parser.add_argument("--top_k_paths", type=int, default=None,
                        help="Number of top-k loopless paths for path-based formulation (default: 3)")

    # SB3 hyperparameters
    parser.add_argument("--sb3_lr", type=float, default=None, help="SB3 learning rate")
    parser.add_argument("--sb3_gamma", type=float, default=None, help="SB3 discount factor")
    parser.add_argument("--sb3_net_arch", type=int, nargs="+", default=None,
                        help="SB3 hidden layer sizes (e.g. --sb3_net_arch 64 64)")
    parser.add_argument("--sb3_batch_size", type=int, default=None, help="SB3 batch size")
    parser.add_argument("--sb3_buffer_size", type=int, default=None, help="DQN replay buffer size")
    parser.add_argument("--sb3_n_steps", type=int, default=None, help="PPO/A2C rollout length")

    # W&B
    parser.add_argument("--wandb", action="store_true", default=None,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default=None, help="W&B project name")

    # Render
    parser.add_argument("--render", default=None, choices=["interval", "end"],
                        help="Render episodes: 'interval' or 'end'")
    parser.add_argument("--render_format", default=None, choices=["gif", "mp4", "live"],
                        help="Render format: gif, mp4, or live")
    parser.add_argument("--render_fps", type=int, default=None, help="FPS for rendered animation")
    parser.add_argument("--render_hours", type=float, nargs=2, default=None,
                        help="Start and end hours for rendering")
    parser.add_argument("--render_speed", type=int, default=None, help="Speed factor for rendering")

    parser.add_argument("--epsilon_start", type=float, default=None)
    parser.add_argument("--epsilon_end", type=float, default=None)
    parser.add_argument("--epsilon_decay", type=float, default=None)
    parser.add_argument("--ucb_c", type=float, default=None, help="Exploration constant for UCB")

    return parser


# Mapping from CLI arg names to train() kwarg names
_CLI_TO_KWARGS = {
    "scenario": "scenario_path",
    "population": "population_filter",
    "agent": "agent_type",
    "episodes": "n_episodes",
    "max_steps": "max_steps",
    "stuck_threshold": "stuck_threshold",
    "timestep": "timestep",
    "device": "device",
    "seed": "seed",
    "log_interval": "log_interval",
    "formulation": "formulation",
    "top_k_paths": "top_k_paths",
    "sb3_lr": "sb3_lr",
    "epsilon_start": "epsilon_start",
    "epsilon_end": "epsilon_end",
    "epsilon_decay": "epsilon_decay",
    "ucb_c": "ucb_c",
    "sb3_gamma": "sb3_gamma",
    "sb3_net_arch": "sb3_net_arch",
    "sb3_batch_size": "sb3_batch_size",
    "sb3_buffer_size": "sb3_buffer_size",
    "sb3_n_steps": "sb3_n_steps",
    "wandb": "wandb_enabled",
    "wandb_project": "wandb_project",
    "render": "render",
    "render_format": "render_format",
    "render_fps": "render_fps",
    "render_hours": "render_hours",
    "render_speed": "render_speed",
    "relative_gap": "relative_gap",
}


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # 1. Start with train() defaults
    import inspect
    sig = inspect.signature(train)
    kwargs = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    # 2. Override with JSON config if provided
    if args.config:
        config_path = args.config
        if not os.path.isfile(config_path):
            parser.error(f"Config file not found: {config_path}")
        json_kwargs = load_config(config_path)
        kwargs.update(json_kwargs)
        print(f"  📄 Loaded config from: {config_path}")

    # 3. Override with CLI arguments (only those explicitly set)
    args_dict = vars(args)
    for cli_name, kwarg_name in _CLI_TO_KWARGS.items():
        cli_val = args_dict.get(cli_name)
        if cli_val is not None:
            kwargs[kwarg_name] = cli_val

    # 4. Ensure scenario_path is set
    if "scenario_path" not in kwargs or kwargs["scenario_path"] is None:
        parser.error("--scenario is required (via CLI or config file)")

    train(**kwargs)
