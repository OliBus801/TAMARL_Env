"""RL training runner for the DTA Markov Game environment.

Runs episodes with a given policy, computes expanded metrics, and optionally
logs to Weights & Biases for real-time monitoring.
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional

from tqdm import tqdm

from tamarl.envs.dta_markov_game_parallel import DTAMarkovGameEnv
from tamarl.envs.components.metrics import (
    compute_tstt, compute_mean_travel_time, compute_arrival_rate,
    compute_travel_time_stats, compute_relative_gap,
    compute_mean_episode_reward, compute_reward_stats,
)
from tamarl.envs.scenario_loader import load_scenario
from tamarl.rl.wandb_logger import WandbLogger
from tamarl.rl.render_helper import render_episode
from tamarl.rl_models.random_agent import RandomAgent
from tamarl.rl_models.q_learning import QLearningAgent
from tamarl.rl_models.msa_agent import MSAAgent
from tamarl.rl_models.aon_agent import AONAgent
from tamarl.rl_models.sb3_agent import SB3Agent


def run_episode(env: DTAMarkovGameEnv, agent):
    """Run a single episode using the appropriate API for the agent type.

    Uses the batched tensor API for agents that support it (SB3, Random, MSA, Q-learning),
    falls back to the dict-based PettingZoo API for legacy agents.
    """
    use_batched = hasattr(agent, 'get_actions_batched')

    if use_batched:
        return _run_episode_batched(env, agent)
    else:
        return _run_episode_dict(env, agent)


def _run_episode_batched(env: DTAMarkovGameEnv, agent):
    """Run a single episode using the batched tensor API — no dicts, no string parsing."""
    obs_all, masks, deciding, active = env.reset_batched()

    cumulative_rewards = torch.zeros(env.dnl.num_agents, device=env.dnl.device)
    n_decisions = 0
    is_sb3 = isinstance(agent, SB3Agent)
    is_msa = isinstance(agent, MSAAgent)
    is_ql = isinstance(agent, QLearningAgent)

    # Reset Q-learning per-episode state
    if is_ql and hasattr(agent, 'reset_episode'):
        agent.reset_episode()

    while env.has_active_agents():
        K = deciding.numel()
        if K > 0:
            # Get observations for deciding agents only
            obs_deciding = obs_all[deciding] if obs_all.shape[0] == env.dnl.num_agents else \
                           env._obs_builder.build_observations_batched(deciding)

            # Get actions — one forward pass for all K agents
            if is_msa:
                leg_indices = env.dnl.current_leg[deciding]
                actions = agent.get_actions_batched(obs_deciding, masks, deciding, leg_indices)
            else:
                actions = agent.get_actions_batched(obs_deciding, masks, deciding)

            # Path-based: record chosen paths in Q-learning agent
            if is_ql and agent.formulation == 'path-based':
                path_choices = actions.cpu().numpy()
                for idx, aid in enumerate(deciding.cpu().numpy()):
                    agent.set_chosen_path(int(aid), int(path_choices[idx]))

            n_decisions += K
        else:
            actions = torch.empty(0, device=env.dnl.device, dtype=torch.long)

        prev_active = env._active_indices.clone()

        obs_active, rewards, terminated, truncated, masks, deciding = env.step_batched(
            deciding if K > 0 else torch.empty(0, device=env.dnl.device, dtype=torch.long),
            actions,
        )
        obs_all = obs_active  # For next iteration, use obs indexed by active agents

        # Accumulate rewards
        cumulative_rewards[prev_active] += rewards

        # Learning updates
        if is_sb3:
            agent.update_batched(
                obs_active, rewards, terminated, truncated,
                masks, deciding, prev_active,
            )
        elif is_ql:
            agent.update_batched(
                obs_active, rewards, terminated, truncated,
                masks, deciding, prev_active, env._active_indices,
            )

    # Decay exploration after each episode
    if hasattr(agent, 'decay_epsilon'):
        agent.decay_epsilon()

    # ── Compute metrics ──
    cum_np = cumulative_rewards.cpu().numpy()
    cum_dict = {f"agent_{i}": float(cum_np[i]) for i in range(env.dnl.num_agents)}

    return _compute_stats(env, cum_dict, n_decisions)


def _run_episode_dict(env: DTAMarkovGameEnv, agent):
    """Run a single episode using the PettingZoo dict API (Q-learning)."""
    obs, infos = env.reset()

    cumulative_rewards = {}
    n_decisions = 0
    is_learner = hasattr(agent, 'update')

    while env.agents:
        actions = agent.get_actions(obs, infos)
        n_decisions += len(actions)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent_id, r in rewards.items():
            cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + r

        if is_learner:
            agent.update(obs, rewards, terminations, truncations, infos)

    if hasattr(agent, 'decay_epsilon'):
        agent.decay_epsilon()

    return _compute_stats(env, cumulative_rewards, n_decisions)


def _compute_stats(env: DTAMarkovGameEnv, cumulative_rewards: dict, n_decisions: int) -> dict:
    """Compute metrics from a completed episode."""
    tstt = compute_tstt(env.dnl)
    mean_tt = compute_mean_travel_time(env.dnl)
    arr_rate = compute_arrival_rate(env.dnl)
    tt_stats = compute_travel_time_stats(env.dnl)
    reward_stats = compute_reward_stats(cumulative_rewards)
    mean_reward = compute_mean_episode_reward(cumulative_rewards)

    stats = {
        'tstt': tstt,
        'mean_travel_time': mean_tt,
        'arrival_rate': arr_rate,
        'mean_reward': mean_reward,
        'episode_length': env.dnl.current_step,
        'n_decisions': n_decisions,
        'tt_std': tt_stats['std'],
        'tt_min': tt_stats['min'],
        'tt_max': tt_stats['max'],
        'tt_median': tt_stats['median'],
        'tt_p90': tt_stats['p90'],
        'tt_p95': tt_stats['p95'],
        'reward_std': reward_stats['std'],
        'reward_min': reward_stats['min'],
        'reward_max': reward_stats['max'],
    }

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
    timestep: float = 1.0,
    device: str = "cpu",
    seed: Optional[int] = None,
    # Agent
    agent_type: str = "random",
    ql_alpha: float = 0.7,
    ql_gamma: float = 1.0,
    ql_epsilon_start: float = 1.0,
    ql_epsilon_end: float = 0.05,
    ql_epsilon_decay: float = 0.995,
    ql_n_bins: int = 5,
    # Formulation
    formulation: str = "link-based",
    top_k_paths: int = 3,
    # MSA
    msa_alpha_max: float = 1.0,
    msa_alpha_min: float = 0.05,
    msa_decay: float = 0.05,
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
):
    """Run the training loop with expanded metrics and optional W&B logging.

    Args:
        scenario_path: path to scenario folder
        population_filter: substring to match population file
        n_episodes: number of episodes to run
        max_steps: maximum ticks per episode
        timestep: simulation timestep
        device: 'cpu' or 'cuda'
        seed: random seed
        log_interval: compute & log detailed metrics every N episodes (default 100)
        wandb_enabled: enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_tags: W&B run tags
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
        'timestep': timestep,
        'device': device,
        'seed': seed,
        'log_interval': log_interval,
        'agent_type': agent_type,
        'formulation': formulation,
        'top_k_paths': top_k_paths,
        'relative_gap': relative_gap,
    }
    if agent_type == 'qlearning':
        config.update({
            'ql_alpha': ql_alpha,
            'ql_gamma': ql_gamma,
            'ql_epsilon_start': ql_epsilon_start,
            'ql_epsilon_end': ql_epsilon_end,
            'ql_epsilon_decay': ql_epsilon_decay,
            'ql_n_bins': ql_n_bins,
        })
    elif agent_type == 'msa':
        config.update({
            'msa_alpha_max': msa_alpha_max,
            'msa_alpha_min': msa_alpha_min,
            'msa_decay': msa_decay,
        })
    elif agent_type == 'aon':
        pass  # AON doesn't have specific hyperparameters, uses FF times
    elif agent_type in ('ppo', 'dqn', 'a2c'):
        config.update({
            'sb3_lr': sb3_lr,
            'sb3_gamma': sb3_gamma,
            'sb3_net_arch': sb3_net_arch or [64, 64],
            'sb3_batch_size': sb3_batch_size,
            'sb3_buffer_size': sb3_buffer_size,
            'sb3_n_steps': sb3_n_steps,
        })

    print(f"{'='*65}")
    print(f"  DTA Markov Game — Training Runner")
    print(f"{'='*65}")
    print(f"  Scenario:      {scenario_path}")
    print(f"  Population:    {population_filter}")
    print(f"  Agent:         {agent_type}")
    print(f"  Episodes:      {n_episodes} | Max steps: {max_steps}")
    print(f"  Log interval:  every {log_interval} episodes")
    print(f"  Device:        {device} | Seed: {seed}")
    if agent_type == 'qlearning':
        print(f"  Q-Learning:    α={ql_alpha}, γ={ql_gamma}, ε={ql_epsilon_start}→{ql_epsilon_end} (decay={ql_epsilon_decay})")
        if formulation == 'path-based':
            print(f"  Formulation:   path-based (top-k={top_k_paths})")
        else:
            print(f"  Formulation:   link-based")
    elif agent_type == 'msa':
        print(f"  MSA:         α_max={msa_alpha_max}, α_min={msa_alpha_min}, decay={msa_decay}")
    elif agent_type == 'aon':
        print(f"  AON:         Shortest Path at Free Flow")
    elif agent_type in ('ppo', 'dqn', 'a2c'):
        print(f"  SB3 {agent_type.upper()}:    lr={sb3_lr}, γ={sb3_gamma}, net={sb3_net_arch or [64,64]}, batch={sb3_batch_size}")
    print(f"{'='*65}\n")

    # ── W&B init ──
    logger = WandbLogger(
        project=wandb_project,
        run_name=f"{agent_type}-seed{seed}",
        scenario_id=scenario_id,
        agent_type=agent_type,
        config=config,
        enabled=wandb_enabled,
        tags=wandb_tags,
    )

    # ── Environment ──
    need_events = render is not None
    env = DTAMarkovGameEnv(
        scenario_path=scenario_path,
        population_filter=population_filter,
        timestep=timestep,
        max_steps=max_steps,
        device=device,
        seed=seed,
        track_events=need_events,
        formulation=formulation,
        top_k_paths=top_k_paths,
    )

    # ── Agent ──
    if agent_type == 'qlearning':
        # Determine n_actions based on formulation
        if formulation == 'path-based':
            ql_n_actions = top_k_paths
        else:
            ql_n_actions = env.dnl.max_out_degree
        agent = QLearningAgent(
            n_actions=ql_n_actions,
            n_agents=env.dnl.num_agents,
            od_pairs=env.od_pairs,
            alpha=ql_alpha,
            gamma=ql_gamma,
            epsilon_start=ql_epsilon_start,
            epsilon_end=ql_epsilon_end,
            epsilon_decay=ql_epsilon_decay,
            n_congestion_bins=ql_n_bins,
            formulation=formulation,
            paths_per_od=env.paths_per_od,
            top_k=top_k_paths,
            seed=seed,
        )
    elif agent_type == 'msa':
        agent = MSAAgent(
            num_agents=env.dnl.num_agents,
            num_nodes=env.dnl.num_nodes,
            num_edges=env.dnl.num_edges,
            edge_endpoints=env.dnl.edge_endpoints,
            ff_times=env.dnl.ff_travel_time_steps,
            dt=env.dnl.dt,
            alpha_max=msa_alpha_max,
            alpha_min=msa_alpha_min,
            alpha_decay=msa_decay,
            seed=seed,
        )
    elif agent_type == 'aon':
        agent = AONAgent(
            num_agents=env.dnl.num_agents,
            num_nodes=env.dnl.num_nodes,
            num_edges=env.dnl.num_edges,
            edge_endpoints=env.dnl.edge_endpoints,
            ff_times=env.dnl.ff_travel_time_steps,
            dt=env.dnl.dt,
            seed=seed,
        )
    elif agent_type in ('ppo', 'dqn', 'a2c'):
        agent = SB3Agent(
            algorithm=agent_type,
            env=env,
            od_pairs=env.od_pairs,
            learning_rate=sb3_lr,
            gamma=sb3_gamma,
            net_arch=sb3_net_arch,
            device=device,
            seed=seed,
            batch_size=sb3_batch_size,
            buffer_size=sb3_buffer_size,
            n_steps=sb3_n_steps,
        )
    else:
        agent = RandomAgent(seed=seed)

    print(f"Environment: {env.dnl.num_agents} agents, "
          f"{env.dnl.num_edges} edges, {env.dnl.num_nodes} nodes, "
          f"max_out_degree={env.dnl.max_out_degree}")
    print(f"Agent: {agent}\n")

    logger.log_config({
        'num_agents': env.dnl.num_agents,
        'num_edges': env.dnl.num_edges,
        'num_nodes': env.dnl.num_nodes,
        'max_out_degree': env.dnl.max_out_degree,
    })

    # Load scenario data (required for rendering)
    scenario_data = load_scenario(
        scenario_path, population_filter=population_filter, timestep=timestep
    )

    # Build reverse link ID map for rendering (now that scenario_data is loaded)
    idx_to_link_id = None
    if need_events:
        idx_to_link_id = {v: k for k, v in scenario_data.link_id_to_idx.items()}

    if agent_type in ('msa', 'aon'):
        print(f"\n  Initializing {agent_type.upper()} agent routing with FF times...")
        agent.end_episode(env.dnl, is_init=True)

    # ── Training loop ──
    all_stats = []
    window_stats = []  # stats accumulated between log intervals
    
    aon_cached_stats = None

    pbar = tqdm(range(n_episodes), desc="Training", unit="ep", dynamic_ncols=True)

    for ep in pbar:
        # Determine if this episode requires full metric logging
        is_log_step = (ep == 0) or ((ep + 1) % log_interval == 0) or (ep == n_episodes - 1)
        
        if agent_type == 'aon' and aon_cached_stats is not None:
            stats = aon_cached_stats.copy()
            time.sleep(0.005)  # small delay for tqdm to update smoothly
        else:
            # Configure env to collect metric if RGap is enabled or if using MSA
            needs_tt = (relative_gap and is_log_step) or (agent_type == 'msa') or (agent_type == 'aon' and ep == 0 and relative_gap)
            if needs_tt:
                env.dnl.collect_link_tt = True
                if getattr(env.dnl, 'interval_tt_sum', None) is None:
                    env.dnl.max_intervals = 100
                    import torch
                    env.dnl.interval_tt_sum = torch.zeros((env.dnl.max_intervals, env.dnl.num_edges), device=env.dnl.device, dtype=torch.float32)
                    env.dnl.interval_tt_count = torch.zeros((env.dnl.max_intervals, env.dnl.num_edges), device=env.dnl.device, dtype=torch.float32)
            else:
                env.dnl.collect_link_tt = False

            t0 = time.time()
            stats = run_episode(env, agent)
            wall_time = time.time() - t0

            # Add agent-specific stats
            if hasattr(agent, 'epsilon'):
                stats['epsilon'] = agent.epsilon
            if hasattr(agent, 'q_table_size'):
                stats['q_table_size'] = agent.q_table_size

            if relative_gap and (is_log_step or (agent_type == 'aon' and ep == 0)):
                rgap = compute_relative_gap(env.dnl, link_tt_interval=300.0)
                stats['relative_gap'] = rgap

            stats['wall_time'] = wall_time
            
            if agent_type == 'msa':
                agent.end_episode(env.dnl, is_init=False)
                
            if agent_type == 'aon' and ep == 0:
                aon_cached_stats = stats.copy()

        all_stats.append(stats)
        window_stats.append(stats)

        postfix = {
            'Mean TT': f"{stats['mean_travel_time']:.0f}",
            'Arr': f"{stats['arrival_rate']*100:.0f}%",
        }
        if 'epsilon' in stats:
            postfix['ε'] = f"{stats['epsilon']:.3f}"
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

            # Agent-specific metrics
            if 'epsilon_mean' in agg:
                wandb_metrics['metrics/Epsilon'] = agg['epsilon_mean']
            if 'q_table_size_mean' in agg:
                wandb_metrics['charts/Q-Table Size'] = agg['q_table_size_mean']

            for k, v in agg.items():
                if k not in ['tstt_mean', 'mean_travel_time_mean', 'relative_gap_mean',
                             'arrival_rate_mean', 'mean_reward_mean', 'epsilon_mean',
                             'q_table_size_mean']:
                    nice_name = k.replace('_', ' ').title().replace('Tstt', 'TSTT').replace('Tt ', 'TT ')
                    wandb_metrics[f'charts/{nice_name}'] = v

            logger.log_episode(ep, wandb_metrics)

            # ── Render at interval ──
            if render == 'interval' and need_events:
                tqdm.write(f"  🎬 Rendering episode {ep + 1}...")
                render_episode(
                    scenario_path=scenario_path,
                    dnl=env.dnl,
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
        print(f"\n🎬 Rendering final episode...")
        render_episode(
            scenario_path=scenario_path,
            dnl=env.dnl,
            idx_to_link_id=idx_to_link_id,
            episode=n_episodes,
            fmt=render_format,
            render_fps=render_fps,
            render_hours=render_hours,
            render_speed=render_speed,
            filename=f"{agent_type}-{scenario_id}-{n_episodes}",
        )

    env.close()
    return all_stats


def load_config(config_path: str) -> dict:
    """Load a JSON config file and map it to flat train() kwargs.

    The JSON file uses categorized sections (scenario, training, q_learning,
    msa, sb3, logging, rendering, metrics).  This function flattens them
    into the keyword arguments expected by ``train()``.

    Args:
        config_path: path to a ``.json`` configuration file.

    Returns:
        A dict of keyword arguments for ``train()``.
    """
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
        "timestep": "timestep",
        "device": "device",
        "seed": "seed",
        "formulation": "formulation",
        "top_k_paths": "top_k_paths",
    }
    for json_key, kwarg_key in _map.items():
        if json_key in tr:
            kwargs[kwarg_key] = tr[json_key]

    # ── Q-Learning ──
    ql = cfg.get("q_learning", {})
    _ql_map = {
        "alpha": "ql_alpha",
        "gamma": "ql_gamma",
        "epsilon_start": "ql_epsilon_start",
        "epsilon_end": "ql_epsilon_end",
        "epsilon_decay": "ql_epsilon_decay",
        "n_bins": "ql_n_bins",
    }
    for json_key, kwarg_key in _ql_map.items():
        if json_key in ql:
            kwargs[kwarg_key] = ql[json_key]

    # ── MSA ──
    msa = cfg.get("msa", {})
    _msa_map = {
        "alpha_max": "msa_alpha_max",
        "alpha_min": "msa_alpha_min",
        "decay": "msa_decay",
    }
    for json_key, kwarg_key in _msa_map.items():
        if json_key in msa:
            kwargs[kwarg_key] = msa[json_key]

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
    parser = argparse.ArgumentParser(description="DTA Markov Game — Training Runner")

    # Config file
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file (overrides defaults, CLI overrides config)")

    # Scenario
    parser.add_argument("--scenario", default=None, help="Path to scenario folder")
    parser.add_argument("--population", default=None, help="Population file filter (e.g. '100')")

    # Agent
    parser.add_argument("--agent", default=None,
                        choices=["random", "qlearning", "msa", "ppo", "dqn", "a2c", "aon"],
                        help="Agent type: 'random', 'qlearning', 'msa', 'ppo', 'dqn', 'a2c', or 'aon'")

    # Training
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=None, help="Max ticks per episode")
    parser.add_argument("--timestep", type=float, default=None, help="Simulation timestep (s)")
    parser.add_argument("--device", default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging
    parser.add_argument("--log_interval", type=int, default=None,
                        help="Log metrics every N episodes (default: 100)")

    # Q-Learning hyperparameters
    parser.add_argument("--ql_alpha", type=float, default=None, help="Q-learning rate")
    parser.add_argument("--ql_gamma", type=float, default=None, help="Q-learning discount")
    parser.add_argument("--ql_epsilon_start", type=float, default=None, help="Initial epsilon")
    parser.add_argument("--ql_epsilon_end", type=float, default=None, help="Final epsilon")
    parser.add_argument("--ql_epsilon_decay", type=float, default=None, help="Epsilon decay per episode")
    parser.add_argument("--ql_n_bins", type=int, default=None, help="Congestion discretisation bins")

    # Formulation
    parser.add_argument("--formulation", default=None, choices=["link-based", "path-based"],
                        help="Formulation: 'link-based' (default) or 'path-based'")
    parser.add_argument("--top_k_paths", type=int, default=None,
                        help="Number of top-k loopless paths for path-based formulation (default: 3)")

    # MSA parameters
    parser.add_argument("--msa_alpha_max", type=float, default=None, help="Initial MSA update probability")
    parser.add_argument("--msa_alpha_min", type=float, default=None, help="Minimum MSA update probability")
    parser.add_argument("--msa_decay", type=float, default=None, help="MSA exponential decay lambda")

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

    # Metrics
    parser.add_argument("--no-relative_gap", action="store_false", dest="relative_gap",
                        default=None, help="Disable Relative Gap computation")

    return parser


# Mapping from CLI arg names to train() kwarg names
_CLI_TO_KWARGS = {
    "scenario": "scenario_path",
    "population": "population_filter",
    "agent": "agent_type",
    "episodes": "n_episodes",
    "max_steps": "max_steps",
    "timestep": "timestep",
    "device": "device",
    "seed": "seed",
    "log_interval": "log_interval",
    "ql_alpha": "ql_alpha",
    "ql_gamma": "ql_gamma",
    "ql_epsilon_start": "ql_epsilon_start",
    "ql_epsilon_end": "ql_epsilon_end",
    "ql_epsilon_decay": "ql_epsilon_decay",
    "ql_n_bins": "ql_n_bins",
    "formulation": "formulation",
    "top_k_paths": "top_k_paths",
    "msa_alpha_max": "msa_alpha_max",
    "msa_alpha_min": "msa_alpha_min",
    "msa_decay": "msa_decay",
    "sb3_lr": "sb3_lr",
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
