"""RL training runner for the DTA Markov Game environment.

Runs episodes with a given policy, computes expanded metrics, and optionally
logs to Weights & Biases for real-time monitoring.
"""

import argparse
import time
import numpy as np
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


def run_episode(env: DTAMarkovGameEnv, agent):
    """Run a single episode and return expanded stats."""
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

        # Q-learning update (or any learnable agent)
        if is_learner:
            agent.update(obs, rewards, terminations, truncations, infos)

    # Decay exploration after each episode
    if hasattr(agent, 'decay_epsilon'):
        agent.decay_epsilon()

    # ── Compute metrics ──
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

    # Agent-specific stats
    if hasattr(agent, 'epsilon'):
        stats['epsilon'] = agent.epsilon
    if hasattr(agent, 'q_table_size'):
        stats['q_table_size'] = agent.q_table_size

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
    ql_alpha: float = 0.1,
    ql_gamma: float = 1.0,
    ql_epsilon_start: float = 1.0,
    ql_epsilon_end: float = 0.05,
    ql_epsilon_decay: float = 0.995,
    ql_n_bins: int = 5,
    # MSA
    msa_alpha_max: float = 1.0,
    msa_alpha_min: float = 0.05,
    msa_decay: float = 0.05,
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
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[list] = None,
    # BPR params for Frank-Wolfe
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
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
        wandb_run_name: W&B run name (auto-generated if None)
        wandb_tags: W&B run tags
        bpr_alpha: BPR alpha parameter for Frank-Wolfe
        bpr_beta: BPR beta parameter for Frank-Wolfe
    """
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
        'bpr_alpha': bpr_alpha,
        'bpr_beta': bpr_beta,
        'agent_type': agent_type,
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
    elif agent_type == 'msa':
        print(f"  S-MSA:         α_max={msa_alpha_max}, α_min={msa_alpha_min}, decay={msa_decay}")
    print(f"  BPR:           α={bpr_alpha}, β={bpr_beta}")
    print(f"  W&B:           {'enabled' if wandb_enabled else 'disabled'}")
    print(f"{'='*65}\n")

    # ── W&B init ──
    logger = WandbLogger(
        project=wandb_project,
        run_name=wandb_run_name,
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
    )

    # ── Agent ──
    if agent_type == 'qlearning':
        agent = QLearningAgent(
            n_actions=env.dnl.max_out_degree,
            alpha=ql_alpha,
            gamma=ql_gamma,
            epsilon_start=ql_epsilon_start,
            epsilon_end=ql_epsilon_end,
            epsilon_decay=ql_epsilon_decay,
            n_congestion_bins=ql_n_bins,
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

    if agent_type == 'msa':
        print("\n  Initializing S-MSA agent routing with FF times...")
        agent.end_episode(env.dnl, is_init=True)

    # ── Training loop ──
    all_stats = []
    window_stats = []  # stats accumulated between log intervals

    pbar = tqdm(range(n_episodes), desc="Training", unit="ep", dynamic_ncols=True)

    for ep in pbar:
        # Determine if this episode requires full metric logging
        is_log_step = ((ep + 1) % log_interval == 0) or (ep == n_episodes - 1)
        
        # Configure env to collect metric if RGap is enabled or if using MSA
        needs_tt = (relative_gap and is_log_step) or (agent_type == 'msa')
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

        if relative_gap and is_log_step:
            rgap = compute_relative_gap(env.dnl, link_tt_interval=300.0)
            stats['relative_gap'] = rgap

        stats['wall_time'] = wall_time
        
        if agent_type == 'msa':
            agent.end_episode(env.dnl, is_init=False)

        all_stats.append(stats)
        window_stats.append(stats)

        postfix = {
            'TSTT': f"{stats['tstt']:.0f}",
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
                    filename=wandb_run_name if wandb_run_name else None,
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
            filename=wandb_run_name if wandb_run_name else None,
        )

    env.close()
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTA Markov Game — Training Runner")

    # Scenario
    parser.add_argument("--scenario", required=True, help="Path to scenario folder")
    parser.add_argument("--population", default=None, help="Population file filter (e.g. '100')")

    # Agent
    parser.add_argument("--agent", default="random", choices=["random", "qlearning", "msa"],
                        help="Agent type: 'random', 'qlearning' or 'msa'")

    # Training
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=3600, help="Max ticks per episode")
    parser.add_argument("--timestep", type=float, default=1.0, help="Simulation timestep (s)")
    parser.add_argument("--device", default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log metrics every N episodes (default: 100)")

    # Q-Learning hyperparameters
    parser.add_argument("--ql_alpha", type=float, default=0.1, help="Q-learning rate (default: 0.1)")
    parser.add_argument("--ql_gamma", type=float, default=1.0, help="Q-learning discount (default: 1.0)")
    parser.add_argument("--ql_epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--ql_epsilon_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--ql_epsilon_decay", type=float, default=0.995, help="Epsilon decay per episode")
    parser.add_argument("--ql_n_bins", type=int, default=5, help="Congestion discretisation bins")

    # MSA parameters
    parser.add_argument("--msa_alpha_max", type=float, default=1.0, help="Initial MSA update probability")
    parser.add_argument("--msa_alpha_min", type=float, default=0.05, help="Minimum MSA update probability")
    parser.add_argument("--msa_decay", type=float, default=0.05, help="MSA exponential decay lambda")

    # BPR parameters
    parser.add_argument("--bpr_alpha", type=float, default=0.15, help="BPR alpha (default: 0.15)")
    parser.add_argument("--bpr_beta", type=float, default=4.0, help="BPR beta (default: 4.0)")

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="tamarl", help="W&B project name")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name")

    # Render
    parser.add_argument("--render", default=None, choices=["interval", "end"],
                        help="Render episodes: 'interval' (at each log_interval) or 'end' (final episode)")
    parser.add_argument("--render_format", default="gif", choices=["gif", "mp4", "live"],
                        help="Render format: gif, mp4, or live")
    parser.add_argument("--render_fps", type=int, default=5, help="FPS for rendered animation (default: 5)")
    parser.add_argument("--render_hours", type=float, nargs=2, default=None,
                        help="Start and end hours for rendering (e.g. --render_hours 0 0.15)")
    parser.add_argument("--render_speed", type=int, default=1, help="Speed factor for rendering (default: 1)")

    # Metrics
    parser.add_argument("--no-relative_gap", action="store_false", dest="relative_gap",
                        help="Disable Relative Gap computation to save compute")

    args = parser.parse_args()

    train(
        scenario_path=args.scenario,
        population_filter=args.population,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        timestep=args.timestep,
        device=args.device,
        seed=args.seed,
        agent_type=args.agent,
        ql_alpha=args.ql_alpha,
        ql_gamma=args.ql_gamma,
        ql_epsilon_start=args.ql_epsilon_start,
        ql_epsilon_end=args.ql_epsilon_end,
        ql_epsilon_decay=args.ql_epsilon_decay,
        ql_n_bins=args.ql_n_bins,
        msa_alpha_max=args.msa_alpha_max,
        msa_alpha_min=args.msa_alpha_min,
        msa_decay=args.msa_decay,
        log_interval=args.log_interval,
        render=args.render,
        render_format=args.render_format,
        render_fps=args.render_fps,
        render_hours=args.render_hours,
        render_speed=args.render_speed,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        bpr_alpha=args.bpr_alpha,
        bpr_beta=args.bpr_beta,
        relative_gap=args.relative_gap,
    )
