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
from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.envs.od_level_wrapper import ODLevelWrapper
from tamarl.envs.centralized_level_wrapper import CentralizedLevelWrapper
from tamarl.envs.components.metrics import (
    compute_tstt, compute_mean_travel_time, compute_arrival_rate,
    compute_travel_time_stats, compute_mean_episode_reward, compute_reward_stats,
)
from tamarl.envs.scenario_loader import load_scenario
from tamarl.rl.wandb_logger import WandbLogger
from tamarl.rl.render_helper import render_episode
from tamarl.rl.agents.random_agent import RandomAgent
from tamarl.core.memory_profiler import analyze_tensor_memory


def run_episode(env, agent, epsilon_ratio: float = 0.10, profile_memory: bool = False):
    """Run a single episode using the bandit paradigm.

    Génère un tenseur ``aggregation_indices`` [N] qui est passé uniformément
    à l'agent pour la sélection d'action et la mise à jour des paramètres.

    - Mode agent  : aggregation_indices = torch.arange(N)  → identité
    - Mode od_pair: aggregation_indices = od_indices        → parameter sharing
    - Mode centralisé (futur): aggregation_indices = torch.zeros(N, dtype=long)

    L'agent ne voit jamais la distinction : il projette toujours ses poids
    [B, K] via self.weights[aggregation_indices] et agrège via scatter_add_.
    """
    if profile_memory:
        analyze_tensor_memory("BEFORE EPISODE")
    
    with torch.no_grad():
        obs, infos = env.reset()
    device = getattr(env, "_device", "cpu")

    # ── Construction de aggregation_indices ──────────────────────────────
    # Si le wrapper expose des od_indices (mode OD), on les utilise.
    # Sinon, on utilise l'identité (chaque véhicule est son propre bloc).
    N = obs.shape[0]
    if "od_indices" in infos:
        aggregation_indices = torch.from_numpy(infos["od_indices"]).to(device)
    else:
        aggregation_indices = torch.arange(N, device=device)

    if hasattr(agent, 'get_actions_batched'):
        # Agents vectorisés : passage uniforme de aggregation_indices
        masks = infos.get("action_mask")
        obs_t = torch.from_numpy(obs).to(device)
        masks_t = torch.from_numpy(masks).to(device)

        actions = agent.get_actions_batched(
            obs_t, masks_t, aggregation_indices=aggregation_indices
        )
        actions = actions.detach().contiguous().cpu().numpy()
    elif hasattr(agent, 'act'):
        # Agents basiques (RandomAgent legacy)
        actions = agent.act()
    elif hasattr(agent, 'predict'):
        # Compatibilité SB3
        actions, _ = agent.predict(obs)
    else:
        actions = env.action_space.sample()

    # ── Blindage des actions ───────────────────────────────────────────
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().contiguous().cpu().numpy()
    elif isinstance(actions, np.ndarray):
        pass # Already numpy
    else:
        # Probablement un scalaire ou une liste, on laisse tel quel ou on convertit si besoin
        pass

    t0 = time.time()
    with torch.no_grad():
        obs, rewards, terminated, truncated, infos = env.step(actions)
    wall_time = time.time() - t0

    # ── Mise à jour de l'agent ───────────────────────────────────────────
    if hasattr(agent, 'update'):
        actions_t = torch.from_numpy(actions).to(device)
        rewards_t = torch.from_numpy(rewards).to(device)
        agent.update(actions_t, rewards_t, aggregation_indices=aggregation_indices)

    if profile_memory:
        analyze_tensor_memory("AFTER EPISODE")
    
    return _compute_stats(env, rewards, infos, wall_time, len(actions), epsilon_ratio)


def _compute_stats(env: AgentLevelWrapper, rewards: np.ndarray, infos: dict, wall_time: float, n_decisions: int, epsilon_ratio: float = 0.10) -> dict:
    """Compute metrics from a completed episode."""
    tt = infos["_episode"]["t"]
    mean_tt = infos["mean_travel_time"]
    tstt = tt.sum()
    
    # In the bandit setup, the simulation runs until all agents finish or max_steps is reached.
    # We can check arrival rate using the DNL's status
    arrival_rate = float((env.bandit.dnl.status >= 3).float().mean().cpu())

    stats = {
        'tstt': float(tstt),
        'mean_travel_time': mean_tt,
        'arrival_rate': arrival_rate,
        'mean_reward': float(rewards.mean()),
        'episode_length': int(env.bandit.dnl.current_step),  # Number of simulation steps taken
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
    
    # Pull empirical Nash metrics from infos (computed in AgentLevelWrapper.step)
    for k in ['mean_regret', 'max_regret', 'epsilon_compliance_rate']:
        if k in infos:
            stats[k] = infos[k]

    return stats


def _aggregate_stats(window: List[Dict]) -> Dict[str, float]:
    """Aggregate stats over a window of episodes."""
    keys = ['tstt', 'mean_travel_time', 'arrival_rate', 'mean_reward',
            'episode_length', 'n_decisions', 'tt_p90', 'tt_p95',
            'wall_time', 'mean_regret', 'max_regret',
            'epsilon_compliance_rate']
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
    scale_factor: float = 1.0,
    device: str = "cpu",
    seed: Optional[int] = None,
    # Agent
    agent_type: str = "random",
    top_k_paths: int = 3,
    formulation: str = "agent",
    bandit_feedback: str = "full",
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
    epsilon_ratio: float = 0.10,
    link_tt_interval: float = 60.0,
    # Epsilon Greedy
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    epsilon_alpha: float = 0.1,
    # UCB
    ucb_c: float = 100.0,
    # Thompson Sampling
    ts_prior_std: float = 10000.0,
    ts_env_std: Optional[float] = None,
    # Exp3
    exp3_eta: float = 0.01,
    exp3_gamma: float = 0.05,
    # Reload paths
    reload_paths: bool = False,
    # Profiling
    profile_memory: bool = False,
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
        'scale_factor': scale_factor,
        'device': device,
        'seed': seed,
        'log_interval': log_interval,
        'agent_type': agent_type,
        'top_k_paths': top_k_paths,
        'formulation': formulation,
        'bandit_feedback': bandit_feedback,
        'epsilon_ratio': epsilon_ratio,
        'epsilon_alpha': epsilon_alpha,
        'ucb_c': ucb_c,
        'ts_prior_std': ts_prior_std,
        'ts_env_std': ts_env_std,
        'exp3_eta': exp3_eta,
        'exp3_gamma': exp3_gamma,
        'reload_paths': reload_paths,
    }


    print(f"{'='*65}")
    print(f"  One-Shot Bandit DTA — Training Runner")
    print(f"{'='*65}")
    print(f"  Scenario:      {scenario_path}")
    print(f"  Population:    {population_filter}")
    print(f"  Agent:         {agent_type}")
    print(f"  Episodes:      {n_episodes} | Max steps: {max_steps} | Stuck threshold: {stuck_threshold}")
    print(f"  Log interval:  every {log_interval} episodes")
    print(f"  Device:        {device} | Seed: {seed}")
    print(f"  Top-k Paths:   {top_k_paths} | Scale Factor: {scale_factor}")
    print(f"  Formulation:   {formulation} | Feedback: {bandit_feedback}")
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
    need_events = render == 'interval'
    need_render = render is not None
    
    # 1. Init DTABanditEnv
    bandit = DTABanditEnv(
        scenario_path=scenario_path,
        population_filter=population_filter,
        timestep=timestep,
        scale_factor=scale_factor,
        max_steps=max_steps,
        stuck_threshold=stuck_threshold,
        device=device,
        seed=seed,
        track_events=need_events,
        link_tt_interval=link_tt_interval,
        profile_memory=profile_memory,
    )
    
    # 2. Wrap it in the appropriate wrapper (based on formulation)
    if formulation == "agent":
        env = AgentLevelWrapper(
            bandit=bandit,
            top_k=top_k_paths,
            feedback_type=bandit_feedback,
            reload_paths=reload_paths,
        )
    elif formulation == "od_pair":
        env = ODLevelWrapper(
            bandit=bandit,
            top_k=top_k_paths,
            feedback_type=bandit_feedback,
            reload_paths=reload_paths,
        )
    elif formulation == "centralized":
        env = CentralizedLevelWrapper(
            bandit=bandit,
            top_k=top_k_paths,
            feedback_type=bandit_feedback,
            reload_paths=reload_paths,
        )
    else:
        raise ValueError(f"Unknown formulation: {formulation}")

    # Nombre de blocs de paramètres B selon la formulation :
    #   agent       → B = N (chaque véhicule a ses propres poids)
    #   od_pair     → B = M (partage par paire OD)
    #   centralized → B = 1 (un seul modèle pour tous)
    if formulation == "od_pair":
        num_agent_models = env.num_od_pairs
    elif formulation == "centralized":
        num_agent_models = env.num_models  # toujours 1
    else:
        num_agent_models = env.num_envs

    # ── Agent ──
    if agent_type == "epsilon_greedy":
        from tamarl.rl.agents.epsilon_greedy_agent import EpsilonGreedyAgent
        agent = EpsilonGreedyAgent(
            num_agents=num_agent_models,
            k_paths=top_k_paths,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            alpha=epsilon_alpha,
            device=device,
            seed=seed
        )
    elif agent_type == "ucb":
        from tamarl.rl.agents.ucb_agent import UCBAgent
        agent = UCBAgent(
            num_agents=num_agent_models,
            k_paths=top_k_paths,
            c_exploration=ucb_c,
            device=device
        )
    elif agent_type == "ts":
        from tamarl.rl.agents.thompson_sampling_agent import ThompsonSamplingAgent
        
        # Dynamic Priors Logic
        prior_means = None
        env_stds = None
        
        if hasattr(env, "fftt_matrix") and hasattr(env, "od_indices_all_legs"):
            if formulation == "od_pair":
                # OD-pair formulation: priors are per OD pair [num_od_pairs, K]
                fftt_per_model = env.fftt_matrix  # [num_od_pairs, K]
            elif formulation == "centralized":
                # Centralized: prior = moyenne globale sur toutes les OD [1, K]
                masked_fftt_all = env.fftt_matrix.copy()
                masked_fftt_all[masked_fftt_all == np.inf] = np.nan
                # nanmean sur l'axe 0 (sur toutes les OD) → [K]
                grand_mean = np.nanmean(masked_fftt_all, axis=0)
                grand_mean = np.nan_to_num(grand_mean, nan=0.0)
                fftt_per_model = grand_mean[np.newaxis, :]  # [1, K]
            else:
                # Agent formulation: map to per-vehicle [TotalLegs, K]
                fftt_per_model = env.fftt_matrix[env.od_indices_all_legs.cpu().numpy()]

            # Rewards are negative travel times
            prior_means = torch.from_numpy(-fftt_per_model).to(device)

            # Env Std is the std of FFTTs among paths, with a floor of 10.0s
            if ts_env_std is None:
                masked_fftt = fftt_per_model.copy()
                masked_fftt[masked_fftt == np.inf] = np.nan
                std_vals = np.nanstd(masked_fftt, axis=1)
                std_vals = np.nan_to_num(std_vals, nan=0.0)
                env_stds = torch.from_numpy(np.maximum(std_vals, 10.0)).to(device)
            else:
                env_stds = torch.full((num_agent_models,), ts_env_std, device=device)

        agent = ThompsonSamplingAgent(
            num_agents=num_agent_models,
            k_paths=top_k_paths,
            prior_mean=prior_means,
            prior_std=ts_prior_std,
            env_std=env_stds,
            device=device,
            seed=seed
        )

    elif agent_type == "exp3":
        from tamarl.rl.agents.exp3_agent import Exp3Agent
        
        # Calculate rho: 2 * max freeflow travel time among all valid paths
        rho = 1.0
        if hasattr(env, "fftt_matrix"):
            # fftt_matrix has np.inf for invalid paths
            valid_fftts = env.fftt_matrix[env.fftt_matrix != np.inf]
            if len(valid_fftts) > 0:
                rho = float(np.max(valid_fftts)) * 2.0
                
        agent = Exp3Agent(
            num_agents=num_agent_models,
            k_paths=top_k_paths,
            eta=exp3_eta,
            gamma=exp3_gamma,
            rho=rho,
            device=device,
            seed=seed
        )

    elif agent_type == "frank_wolfe":
        from tamarl.rl.agents.frank_wolfe_agent import FrankWolfeBanditAgent
        agent = FrankWolfeBanditAgent(
            env=env,
            num_edges=env.bandit.scenario.num_edges,
            device=device
        )
    elif agent_type == "aon":
        from tamarl.rl.agents.aon_agent import AONAgent
        agent = AONAgent(seed=seed)
    elif agent_type == "msa":
        from tamarl.rl.agents.msa_agent import MSAAgent
        agent = MSAAgent(
            env=env,
            k_paths=top_k_paths,
            device=device,
            seed=seed,
        )
    elif agent_type == "evo_swap":
        from tamarl.rl.agents.evo_swap_agent import EvoSwapAgent
        agent = EvoSwapAgent(
            env=env,
            k_paths=top_k_paths,
            device=device,
            seed=seed,
            epsilon_max=epsilon_start,
            epsilon_min=epsilon_end,
            epsilon_decay=epsilon_decay,
        )
    else:
        from tamarl.rl.agents.random_agent import RandomAgent
        agent = RandomAgent(num_agents=env.num_envs, k=top_k_paths, seed=seed)

    print(f"Environment: {env.bandit.num_agents} agents, "
          f"{env.bandit.scenario.num_edges} edges, {env.bandit.scenario.num_nodes} nodes")
    _wrapper_names = {
        "agent": "AgentLevelWrapper",
        "od_pair": "ODLevelWrapper",
        "centralized": "CentralizedLevelWrapper",
    }
    wrapper_name = _wrapper_names.get(formulation, formulation)
    extra_info = ""
    if formulation == "od_pair":
        extra_info = f", num_od_pairs={env.num_od_pairs}"
    elif formulation == "centralized":
        extra_info = f", num_od_pairs={env.num_od_pairs} → 1 shared model"
    print(f"Wrapper: {wrapper_name} (top_k={top_k_paths}{extra_info})")
    print(f"Agent models: {num_agent_models} (formulation={formulation})")
    print(f"Agent: {agent}\n")

    logger.log_config({
        'num_agents': env.bandit.num_agents,
        'num_edges': env.bandit.scenario.num_edges,
        'num_nodes': env.bandit.scenario.num_nodes,
        'bandit_top_k': top_k_paths,
    })

    # Load scenario data (required for rendering reverse mapping)
    idx_to_link_id = None
    if need_render:
        scenario_data = load_scenario(
            scenario_path, population_filter=population_filter, timestep=timestep
        )
        idx_to_link_id = {v: k for k, v in scenario_data.link_id_to_idx.items()}

    # ── Training loop ──
    all_stats = []
    window_stats = []
    cumulative_max_regret = 0.0
    
    # If using SB3, and we want to do proper RL training, we could call agent.learn().
    # But for parity with the manual loop of train.py, we run it manually for now.
    # Note: proper SB3 integration should use agent.model.learn(total_timesteps)

    pbar = tqdm(range(n_episodes), desc="Training", unit="ep", dynamic_ncols=True)

    for ep in pbar:
        # Determine if this episode requires full metric logging
        is_log_step = (ep == 0) or ((ep + 1) % log_interval == 0) or (ep == n_episodes - 1)
        
        # Configure env to collect metric if RGap is enabled
        needs_tt = is_log_step
        # MSA and EvoSwap agents always need collect_link_tt (TD evaluator requirement)
        if agent_type in ["msa", "evo_swap"]:
            needs_tt = True
        env.bandit.collect_link_tt = needs_tt

        # Enable event tracking for rendering if it's the last episode and render == 'end'
        if ep == n_episodes - 1 and render == 'end':
            env.bandit._track_events = True
            env.bandit.collect_link_tt = True

        # Decay epsilon if supported
        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
            


        stats = run_episode(env, agent, epsilon_ratio, profile_memory=profile_memory)
        
        # Accumulate cumulative regret
        if 'max_regret' in stats:
            cumulative_max_regret += stats['max_regret']
        stats['cumulative_max_regret'] = cumulative_max_regret

        # Update agent state at end of episode if supported (e.g. UCB)
        if hasattr(agent, "end_episode"):
            agent.end_episode()


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
            if 'mean_regret_mean' in agg:
                tqdm.write(f"  │ Max Regret: {agg['max_regret_mean']:.1f}s")
                tqdm.write(f"  │ Cum Regret: {stats['cumulative_max_regret']:.1f}s")
                tqdm.write(f"  │ ε-Compl:    {agg['epsilon_compliance_rate_mean']*100:.1f}%")
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
            if 'mean_regret_mean' in agg:
                wandb_metrics['metrics/Mean Nash Regret'] = agg['mean_regret_mean']
                wandb_metrics['metrics/Max Nash Regret'] = agg['max_regret_mean']
                wandb_metrics['metrics/Cumulative Nash Regret'] = stats['cumulative_max_regret']
                wandb_metrics['metrics/Epsilon Compliance'] = agg['epsilon_compliance_rate_mean']
            
            if hasattr(agent, "epsilon"):
                wandb_metrics['agent/epsilon'] = agent.epsilon

            for k, v in agg.items():
                if k not in ['tstt_mean', 'mean_travel_time_mean',
                             'arrival_rate_mean', 'mean_reward_mean', 'mean_regret_mean',
                             'max_regret_mean', 'epsilon_compliance_rate_mean']:
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
    if any('max_regret' in s for s in all_stats):
        print(f"  Max Regret:       {_fmt('max_regret')}s")
        print(f"  Cum Regret:       {cumulative_max_regret:.1f}s")
        compl_vals = [s['epsilon_compliance_rate'] for s in all_stats if 'epsilon_compliance_rate' in s]
        print(f"  Epsilon Compl:    {np.mean(compl_vals)*100:.1f}%")
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
    
    if any('max_regret' in s for s in all_stats):
        summary['metrics/Cumulative Nash Regret Final'] = float(cumulative_max_regret)
        
    logger.log_summary(summary)
    logger.finish()

    # ── Render at end ──
    if render == 'end' and env.bandit._track_events:
        tqdm.write(f"\n🎬 Rendering final training episode...")
        render_episode(
            scenario_path=scenario_path,
            dnl=env.bandit.dnl,
            idx_to_link_id=idx_to_link_id,
            episode=n_episodes,
            fmt=render_format,
            render_fps=render_fps,
            render_hours=render_hours,
            render_speed=render_speed,
            filename=f"{agent_type}-{scenario_id}-final",
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
    if "scale_factor" in sc:
        kwargs["scale_factor"] = float(sc["scale_factor"])

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
        "top_k_paths": "top_k_paths",
        "formulation": "formulation",
        "bandit_feedback": "bandit_feedback",
        "ucb_c": "ucb_c",
        "ts_prior_std": "ts_prior_std",
        "ts_env_std": "ts_env_std",
        "exp3_eta": "exp3_eta",
        "exp3_gamma": "exp3_gamma",
        "reload_paths": "reload_paths",
    }
    for json_key, kwarg_key in _map.items():
        if json_key in tr:
            kwargs[kwarg_key] = tr[json_key]

    # ── Epsilon Greedy (backward compatible with q_learning) ──
    eg = cfg.get("epsilon_greedy", cfg.get("q_learning", {}))
    _eg_map = {
        "epsilon_start": "epsilon_start",
        "epsilon_end": "epsilon_end",
        "epsilon_decay": "epsilon_decay",
        "epsilon_alpha": "epsilon_alpha",
        "alpha": "epsilon_alpha",
    }
    for json_key, kwarg_key in _eg_map.items():
        if json_key in eg:
            kwargs[kwarg_key] = eg[json_key]

    # ── UCB ──
    ucb = cfg.get("ucb", {})
    if "c" in ucb:
        kwargs["ucb_c"] = ucb["c"]

    # ── Thompson Sampling ──
    ts = cfg.get("thompson_sampling", {})
    if "prior_std" in ts:
        kwargs["ts_prior_std"] = ts["prior_std"]
    if "env_std" in ts:
        kwargs["ts_env_std"] = ts["env_std"]

    # ── Exp3 ──
    exp3 = cfg.get("exp3", {})
    if "eta" in exp3:
        kwargs["exp3_eta"] = exp3["eta"]
    if "gamma" in exp3:
        kwargs["exp3_gamma"] = exp3["gamma"]

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
    if "epsilon_ratio" in met:
        kwargs["epsilon_ratio"] = met["epsilon_ratio"]
    if "link_tt_interval" in met:
        kwargs["link_tt_interval"] = met["link_tt_interval"]

    return kwargs


def _parse_wandb_tags(tags_str: str) -> List[str]:
    """Parse wandb_tags string from CLI.
    
    Handles JSON-like list strings (e.g. '["tag1", "tag2"]') or comma-separated lists.
    """
    if not tags_str:
        return []
    tags_str = tags_str.strip()
    if (tags_str.startswith('[') and tags_str.endswith(']')) or (tags_str.startswith('(') and tags_str.endswith(')')):
        try:
            normalized = tags_str.replace("'", '"')
            parsed = json.loads(normalized)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed]
        except Exception:
            tags_str = tags_str[1:-1]
            
    return [t.strip().strip('"').strip("'").strip() for t in tags_str.split(',') if t.strip()]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for CLI arguments."""
    parser = argparse.ArgumentParser(description="One-Shot Bandit DTA — Training Runner")

    # Config file
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file (overrides defaults, CLI overrides config)")

    # Scenario
    parser.add_argument("--scenario", default=None, help="Path to scenario folder")
    parser.add_argument("--population", default=None, help="Population file filter (e.g. '100')")
    parser.add_argument("--formulation", default=None, choices=["agent", "od_pair", "centralized"],
                        help="Environment formulation: 'agent', 'od_pair', or 'centralized'")
    parser.add_argument("--bandit_feedback", default=None, choices=["full", "semi"],
                        help="Feedback type for bandit formulation: 'full' or 'semi'")

    # Agent
    parser.add_argument("--agent", default=None,
                        choices=["random", "epsilon_greedy", "ucb", "aon", "frank_wolfe", "ts", "exp3", "msa", "evo_swap"],
                        help="Agent type: 'random', 'epsilon_greedy', 'ucb', 'aon', 'frank_wolfe', 'ts', 'exp3', 'msa', or 'evo_swap'")

    # Training
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=None, help="Max ticks per episode")
    parser.add_argument("--stuck_threshold", type=int, default=None, help="Steps before a stuck agent is removed")
    parser.add_argument("--timestep", type=float, default=None, help="Simulation timestep (s)")
    parser.add_argument("--scale_factor", type=float, default=None, help="Network capacity scale factor (default: 1.0)")
    parser.add_argument("--device", default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging
    parser.add_argument("--log_interval", type=int, default=None,
                        help="Log metrics every N episodes (default: 100)")

    parser.add_argument("--top_k_paths", type=int, default=None,
                        help="Number of top-k loopless paths (default: 3)")

    # W&B
    parser.add_argument("--wandb", action="store_true", default=None,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default=None, help="W&B project name")
    parser.add_argument("--wandb_tags", default=None,
                        help="W&B run tags (comma-separated or JSON list string)")

    # Render
    parser.add_argument("--render", default=None, choices=["interval", "end"],
                        help="Render episodes: 'interval' or 'end'")
    parser.add_argument("--render_format", default=None, choices=["gif", "mp4", "live"],
                        help="Render format: gif, mp4, or live")
    parser.add_argument("--render_fps", type=int, default=None, help="FPS for rendered animation")
    parser.add_argument("--render_hours", type=float, nargs=2, default=None,
                        help="Start and end hours for rendering")
    parser.add_argument("--render_speed", type=int, default=None, help="Speed factor for rendering")

    parser.add_argument("--profile_memory", action="store_true", help="Enable memory profiling")
    parser.add_argument("--reload_paths", action="store_true", default=None,
                        help="Force recalculation and overwrite of the cached shortest paths .pkl file")

    parser.add_argument("--epsilon_start", type=float, default=None)
    parser.add_argument("--epsilon_end", type=float, default=None)
    parser.add_argument("--epsilon_decay", type=float, default=None)
    parser.add_argument("--epsilon_alpha", type=float, default=None, help="Learning rate for Epsilon-Greedy Q-values")
    parser.add_argument("--ucb_c", type=float, default=None, help="Exploration constant for UCB")
    parser.add_argument("--ts_prior_std", type=float, default=None, help="Initial belief uncertainty for TS")
    parser.add_argument("--ts_env_std", type=float, default=None, help="Assumed environment noise for TS (overrides dynamic std if set)")
    parser.add_argument("--exp3_eta", type=float, default=None, help="Learning rate for Exp3 agent")
    parser.add_argument("--exp3_gamma", type=float, default=None, help="Exploration parameter for Exp3 agent")

    # Metrics Config
    parser.add_argument("--epsilon_ratio", type=float, default=None, help="Threshold ratio for Epsilon-compliance (e.g., 0.10 for 10%%)")
    parser.add_argument("--link_tt_interval", type=float, default=None, help="Aggregation interval for Nash metrics (seconds)")

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
    "scale_factor": "scale_factor",
    "device": "device",
    "seed": "seed",
    "log_interval": "log_interval",
    "top_k_paths": "top_k_paths",
    "formulation": "formulation",
    "bandit_feedback": "bandit_feedback",
    "epsilon_start": "epsilon_start",
    "epsilon_end": "epsilon_end",
    "epsilon_decay": "epsilon_decay",
    "epsilon_alpha": "epsilon_alpha",
    "ucb_c": "ucb_c",
    "ts_prior_std": "ts_prior_std",
    "ts_env_std": "ts_env_std",
    "exp3_eta": "exp3_eta",
    "exp3_gamma": "exp3_gamma",
    "wandb": "wandb_enabled",
    "wandb_project": "wandb_project",
    "wandb_tags": "wandb_tags",
    "render": "render",
    "render_format": "render_format",
    "render_fps": "render_fps",
    "render_hours": "render_hours",
    "render_speed": "render_speed",
    "epsilon_ratio": "epsilon_ratio",
    "link_tt_interval": "link_tt_interval",
    "profile_memory": "profile_memory",
    "reload_paths": "reload_paths",
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
            if cli_name == "wandb_tags" and isinstance(cli_val, str):
                cli_val = _parse_wandb_tags(cli_val)
            kwargs[kwarg_name] = cli_val

    # 4. Ensure scenario_path is set
    if "scenario_path" not in kwargs or kwargs["scenario_path"] is None:
        parser.error("--scenario is required (via CLI or config file)")

    train(**kwargs)
