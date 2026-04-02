"""Metrics for evaluating DTA environment performance."""

from typing import Dict, List, Optional
import numpy as np
import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_tstt(dnl: TorchDNLMATSim) -> float:
    """Compute Total System Travel Time (TSTT).
    
    Sum of all agents' travel times (agent_metrics[:, 1]).
    Only considers agents that have arrived (status == 3).
    """
    done_mask = (dnl.status == 3)
    if done_mask.any():
        valid_legs = dnl.leg_metrics[done_mask, :, 1] > 0
        if valid_legs.any():
            return dnl.leg_metrics[done_mask, :, 1][valid_legs].sum().item() * dnl.dt
    return 0.0


def compute_mean_travel_time(dnl: TorchDNLMATSim) -> float:
    """Compute mean travel time across arrived agents."""
    done_mask = (dnl.status == 3)
    if done_mask.any():
        valid_legs = dnl.leg_metrics[done_mask, :, 1] > 0
        if valid_legs.any():
            return dnl.leg_metrics[done_mask, :, 1][valid_legs].mean().item() * dnl.dt
    return 0.0




def compute_arrival_rate(dnl: TorchDNLMATSim) -> float:
    """Fraction of agents that have arrived at their destination."""
    done_count = (dnl.status == 3).sum().item()
    return done_count / dnl.num_agents if dnl.num_agents > 0 else 0.0


# ── Travel Time Distribution ────────────────────────────────────────────────

def compute_travel_time_stats(dnl: TorchDNLMATSim) -> Dict[str, float]:
    """Compute detailed travel time statistics across arrived agents.
    
    Returns:
        Dict with keys: mean, std, min, max, median, p90, p95
    """
    done_mask = (dnl.status == 3)
    if not done_mask.any():
        return {k: 0.0 for k in ['mean', 'std', 'min', 'max', 'median', 'p90', 'p95']}
    
    valid_legs = dnl.leg_metrics[done_mask, :, 1] > 0
    if not valid_legs.any():
        return {k: 0.0 for k in ['mean', 'std', 'min', 'max', 'median', 'p90', 'p95']}
        
    tt = dnl.leg_metrics[done_mask, :, 1][valid_legs].cpu().float() * dnl.dt
    tt_np = tt.numpy()
    
    return {
        'mean': float(tt_np.mean()),
        'std': float(tt_np.std()),
        'min': float(tt_np.min()),
        'max': float(tt_np.max()),
        'median': float(np.median(tt_np)),
        'p90': float(np.percentile(tt_np, 90)),
        'p95': float(np.percentile(tt_np, 95)),
    }


# ── Episode Reward ──────────────────────────────────────────────────────────

def compute_mean_episode_reward(cumulative_rewards: Dict[str, float]) -> float:
    """Mean cumulative reward across all agents in an episode."""
    if not cumulative_rewards:
        return 0.0
    return float(np.mean(list(cumulative_rewards.values())))


def compute_reward_stats(cumulative_rewards: Dict[str, float]) -> Dict[str, float]:
    """Detailed reward statistics across all agents."""
    if not cumulative_rewards:
        return {k: 0.0 for k in ['mean', 'std', 'min', 'max']}
    
    vals = np.array(list(cumulative_rewards.values()))
    return {
        'mean': float(vals.mean()),
        'std': float(vals.std()),
        'min': float(vals.min()),
        'max': float(vals.max()),
    }


# ── Nash Gap Approximation ──────────────────────────────────────────────────

def compute_nash_gap_approx(
    env,
    policy,
    n_episodes: int = 5,
    n_agent_samples: int = 10,
    seed: Optional[int] = None,
) -> float:
    """Approximate Nash Gap by sampling best-response improvements.
    
    For each sampled agent, try all alternative routes while holding
    other agents' policies fixed. The Nash Gap is the maximum improvement
    any agent can achieve by deviating.
    
    NashGap(π) = max_i [ max_{π'_i} J_i(π'_i, π_{-i}) - J_i(π_i, π_{-i}) ]
    
    This is expensive and should only be run post-training.
    
    Args:
        env: DTAMarkovGameEnv instance
        policy: policy that implements get_actions(obs, infos)
        n_episodes: number of episodes to average over
        n_agent_samples: number of agents to sample for best-response computation
        seed: random seed for reproducibility
        
    Returns:
        Approximate Nash Gap (max improvement from unilateral deviation)
    """
    rng = np.random.default_rng(seed)
    
    # Step 1: Run a baseline episode to get baseline travel times
    baseline_travel_times = _run_episode_get_travel_times(env, policy)
    
    if not baseline_travel_times:
        return 0.0
    
    # Select agents to sample
    all_agents = list(baseline_travel_times.keys())
    n_samples = min(n_agent_samples, len(all_agents))
    sampled_agents = rng.choice(all_agents, size=n_samples, replace=False).tolist()
    
    max_improvement = 0.0
    
    for target_agent in sampled_agents:
        baseline_tt = baseline_travel_times[target_agent]
        
        # Try random alternative routes for this agent
        # Run multiple episodes where this agent deviates (random actions)
        # while others follow the policy
        best_alt_tt = baseline_tt
        
        for _ in range(n_episodes):
            alt_tt = _run_episode_with_deviation(
                env, policy, target_agent, rng
            )
            if alt_tt is not None and alt_tt < best_alt_tt:
                best_alt_tt = alt_tt
        
        improvement = baseline_tt - best_alt_tt  # Positive = deviation is better
        max_improvement = max(max_improvement, improvement)
    
    return max_improvement


def _run_episode_get_travel_times(env, policy) -> Dict[str, float]:
    """Run one episode, return per-agent travel times."""
    obs, infos = env.reset()
    
    while env.agents:
        actions = policy.get_actions(obs, infos)
        obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Extract travel times from DNL
    travel_times = {}
    done_mask = (env.dnl.status == 3)
    if done_mask.any():
        for i in range(env.dnl.num_agents):
            if env.dnl.status[i].item() == 3:
                agent_tt = env.dnl.leg_metrics[i, :, 1]
                valid_tt = agent_tt[agent_tt > 0]
                travel_times[f"agent_{i}"] = float(valid_tt.sum().item() * env.dnl.dt)
    
    return travel_times


def _run_episode_with_deviation(env, policy, deviant_agent: str, rng) -> Optional[float]:
    """Run episode where deviant_agent uses random actions, others use policy.
    
    Returns the deviant agent's travel time, or None if it didn't arrive.
    """
    obs, infos = env.reset()
    
    while env.agents:
        actions = policy.get_actions(obs, infos)
        
        # Override deviant's action with random valid action
        if deviant_agent in infos:
            mask = infos[deviant_agent].get("action_mask")
            if mask is not None and mask.sum() > 0:
                valid = np.where(mask > 0)[0]
                actions[deviant_agent] = int(rng.choice(valid))
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Get deviant's travel time
    agent_idx = int(deviant_agent.split("_")[-1])
    if env.dnl.status[agent_idx].item() == 3:
        agent_tt = env.dnl.leg_metrics[agent_idx, :, 1]
        valid_tt = agent_tt[agent_tt > 0]
        return float(valid_tt.sum().item() * env.dnl.dt)
    return None
