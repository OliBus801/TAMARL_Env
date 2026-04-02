"""Metrics for evaluating DTA environment performance."""

from typing import Dict, List, Optional
import numpy as np
import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim
from tamarl.envs.components.time_dependent_dijkstra import build_adjacency_list, compute_td_shortest_paths


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

# ── Relative Gap ────────────────────────────────────────────────────────────

def compute_relative_gap(dnl: TorchDNLMATSim, link_tt_interval: float = 300.0) -> float:
    """Computes the Relative Gap (RGap) metric over all arrived agents.
    
    RGap = sum(t_i - t_i_SP) / sum(t_i)
    """
    if not getattr(dnl, 'collect_link_tt', False):
        return float('inf')
        
    tt_matrix = dnl.get_dynamic_link_travel_times()
    if tt_matrix is None:
        return float('inf')
        
    tt_matrix_np = tt_matrix.cpu().numpy()
    
    done_mask = (dnl.status == 3).cpu().numpy()
    if not done_mask.any():
        return 0.0
        
    done_agents = np.where(done_mask)[0]
    leg_metrics = dnl.leg_metrics.cpu().numpy()
    
    total_real_tt = 0.0
    
    start_times_list = []
    origins_list = []
    destinations_list = []
    
    all_first_edges = dnl._all_first_edges.cpu().numpy()
    dests = dnl.destinations.cpu().numpy()
    edge_endpoints = dnl.edge_endpoints.cpu().numpy()
    leg_departure_times = dnl.leg_departure_times.cpu().numpy()
    num_legs = dnl.num_legs.cpu().numpy()
    
    for agent_id in done_agents:
        for leg_idx in range(num_legs[agent_id]):
            real_tt = leg_metrics[agent_id, leg_idx, 1]
            if real_tt <= 0:
                continue
                
            first_edge = all_first_edges[agent_id, leg_idx]
            # MATSim spawns agents into the capacity buffer of the their first link,
            # meaning their physical start node is the destination of the first link.
            origin_node = edge_endpoints[first_edge, 1]
            dest_node = dests[agent_id, leg_idx]
            dep_time_steps = leg_departure_times[agent_id, leg_idx]
            dep_time_sec = dep_time_steps * dnl.dt
            
            start_times_list.append(dep_time_sec)
            origins_list.append(origin_node)
            destinations_list.append(dest_node)
            
            # Since real_tt is already in seconds, just add it
            total_real_tt += real_tt
            
    if len(start_times_list) == 0 or total_real_tt <= 0:
        return 0.0
        
    start_times_np = np.array(start_times_list, dtype=np.float32)
    origins_np = np.array(origins_list, dtype=np.int32)
    dests_np = np.array(destinations_list, dtype=np.int32)
    
    adj = build_adjacency_list(dnl.num_nodes, edge_endpoints)
    t_sp = compute_td_shortest_paths(
        adj=adj,
        start_times=start_times_np,
        origin_nodes=origins_np,
        destination_nodes=dests_np,
        tt_matrix=tt_matrix_np,
        interval=float(link_tt_interval)
    )
    
    # Exclude unreachable agents from the gap, though in a sane scenario all matched
    valid_sp_mask = (t_sp != float('inf'))
    if not valid_sp_mask.any():
        return 0.0
        
    total_sp_tt = np.sum(t_sp[valid_sp_mask])
    
    # Ensure RGap doesn't become negative due to precision
    rgap = max(0.0, (total_real_tt - float(total_sp_tt)) / total_real_tt)
    return float(rgap)
