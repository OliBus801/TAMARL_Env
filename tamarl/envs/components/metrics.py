"""Metrics for evaluating DTA environment performance."""

from typing import Dict, List, Optional
import numpy as np
import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_all_leg_travel_times(dnl: TorchDNLMATSim) -> torch.Tensor:
    """Collect travel times for ALL agents, including those still en-route.

    For arrived agents (status == 3): uses the recorded leg_metrics.
    For en-route agents (status 0/1/2/4): uses current_step - departure_time
    for the leg they are currently on (lower-bound travel time).

    Returns:
        1-D float tensor of per-leg travel times (in simulation steps).
        May be empty if no valid times exist.
    """
    parts: list[torch.Tensor] = []

    # 1. Completed legs from arrived agents
    done_mask = (dnl.status == 3)
    if done_mask.any():
        done_tt = dnl.leg_metrics[done_mask, :, 1]  # [D, MaxLegs]
        valid = done_tt > 0
        if valid.any():
            parts.append(done_tt[valid])

    # 2. Completed intermediate legs of en-route agents (legs before the current one)
    en_route_mask = (dnl.status == 1) | (dnl.status == 2) | (dnl.status == 4)
    # Also include status==0 agents that have already departed at least once
    waiting_departed = (dnl.status == 0) & (dnl.current_leg > 0)
    en_route_mask = en_route_mask | waiting_departed

    if en_route_mask.any():
        er_agents = torch.nonzero(en_route_mask, as_tuple=True)[0]
        c_legs = dnl.current_leg[er_agents]  # current leg index

        # 2a. Already-completed intermediate legs (indices 0..c_leg-1)
        er_tt = dnl.leg_metrics[er_agents, :, 1]  # [K, MaxLegs]
        completed_mask = er_tt > 0
        # Exclude the current leg slot – it hasn't been finalized by the engine
        leg_range = torch.arange(er_tt.shape[1], device=dnl.device).unsqueeze(0)  # [1, MaxLegs]
        completed_mask = completed_mask & (leg_range < c_legs.unsqueeze(1))
        if completed_mask.any():
            parts.append(er_tt[completed_mask])

        # 2b. Current (in-progress) leg: travel_time = current_step - departure_time
        dep_times = dnl.leg_departure_times[er_agents, c_legs]  # [K]
        # Status 1/2/4 agents are by definition on an active leg.
        # Status 0 agents (between legs) haven't started their current leg yet,
        # so only their completed legs (2a) count.
        actually_en_route = (dnl.status[er_agents] != 0)
        if actually_en_route.any():
            in_progress_tt = (dnl.current_step - dep_times[actually_en_route]).float()
            # Clamp to 0 in case of edge timing quirks
            in_progress_tt = torch.clamp(in_progress_tt, min=0.0)
            parts.append(in_progress_tt)

    if not parts:
        return torch.empty(0, device=dnl.device)
    return torch.cat(parts)


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_tstt(dnl: TorchDNLMATSim) -> float:
    """Compute Total System Travel Time (TSTT).

    Includes all agents: arrived agents use their recorded travel time;
    en-route agents use current_step - departure_time as a lower bound
    for their current leg.
    """
    tt = _get_all_leg_travel_times(dnl)
    if tt.numel() > 0:
        return tt.sum().item() * dnl.dt
    return 0.0


def compute_mean_travel_time(dnl: TorchDNLMATSim) -> float:
    """Compute mean travel time across all agents (arrived + en-route)."""
    tt = _get_all_leg_travel_times(dnl)
    if tt.numel() > 0:
        return tt.mean().item() * dnl.dt
    return 0.0


def compute_arrival_rate(dnl: TorchDNLMATSim) -> float:
    """Fraction of agents that have arrived at their destination."""
    done_count = (dnl.status == 3).sum().item()
    return done_count / dnl.num_agents if dnl.num_agents > 0 else 0.0


# ── Travel Time Distribution ────────────────────────────────────────────────

def compute_travel_time_stats(dnl: TorchDNLMATSim) -> Dict[str, float]:
    """Compute detailed travel time statistics across all agents.

    Includes en-route agents whose current-leg travel time is estimated
    as current_step - departure_time.

    Returns:
        Dict with keys: mean, std, min, max, median, p90, p95
    """
    tt = _get_all_leg_travel_times(dnl)
    if tt.numel() == 0:
        return {k: 0.0 for k in ['mean', 'std', 'min', 'max', 'median', 'p90', 'p95']}

    tt_sec = tt.cpu().float() * dnl.dt
    tt_np = tt_sec.numpy()

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




# ── Nash Regret & Relative Gap (Empirical O(N)) ─────────────────────────────



def compute_empirical_nash_metrics_tensor(
    actual_travel_times: torch.Tensor, # [N] Temps réels vécus
    actions: torch.Tensor,             # [N] Index du chemin pris (0 à K-1)
    estimated_times: torch.Tensor,     # [N, K] Temps estimés par les N-Curves via l'évaluateur
    epsilon_threshold: float = 60.0
) -> dict:
    """
    Calcule le Nash Regret et le Relative Gap de façon 100% tensorisée.
    """
    N = actual_travel_times.shape[0]
    if N == 0:
        return {'mean_regret': 0.0, 'max_regret': 0.0, 'epsilon_compliance_rate': 0.0}

    # 1. Copie pour ne pas altérer le tenseur de l'évaluateur
    agent_options = estimated_times.clone()
    
    # 2. RÈGLE D'OR : Masquer l'action prise avec infini
    batch_indices = torch.arange(N, device=actions.device)
    agent_options[batch_indices, actions] = float('inf')
    
    # 3. Meilleure alternative (Best-Response)
    best_alt_tt, _ = torch.min(agent_options, dim=1)
    
    # 4. Regret (borné à 0)
    regrets = torch.clamp(actual_travel_times - best_alt_tt, min=0.0)
    
    # 5. Synthèse
    total_regret = regrets.sum()
    epsilon_compliance = (regrets <= epsilon_threshold).float().mean().item()
    
    return {
        'mean_regret': regrets.mean().item(),
        'max_regret': regrets.max().item(),
        'epsilon_compliance_rate': epsilon_compliance
    }
