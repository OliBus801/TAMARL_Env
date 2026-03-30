"""Metrics for evaluating DTA environment performance."""

import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim


def compute_tstt(dnl: TorchDNLMATSim) -> float:
    """Compute Total System Travel Time (TSTT).
    
    Sum of all agents' travel times (agent_metrics[:, 1]).
    Only considers agents that have arrived (status == 3).
    """
    done_mask = (dnl.status == 3)
    if done_mask.any():
        return dnl.agent_metrics[done_mask, 1].sum().item() * dnl.dt
    return 0.0


def compute_mean_travel_time(dnl: TorchDNLMATSim) -> float:
    """Compute mean travel time across arrived agents."""
    done_mask = (dnl.status == 3)
    if done_mask.any():
        return dnl.agent_metrics[done_mask, 1].mean().item() * dnl.dt
    return 0.0


def compute_normalized_score(tstt_rl: float, tstt_analytical: float) -> float:
    """Normalized performance score: (TSTT_RL - TSTT_analytical) / TSTT_analytical.
    
    Lower is better. 0 = matches analytical optimum.
    """
    if tstt_analytical <= 0:
        return float('inf')
    return (tstt_rl - tstt_analytical) / tstt_analytical


def compute_arrival_rate(dnl: TorchDNLMATSim) -> float:
    """Fraction of agents that have arrived at their destination."""
    done_count = (dnl.status == 3).sum().item()
    return done_count / dnl.num_agents if dnl.num_agents > 0 else 0.0
