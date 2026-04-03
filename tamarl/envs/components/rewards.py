"""Reward computation for the DTA Markov Game environment."""

from typing import Dict, Set
import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim


class Rewarder:
    """Dense reward: -dt per tick spent traveling.
    
    Maximizing cumulative reward = minimizing total travel time (γ=1).
    This implementation tracks exact continuous travel time dynamically, ensuring 
    Sum(step_rewards) over an episode exactly equals -Total Travel Time.
    """

    def __init__(self, dnl: TorchDNLMATSim):
        self.dnl = dnl
        self._prev_tt_sums: torch.Tensor = None
        self.reset()

    def reset(self):
        """Reset the continuous travel time trackers for a new episode."""
        self._prev_tt_sums = torch.zeros(self.dnl.num_agents, device=self.dnl.device, dtype=torch.float32)

    def _get_continuous_tt(self) -> torch.Tensor:
        """Calculate continuous travel time for all agents at the current step."""
        # Sum of all completed legs' TT
        tt_sum = self.dnl.leg_metrics[:, :, 1].sum(dim=1)
        
        # Add active current leg (en-route or waiting for route)
        active_mask = (self.dnl.status == 0) | (self.dnl.status == 1) | (self.dnl.status == 2)
        c_leg = self.dnl.current_leg
        deps = self.dnl.leg_departure_times.gather(1, c_leg.unsqueeze(1)).squeeze(1)
        
        tt_active = torch.zeros_like(tt_sum)
        tt_active[active_mask] = (self.dnl.current_step - deps[active_mask]).float()
        
        return tt_sum + tt_active

    def compute_step_rewards(self, active_agents: Set[str], n_ticks: int = 1) -> Dict[str, float]:
        """Compute rewards for all active agents.
        
        The reward is strictly proportional to the exact delta of accumulated travel time 
        since the last time step_rewards was called.
        """
        if not active_agents:
            return {}

        # Convert str IDs to tensor indices once
        indices = torch.tensor(
            [int(a.split("_")[-1]) for a in active_agents],
            device=self.dnl.device, dtype=torch.long,
        )
        rewards_t = self.compute_batch_rewards(indices, n_ticks)
        rewards_np = rewards_t.cpu().numpy()

        # Build dict from the tensor results
        return {a: float(rewards_np[i]) for i, a in enumerate(active_agents)}

    def compute_batch_rewards(self, active_agent_indices: torch.Tensor, n_ticks: int = 1) -> torch.Tensor:
        """Vectorised reward computation."""
        curr_tt_sums = self._get_continuous_tt()
        tt_deltas = curr_tt_sums[active_agent_indices] - self._prev_tt_sums[active_agent_indices]
        
        rewards = -tt_deltas * self.dnl.dt
        
        # Update trackers only for evaluated agents
        self._prev_tt_sums[active_agent_indices] = curr_tt_sums[active_agent_indices]
        
        return rewards
