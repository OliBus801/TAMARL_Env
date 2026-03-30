"""Reward computation for the DTA Markov Game environment."""

from typing import Dict, Set
import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim


class Rewarder:
    """Dense reward: -dt per tick for all en-route agents.
    
    Maximizing cumulative reward = minimizing total travel time (γ=1).
    """

    def __init__(self, dnl: TorchDNLMATSim):
        self.dnl = dnl

    def compute_step_rewards(self, active_agents: Set[str], n_ticks: int = 1) -> Dict[str, float]:
        """Compute rewards for all active agents over n_ticks of simulation.
        
        Args:
            active_agents: set of agent_id strings currently active in the env
            n_ticks: number of DNL ticks that were advanced in this macro-step
            
        Returns:
            Dict mapping agent_id → reward (float)
        """
        rewards = {}
        dt = self.dnl.dt
        
        for agent_id in active_agents:
            agent_idx = int(agent_id.split("_")[-1])
            status = self.dnl.status[agent_idx].item()
            
            if status == 3:  # Done/arrived
                rewards[agent_id] = 0.0
            else:
                # En-route: penalty proportional to time spent traveling
                rewards[agent_id] = -dt * n_ticks
        
        return rewards

    def compute_batch_rewards(self, active_agent_indices: torch.Tensor, n_ticks: int = 1) -> torch.Tensor:
        """Vectorised reward computation.
        
        Returns:
            Tensor [K] of rewards for each agent in active_agent_indices
        """
        statuses = self.dnl.status[active_agent_indices]
        rewards = torch.full((active_agent_indices.size(0),), -self.dnl.dt * n_ticks, 
                           device=self.dnl.device)
        rewards[statuses == 3] = 0.0  # Done agents get 0
        return rewards
