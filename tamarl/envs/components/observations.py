"""Observation builder for the DTA Markov Game environment."""

from typing import Dict
import numpy as np
import torch
from gymnasium import spaces
from tamarl.core.dnl_matsim import TorchDNLMATSim


class ObservationBuilder:
    """Builds per-agent observations from DNL state.
    
    Observation vector for each agent:
        [current_node_id, destination_node_id, normalized_time,
         outgoing_edge_occupancies (max_out_degree),
         outgoing_ff_times (max_out_degree)]
    
    Total obs size = 3 + 2 * max_out_degree
    """

    def __init__(self, dnl: TorchDNLMATSim, max_steps: int = 86400):
        self.dnl = dnl
        self.max_steps = max_steps
        self.obs_size = 3 + 2 * dnl.max_out_degree

    def observation_space(self) -> spaces.Box:
        """Return the Gymnasium observation space."""
        return spaces.Box(
            low=-1.0, high=float('inf'),
            shape=(self.obs_size,),
            dtype=np.float32,
        )

    def build_observations(self, deciding_agent_indices: torch.Tensor) -> Dict[str, np.ndarray]:
        """Build observations for a set of deciding agents.
        
        Args:
            deciding_agent_indices: tensor of agent indices needing observations
            
        Returns:
            Dict mapping agent_id → observation ndarray
        """
        observations = {}
        if deciding_agent_indices.numel() == 0:
            return observations

        max_deg = self.dnl.max_out_degree
        
        # Gather common info
        curr_edges = self.dnl.current_edge[deciding_agent_indices]
        curr_to_nodes = self.dnl.edge_endpoints[curr_edges, 1].long()
        dests = self.dnl.destinations[deciding_agent_indices]
        norm_time = self.dnl.current_step / self.max_steps

        for i, agent_idx in enumerate(deciding_agent_indices.tolist()):
            node = curr_to_nodes[i].item()
            dest = dests[i].item()

            # Outgoing edge info
            out_edges = self.dnl.node_out_edges[node]  # [max_out_degree], -1 padded
            valid_mask = (out_edges != -1)
            
            # Occupancy of outgoing edges (normalized by storage capacity)
            occupancies = torch.zeros(max_deg, device=self.dnl.device)
            valid_edges = out_edges[valid_mask]
            if valid_edges.numel() > 0:
                occ = self.dnl.edge_occupancy[valid_edges].float()
                cap = self.dnl.storage_capacity[valid_edges]
                occupancies[valid_mask] = occ / cap.clamp(min=1.0)

            # Free-flow travel time of outgoing edges (in steps, normalized)
            ff_times = torch.zeros(max_deg, device=self.dnl.device)
            if valid_edges.numel() > 0:
                ff_times[valid_mask] = self.dnl.ff_travel_time_steps[valid_edges].float() / self.max_steps

            obs = np.zeros(self.obs_size, dtype=np.float32)
            obs[0] = float(node)
            obs[1] = float(dest)
            obs[2] = norm_time
            obs[3:3 + max_deg] = occupancies.cpu().numpy()
            obs[3 + max_deg:3 + 2 * max_deg] = ff_times.cpu().numpy()

            observations[f"agent_{agent_idx}"] = obs

        return observations

    def build_initial_observations(self) -> Dict[str, np.ndarray]:
        """Build observations for all agents at reset time.
        
        At reset, agents are waiting (status=0). We provide a minimal observation
        with their origin node (from_node of first_edge) and destination.
        """
        observations = {}
        num_agents = self.dnl.num_agents
        max_deg = self.dnl.max_out_degree

        first_edges = self.dnl._first_edges
        origin_nodes = self.dnl.edge_endpoints[first_edges, 0].long()

        for agent_idx in range(num_agents):
            origin = origin_nodes[agent_idx].item()
            dest = self.dnl.destinations[agent_idx].item()

            out_edges = self.dnl.node_out_edges[origin]
            valid_mask = (out_edges != -1)
            valid_edges = out_edges[valid_mask]

            occupancies = torch.zeros(max_deg, device=self.dnl.device)
            ff_times = torch.zeros(max_deg, device=self.dnl.device)
            if valid_edges.numel() > 0:
                ff_times[valid_mask] = self.dnl.ff_travel_time_steps[valid_edges].float() / self.max_steps

            obs = np.zeros(self.obs_size, dtype=np.float32)
            obs[0] = float(origin)
            obs[1] = float(dest)
            obs[2] = 0.0  # time = 0 at reset
            obs[3:3 + max_deg] = occupancies.cpu().numpy()
            obs[3 + max_deg:3 + 2 * max_deg] = ff_times.cpu().numpy()

            observations[f"agent_{agent_idx}"] = obs

        return observations
