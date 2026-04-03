"""Observation builder for the DTA Markov Game environment."""

from typing import Dict, Tuple
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

        # Pre-compute normalised free-flow times (immutable across episodes)
        self._ff_norm = dnl.ff_travel_time_steps.float() / max_steps  # [E]

    def observation_space(self) -> spaces.Box:
        """Return the Gymnasium observation space."""
        return spaces.Box(
            low=-1.0, high=float('inf'),
            shape=(self.obs_size,),
            dtype=np.float32,
        )

    # ── Batched API (tensors in, tensors out) ─────────────────────────

    def build_observations_batched(
        self, deciding_agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Build observations for deciding agents as a single tensor.

        Args:
            deciding_agent_indices: [K] tensor of agent indices.

        Returns:
            obs: [K, obs_size] float32 tensor on self.dnl.device.
        """
        K = deciding_agent_indices.numel()
        if K == 0:
            return torch.empty((0, self.obs_size), device=self.dnl.device)

        dnl = self.dnl
        max_deg = dnl.max_out_degree

        # Current nodes and destinations  ──  all vectorised
        curr_edges = dnl.current_edge[deciding_agent_indices]           # [K]
        nodes = dnl.edge_endpoints[curr_edges, 1].long()               # [K]
        c_legs = dnl.current_leg[deciding_agent_indices]                # [K]
        dests = dnl.destinations[deciding_agent_indices, c_legs]        # [K]
        norm_time = dnl.current_step / self.max_steps

        # Outgoing edges for each node  ──  [K, max_deg], padded with -1
        out_edges = dnl.node_out_edges[nodes]                          # [K, max_deg]
        valid_mask = (out_edges != -1)                                 # [K, max_deg]

        # Clamp -1→0 so we can safely index into edge arrays
        safe_edges = out_edges.clamp(min=0)                            # [K, max_deg]

        # Occupancy / capacity  ──  vectorised gather
        occ = dnl.edge_occupancy[safe_edges].float()                   # [K, max_deg]
        cap = dnl.storage_capacity[safe_edges].clamp(min=1.0)          # [K, max_deg]
        occupancies = torch.where(valid_mask, occ / cap,
                                  torch.zeros_like(occ))               # [K, max_deg]

        # Free-flow times  ──  vectorised gather
        ff = self._ff_norm[safe_edges]                                 # [K, max_deg]
        ff_times = torch.where(valid_mask, ff, torch.zeros_like(ff))   # [K, max_deg]

        # Assemble observation tensor  ──  [K, obs_size]
        obs = torch.zeros((K, self.obs_size), device=dnl.device)
        obs[:, 0] = nodes.float()
        obs[:, 1] = dests.float()
        obs[:, 2] = norm_time
        obs[:, 3:3 + max_deg] = occupancies
        obs[:, 3 + max_deg:3 + 2 * max_deg] = ff_times

        return obs

    def build_initial_observations_batched(self) -> torch.Tensor:
        """Build observations for ALL agents at reset time as a tensor.

        Returns:
            obs: [N, obs_size] float32 tensor on self.dnl.device.
        """
        dnl = self.dnl
        N = dnl.num_agents
        max_deg = dnl.max_out_degree

        origin_nodes = dnl.edge_endpoints[dnl._first_edges, 0].long()  # [N]
        dests = dnl.destinations[:, 0]                                  # [N]

        out_edges = dnl.node_out_edges[origin_nodes]                   # [N, max_deg]
        valid_mask = (out_edges != -1)
        safe_edges = out_edges.clamp(min=0)

        ff = self._ff_norm[safe_edges]
        ff_times = torch.where(valid_mask, ff, torch.zeros_like(ff))

        obs = torch.zeros((N, self.obs_size), device=dnl.device)
        obs[:, 0] = origin_nodes.float()
        obs[:, 1] = dests.float()
        # obs[:, 2] = 0.0  already zero
        # obs[:, 3:3+max_deg] = 0.0  occupancy is zero at reset
        obs[:, 3 + max_deg:3 + 2 * max_deg] = ff_times

        return obs

    # ── Dict API (legacy, for PettingZoo compat) ─────────────────────

    def build_observations(self, deciding_agent_indices: torch.Tensor) -> Dict[str, np.ndarray]:
        """Build observations as a dict (PettingZoo interface)."""
        if deciding_agent_indices.numel() == 0:
            return {}
        obs_t = self.build_observations_batched(deciding_agent_indices)
        obs_np = obs_t.cpu().numpy()
        indices = deciding_agent_indices.tolist()
        return {f"agent_{idx}": obs_np[i] for i, idx in enumerate(indices)}

    def build_initial_observations(self) -> Dict[str, np.ndarray]:
        """Build initial observations as a dict (PettingZoo interface)."""
        obs_t = self.build_initial_observations_batched()
        obs_np = obs_t.cpu().numpy()
        return {f"agent_{i}": obs_np[i] for i in range(self.dnl.num_agents)}
