"""Observation builder for the DTA Markov Game environment."""

from typing import Dict, Tuple
import heapq
import numpy as np
import torch
from gymnasium import spaces
from tamarl.core.dnl_matsim import TorchDNLMATSim


class ObservationBuilder:
    """Builds per-agent observations from DNL state.
    
    Observation vector for each agent:
        [current_node_id, destination_node_id, normalized_time,
         outgoing_edge_occupancies (max_out_degree),
         outgoing_ff_times (max_out_degree),
         aon_recommended_edge_one_hot (max_out_degree)]
    
    Total obs size = 3 + 3 * max_out_degree
    """

    FREE_THRESH = 0.5
    JAM_THRESH = 0.8
    N_TIME_BINS = 48

    def __init__(self, dnl: TorchDNLMATSim, max_steps: int = 86400):
        self.dnl = dnl
        self.max_steps = max_steps
        self.obs_size = 3 + 3 * dnl.max_out_degree

        # Pre-compute normalised free-flow times (immutable across episodes)
        self._ff_norm = dnl.ff_travel_time_steps.float() / max_steps  # [E]
        
        # Pre-compute AoN Shortest Paths
        self._precompute_aon_gps()

    def _precompute_aon_gps(self):
        """Pre-computes Reverse Dijkstra for all unique destinations to provide AoN GPS signal."""
        dnl = self.dnl
        unique_dests = torch.unique(dnl.destinations).cpu().numpy()
        
        # Matrix to store the local index [0..max_deg-1] of the best outgoing edge for (node, dest)
        self._aon_edge_idx = torch.full(
            (dnl.num_nodes, dnl.num_nodes), -1, dtype=torch.long, device=dnl.device
        )
        
        # Build reverse adjacency list: rev_adj[v] = [(u, edge_id, ff_time), ...]
        rev_adj = [[] for _ in range(dnl.num_nodes)]
        endpoints = dnl.edge_endpoints.cpu().numpy()
        ff_times = dnl.ff_travel_time_steps.cpu().numpy()
        
        for e in range(dnl.num_edges):
            u, v = endpoints[e]
            rev_adj[int(v)].append((int(u), e, float(ff_times[e])))
            
        out_edges_tensor = dnl.node_out_edges.cpu().numpy()  # [num_nodes, max_deg]
            
        for dest in unique_dests:
            dest = int(dest)
            if dest == -1: 
                continue  # Skip padding
                
            dist = {dest: 0.0}
            pq = [(0.0, dest)]
            best_edge = {}  # node -> edge_id taking node to shortest path
            
            # Standard Dijkstra on reverse graph (from dest to all nodes)
            while pq:
                d, v = heapq.heappop(pq)
                if d > dist.get(v, float('inf')):
                    continue
                    
                for u, edge_id, ff in rev_adj[v]:
                    new_d = d + ff
                    if new_d < dist.get(u, float('inf')):
                        dist[u] = new_d
                        best_edge[u] = edge_id
                        heapq.heappush(pq, (new_d, u))
                        
            # Map global edge_id back to local out-degree relative index [0..max_deg-1]
            aon_arr = np.full(dnl.num_nodes, -1, dtype=np.int64)
            for u, edge_id in best_edge.items():
                out_edges_u = out_edges_tensor[u]
                pos = np.where(out_edges_u == edge_id)[0]
                if len(pos) > 0:
                    aon_arr[u] = pos[0]
            
            # Move to device tensor
            self._aon_edge_idx[:, dest] = torch.from_numpy(aon_arr).to(dnl.device)

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
        
        # Discretise time into 48 bins
        norm_time = float(min((dnl.current_step * self.N_TIME_BINS) // self.max_steps, self.N_TIME_BINS - 1))

        # Outgoing edges for each node  ──  [K, max_deg], padded with -1
        out_edges = dnl.node_out_edges[nodes]                          # [K, max_deg]
        valid_mask = (out_edges != -1)                                 # [K, max_deg]

        # Clamp -1→0 so we can safely index into edge arrays
        safe_edges = out_edges.clamp(min=0)                            # [K, max_deg]

        # Occupancy / capacity  ──  vectorised gather
        occ = dnl.edge_occupancy[safe_edges].float()                   # [K, max_deg]
        cap = dnl.storage_capacity[safe_edges].clamp(min=1.0)          # [K, max_deg]
        
        raw_density = occ / cap
        
        # Discretise into 3 states: free (0.0), resistance (1.0), jam (2.0)
        discrete_density = torch.zeros_like(raw_density)
        discrete_density[raw_density >= self.FREE_THRESH] = 1.0
        discrete_density[raw_density > self.JAM_THRESH] = 2.0
        
        occupancies = torch.where(valid_mask, discrete_density,
                                  torch.zeros_like(discrete_density))  # [K, max_deg]

        # Free-flow times  ──  vectorised gather
        ff = self._ff_norm[safe_edges]                                 # [K, max_deg]
        ff_times = torch.where(valid_mask, ff, torch.zeros_like(ff))   # [K, max_deg]

        # AoN GPS Indicator ── one-hot mapping
        aon_idx = self._aon_edge_idx[nodes, dests]                     # [K]
        aon_one_hot = torch.zeros((K, max_deg), device=dnl.device)
        valid_aon = aon_idx >= 0
        if valid_aon.any():
            # scatter expects [K, 1], so we unsqueeze and convert to int64
            aon_one_hot[valid_aon] = aon_one_hot[valid_aon].scatter_(
                1, aon_idx[valid_aon].unsqueeze(1), 1.0
            )

        # Assemble observation tensor  ──  [K, obs_size]
        obs = torch.zeros((K, self.obs_size), device=dnl.device)
        obs[:, 0] = nodes.float()
        obs[:, 1] = dests.float()
        obs[:, 2] = norm_time
        obs[:, 3:3 + max_deg] = occupancies
        obs[:, 3 + max_deg:3 + 2 * max_deg] = ff_times
        obs[:, 3 + 2 * max_deg:3 + 3 * max_deg] = aon_one_hot

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

        # AoN GPS Indicator
        aon_idx = self._aon_edge_idx[origin_nodes, dests]
        aon_one_hot = torch.zeros((N, max_deg), device=dnl.device)
        valid_aon = aon_idx >= 0
        if valid_aon.any():
            aon_one_hot[valid_aon] = aon_one_hot[valid_aon].scatter_(
                1, aon_idx[valid_aon].unsqueeze(1), 1.0
            )

        obs = torch.zeros((N, self.obs_size), device=dnl.device)
        obs[:, 0] = origin_nodes.float()
        obs[:, 1] = dests.float()
        # obs[:, 2] = 0.0  already zero
        # obs[:, 3:3+max_deg] = 0.0  occupancy is zero at reset
        obs[:, 3 + max_deg:3 + 2 * max_deg] = ff_times
        obs[:, 3 + 2 * max_deg:3 + 3 * max_deg] = aon_one_hot

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
