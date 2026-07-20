import numpy as np
import torch

class POMDPWrapper:
    """POMDP Wrapper around TorchDNL.
    
    Steps through time. At each step where agents depart, it returns their observations.
    The agent takes actions, we update their paths in TorchDNL, and fast-forward to the next departure.
    """
    def __init__(self, agent_wrapper):
        self.aw = agent_wrapper
        self.bandit = self.aw.bandit
        self.dnl = None
        self.device = self.aw._device
        
        self.num_agents = self.aw.num_envs
        self.K = self.aw.K
        
    def reset(self):
        # We need to initialize TorchDNL with a dummy paths_flat that has enough space.
        # But wait, AgentLevelWrapper.step() builds the exact paths_flat based on actions.
        # If we just allocate maximum possible length for paths_flat, we can mutate it.
        
        # Find max route length
        route_lens = self.aw.routes_offsets_csr[1:] - self.aw.routes_offsets_csr[:-1]
        self.max_route_len = int(route_lens.max().item())
        
        A = self.num_agents
        
        # Each agent gets exactly max_route_len edges, prepended with first_edge, and a terminator.
        # So we need max_route_len + 2 spaces per agent.
        
        self.path_offsets = torch.arange(0, (A + 1) * (self.max_route_len + 2), self.max_route_len + 2, device=self.device, dtype=torch.long)
        self.paths_flat = torch.full((A * (self.max_route_len + 2),), -1, device=self.device, dtype=torch.int32)
        
        # Reset bandit with dummy paths
        self.bandit.reset(paths_flat=self.paths_flat, path_offsets=self.path_offsets)
        self.dnl = self.bandit.dnl
        
        # Sort departure times to know when to stop
        # Assuming 1 leg per agent, departure_times is [A]
        self.departure_times = self.dnl.departure_times.clone()
        
        self.t = 0
        self.done = False
        
        # Find first active agents
        return self._fast_forward()
        
    def _fast_forward(self):
        """Advance DNL until at least one agent is scheduled to depart at current time."""
        while self.dnl.current_step < self.bandit._max_steps:
            if self.dnl.active_agents_count == 0 and self.dnl.current_step > 0:
                self.done = True
                break
                
            # Find agents departing at current_step
            # In TorchDNL, agents depart if start_time == current_step
            active_mask = (self.dnl.start_time == self.dnl.current_step) & (self.dnl.status == 0)
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            
            if len(active_indices) > 0:
                # We have agents departing! Return observations
                obs, S = self._get_obs(active_indices)
                return obs, active_indices, None, False
                
            self.dnl.step()
            self.t += 1
            
        self.done = True
        
        # Extract rewards
        self.bandit.dnl.finalize_stuck_agents()
        travel_times = self.bandit.dnl.leg_metrics[:, 0, 1] # Assuming 1 leg
        rewards = -travel_times
        return None, None, rewards, True
        
    def _get_obs(self, active_indices):
        # Global state S: self.dnl.edge_occupancy
        S = self.dnl.edge_occupancy.float()
        
        # Partial obs Oi for each active agent
        # Oi = [t_norm, FFTT_1, ..., FFTT_K, Occ_1, ..., Occ_K] (simplified)
        # We will extract it efficiently.
        
        t_norm = self.t / 3600.0 # Normalize by 1 hour (example)
        
        od_indices = self.aw.od_indices_all_legs[active_indices]
        fftt = self.aw.fftt_matrix[od_indices.cpu().numpy()] # [N, K]
        fftt_t = torch.from_numpy(fftt).to(self.device)
        
        # To get occupancy of the routes, we'd need to map route edges to S.
        # For baseline, we can just return a flat vector per agent.
        
        N = len(active_indices)
        obs = torch.zeros((N, 1 + self.K + self.K), device=self.device)
        obs[:, 0] = t_norm
        obs[:, 1:1+self.K] = fftt_t
        # Occupancy omitted for speed in this prototype, or we can add it later.
        
        return obs, S

    def step(self, actions, active_indices):
        """Apply actions for active_indices and fast-forward."""
        # actions is [N] array of route choices (0 to K-1)
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
            
        od_indices = self.aw.od_indices_all_legs[active_indices]
        route_rows = od_indices * self.K + actions
        
        route_starts = self.aw.routes_offsets_csr[route_rows]
        route_ends = self.aw.routes_offsets_csr[route_rows + 1]
        route_lens = route_ends - route_starts
        
        # Write into paths_flat
        for i, agent_idx in enumerate(active_indices):
            start = route_starts[i]
            end = route_ends[i]
            length = route_lens[i]
            
            # route edges
            edges = self.aw.routes_flat_csr[start:end]
            
            # where to write?
            write_start = self.path_offsets[agent_idx]
            
            # First edge
            self.paths_flat[write_start] = self.aw.first_edges_all_legs[agent_idx].int()
            
            # Rest of the route
            if length > 0:
                self.paths_flat[write_start + 1 : write_start + 1 + length] = edges
                
            # Terminator
            self.paths_flat[write_start + 1 + length] = -1
            
            # CRITICAL: update next_edge because TorchDNL reads it at reset()
            self.dnl.next_edge[agent_idx] = self.paths_flat[write_start].long()
            
        # We must advance the simulation by one step so we don't query the same agents again
        self.dnl.step()
        self.t += 1
        
        # Fast forward
        return self._fast_forward()
