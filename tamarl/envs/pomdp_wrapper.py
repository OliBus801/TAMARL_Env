import torch
import numpy as np
from gymnasium import spaces

class POMDPWrapper:
    """Wrapper that turns the Bandit environment into a sequential POMDP.
    
    Instead of passing all paths at once, this wrapper intercepts the simulation
    and pauses it whenever an agent is scheduled to depart. It then requests
    an action (route choice) from the RL policy for that specific agent.
    """
    
    def __init__(self, agent_wrapper):
        self.aw = agent_wrapper
        self.bandit = self.aw.bandit
        self.device = self.bandit._device
        self.K = self.aw.K
        
        # num_envs in AgentLevelWrapper is the total number of LEGS.
        # But TorchDNL handles vehicles (agents).
        self.num_vehicles = self.bandit.num_agents
        self.max_legs = int(self.bandit.scenario.num_legs.max().item())
        
        # Route tracking
        self.max_route_len = int(self.aw.routes_offsets_csr.diff().max().item()) if len(self.aw.routes_offsets_csr) > 1 else 0
        
        # Build inverse mapping from (vehicle_idx, leg_idx) to global_leg_idx
        self.agent_leg_to_global_leg = torch.full((self.num_vehicles, self.max_legs), -1, dtype=torch.long, device=self.device)
        global_legs = torch.arange(self.aw.num_envs, device=self.device)
        self.agent_leg_to_global_leg[self.aw._leg_agent_idx, self.aw._leg_leg_idx] = global_legs
        
        # Track which leg we have already provided an action for, to avoid querying every step
        # if the agent is stuck in the departure queue (status=0 and wakeup_time <= current_step)
        self.action_provided_leg = torch.full((self.num_vehicles,), -1, dtype=torch.long, device=self.device)
        
        # Space required per vehicle
        self.slots_per_leg = self.max_route_len + 2 # first_edge + route + terminator
        self.slots_per_vehicle = self.max_legs * self.slots_per_leg
        
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1 + self.K * 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.K)
        
    def reset(self):
        V = self.num_vehicles
        
        self.path_offsets = torch.arange(0, V * self.slots_per_vehicle, self.slots_per_vehicle, device=self.device, dtype=torch.long)
        self.paths_flat = torch.full((V * self.slots_per_vehicle,), -1, device=self.device, dtype=torch.int32)
        
        self.agent_write_ptrs = self.path_offsets.clone()
        self.action_provided_leg.fill_(-1)
        
        # Pre-write Leg 0 first edges so TorchDNL initializes next_edge properly
        leg0_global = self.agent_leg_to_global_leg[:, 0]
        valid_leg0 = leg0_global >= 0
        v_valid = torch.nonzero(valid_leg0, as_tuple=True)[0]
        self.paths_flat[self.path_offsets[v_valid]] = self.aw.first_edges_all_legs[leg0_global[v_valid]].int()
        
        path_offsets_dnl = torch.zeros(V + 1, device=self.device, dtype=torch.long)
        path_offsets_dnl[:-1] = self.path_offsets
        path_offsets_dnl[-1] = V * self.slots_per_vehicle
        
        self.bandit.reset(paths_flat=self.paths_flat, path_offsets=path_offsets_dnl)
        
        self.dnl = self.bandit.dnl
        self.t = 0
        self.done = False
        
        return self._fast_forward()
        
    def _fast_forward(self):
        """Advance DNL until at least one agent is scheduled to depart at current time."""
        while self.dnl.current_step < self.bandit._max_steps:
            if self.dnl.active_agents_count == 0 and self.dnl.current_step > 0:
                self.done = True
                break
                
            active_mask = (self.dnl.wakeup_time <= self.dnl.current_step) & (self.dnl.status != 3) & (self.dnl.current_leg > self.action_provided_leg)
            active_vehicles = active_mask.nonzero(as_tuple=True)[0]
            
            if len(active_vehicles) > 0:
                active_legs = self.dnl.current_leg[active_vehicles]
                active_global_legs = self.agent_leg_to_global_leg[active_vehicles, active_legs]
                
                obs, S = self._get_obs(active_global_legs)
                return obs, active_global_legs, None, False
                
            self.dnl.step()
            self.t += 1
            
        self.done = True
        
        # Extract rewards
        self.bandit.dnl.finalize_stuck_agents()
        # travel_times is [A, MaxLegs, 2] -> sum over legs to get vehicle travel time
        # BUT the MARL script expects reward per active_global_leg. 
        # So we return it per global leg.
        tt_matrix = self.bandit.dnl.leg_metrics[:, :, 1]
        tt_obs = tt_matrix[self.aw._leg_agent_idx, self.aw._leg_leg_idx]
        rewards = -tt_obs
        
        return None, None, rewards, True
        
    def _get_obs(self, active_global_legs):
        # Global state S: self.dnl.edge_occupancy
        S = self.dnl.edge_occupancy.float()
        
        t_norm = self.t / 3600.0 # Normalize by 1 hour
        
        od_indices = self.aw.od_indices_all_legs[active_global_legs]
        fftt = self.aw.fftt_matrix[od_indices.cpu().numpy()] # [N, K]
        fftt_t = torch.from_numpy(fftt).to(self.device)
        fftt_t = torch.nan_to_num(fftt_t, nan=0.0, posinf=0.0, neginf=0.0)
        
        N = len(active_global_legs)
        obs = torch.zeros((N, 1 + self.K + self.K), device=self.device)
        obs[:, 0] = t_norm
        obs[:, 1:1+self.K] = fftt_t
        
        return obs, S

    def step(self, actions, active_global_legs):
        """Apply actions for active_global_legs and fast-forward."""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
            
        od_indices = self.aw.od_indices_all_legs[active_global_legs]
        route_rows = od_indices * self.K + actions
        
        route_starts = self.aw.routes_offsets_csr[route_rows]
        route_ends = self.aw.routes_offsets_csr[route_rows + 1]
        route_lens = (route_ends - route_starts).long()
        
        vehicles = self.aw._leg_agent_idx[active_global_legs]
        legs = self.aw._leg_leg_idx[active_global_legs]
        num_legs_per_vehicle = self.bandit.scenario.num_legs[vehicles].to(self.device)
        
        write_ptrs = self.agent_write_ptrs[vehicles]
        
        # Write first edges
        self.paths_flat[write_ptrs] = self.aw.first_edges_all_legs[active_global_legs].int()
        
        # Write route edges
        total_edges = int(route_lens.sum().item())
        if total_edges > 0:
            loe = torch.repeat_interleave(torch.arange(len(active_global_legs), device=self.device), route_lens)
            cumsum_lens = torch.zeros(len(active_global_legs) + 1, device=self.device, dtype=torch.long)
            cumsum_lens[1:] = torch.cumsum(route_lens, dim=0)
            
            edge_rank = torch.arange(total_edges, device=self.device) - cumsum_lens[loe]
            
            src_idx = route_starts[loe] + edge_rank
            dst_idx = write_ptrs[loe] + 1 + edge_rank
            
            self.paths_flat[dst_idx] = self.aw.routes_flat_csr[src_idx]
            
        # Write terminators
        term_idx = write_ptrs + 1 + route_lens
        is_last_leg = legs == (num_legs_per_vehicle - 1)
        self.paths_flat[term_idx] = torch.where(is_last_leg, -1, -2).int()
        
        # Pre-write first edge of next leg to prevent TorchDNL crash on instant transition
        has_next_leg = ~is_last_leg
        if has_next_leg.any():
            next_global_legs = active_global_legs[has_next_leg] + 1
            next_first_edges = self.aw.first_edges_all_legs[next_global_legs]
            self.paths_flat[term_idx[has_next_leg] + 1] = next_first_edges.int()
        
        # Update write pointers for NEXT legs
        self.agent_write_ptrs[vehicles] = write_ptrs + 1 + route_lens + 1
        
        # Mark action as provided for this leg
        self.action_provided_leg[vehicles] = legs
        
        # Update TorchDNL state so it knows where to move
        status = self.dnl.status[vehicles]
        is_waiting = status == 0
        next_idx = torch.where(is_waiting, write_ptrs, write_ptrs + 1)
        self.dnl.next_edge[vehicles] = self.paths_flat[next_idx].long()
            
        # We must advance the simulation by one step so we don't query the same agents again
        self.dnl.step()
        self.t += 1
        
        return self._fast_forward()
