
import torch
import cProfile
import pstats
import io
import math

class TorchDNLMATSim:
    def __init__(self, 
                 edge_static: torch.Tensor, 
                 paths: torch.Tensor,
                 device: str = 'cuda',
                 departure_times: torch.Tensor = None,
                 stuck_threshold: int = 10,
                 dt: float = 1.0,
                 enable_profiling: bool = False):
        """
        Initialize the TorchDNLMATSim simulation engine.
        
        Args:
            edge_static: Tensor [E, 5] -> [length, free_flow_speed, capacity_storage (c_e), capacity_flow (D_e per hour), ff_travel_time]
            paths: Tensor [A, MaxPathLen] -> Pre-calculated path indices for each agent.
            device: Device to run on ('cuda' or 'cpu').
            departure_times: Tensor [A] -> Departure times for each agent.
            stuck_threshold: Time in steps an agent waits in buffer before forcing entry.
            dt: Simulation time step in seconds.
            enable_profiling: If True, enables cProfile.
        """
        self.device = device
        self.stuck_threshold = stuck_threshold
        self.dt = dt
        
        # Profiling
        self.profiler = None
        if enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # Move constants to device
        self.edge_static = edge_static.to(device)
        self.paths = paths.to(device)
        
        self.num_edges = self.edge_static.shape[0]
        self.num_agents = self.paths.shape[0]
        self.max_path_len = self.paths.shape[1]

        # Edge Attributes
        # Using contiguous arrays for potential speedup
        self.length = self.edge_static[:, 0].contiguous()
        self.free_speed = self.edge_static[:, 1].contiguous()
        self.storage_capacity = self.edge_static[:, 2].contiguous()
        
        # Flow capacity is usually veh/hour. Convert to veh/step.
        self.flow_capacity_per_step = (self.edge_static[:, 3] / 3600.0) * self.dt
        
        # Free flow travel time (steps)
        self.ff_travel_time_steps = torch.ceil(self.edge_static[:, 4] / self.dt).long().contiguous()

        # Edge Dynamic State [E]
        self.edge_occupancy = torch.zeros(self.num_edges, device=self.device, dtype=torch.long)
        
        # Flow Capacity Accumulator & Daily Limits
        self.edge_capacity_accumulator = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)
        self.step_edge_limits = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        # Agent State - Structure of Arrays (SOA)
        # 0: Waiting, 1: Traveling, 2: Buffer, 3: Done
        self.status = torch.zeros(self.num_agents, device=self.device, dtype=torch.uint8)
        self.current_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.long)
        self.next_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.long)
        self.path_ptr = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
        self.arrival_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
        self.stuck_since = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
        self.start_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
        
        # Initialize Status
        if departure_times is not None:
             self.departure_times = departure_times.to(self.device).long()
        else:
             self.departure_times = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
             
        self.start_time.copy_(self.departure_times)
        
        # Pre-set first edge as next_edge for waiting agents
        self.next_edge[:] = self.paths[:, 0]
        
        # Agent Metrics [A, 2] -> [accumulated_distance, final_travel_time]
        self.agent_metrics = torch.zeros((self.num_agents, 2), device=self.device, dtype=torch.float32)

        # Optimization: Wakeup Time [A]
        # Unified scheduler: agents are only processed if current_step >= wakeup_time
        self.wakeup_time = self.departure_times.clone()
        self.infinity = 2**62

        self.current_step = 0

    def reset(self):
        self.edge_occupancy.fill_(0)
        self.edge_capacity_accumulator.fill_(0)
        self.step_edge_limits.fill_(0)
        
        self.status.fill_(0)
        self.current_edge.fill_(-1)
        self.next_edge[:] = self.paths[:, 0]
        self.path_ptr.fill_(0)
        self.arrival_time.fill_(0)
        self.stuck_since.fill_(0)
        self.start_time.copy_(self.departure_times)
        
        self.agent_metrics.fill_(0)
        self.wakeup_time.copy_(self.departure_times)
        self.current_step = 0
        
        if self.profiler:
            self.profiler.clear()

    def _update_flow_limits(self):
        # 0. Update Flow Capacity Limits
        self.edge_capacity_accumulator += self.flow_capacity_per_step
        torch.floor(self.edge_capacity_accumulator, out=self.step_edge_limits)
        self.edge_capacity_accumulator -= self.step_edge_limits
        self.edge_capacity_accumulator.clamp_(min=0.0)

    def _handle_arrivals_A(self, active_indices, statuses):
        # --- A. Handle Arrivals (Status 1 -> 2) ---
        arrival_submask = (statuses == 1)
        arrived_agents = active_indices[arrival_submask]
        
        newly_buffered_agents = None
        
        if arrived_agents.numel() > 0:
            # 1. Update State -> Buffer (2)
            self.status[arrived_agents] = 2
            self.stuck_since[arrived_agents] = self.current_step
            
            # 2. Update Metrics
            finished_edges = self.current_edge[arrived_agents]
            lengths = self.length[finished_edges]
            
            # Only add length if not the first link (path_ptr > 0)
            # MATSim logic: "vehicle enters traffic" on link, but don't traverse it.
            
            current_ptrs = self.path_ptr[arrived_agents]
            mask_not_first = (current_ptrs > 0)
            
            if mask_not_first.any():
                 # Only add lengths for non-first links
                 agents_to_update = arrived_agents[mask_not_first]
                 lengths_to_add = lengths[mask_not_first]
                 self.agent_metrics[agents_to_update, 0] += lengths_to_add
            
            # 3. Update Pointers
            curr_ptrs = self.path_ptr[arrived_agents]
            next_ptrs = curr_ptrs + 1
            
            # Check validity
            next_edges = torch.full_like(curr_ptrs, -1)
            valid_ptr_mask = next_ptrs < self.max_path_len
            
            if valid_ptr_mask.any():
                 v_agents = arrived_agents[valid_ptr_mask]
                 v_next_ptrs = next_ptrs[valid_ptr_mask]
                 next_edges[valid_ptr_mask] = self.paths[v_agents, v_next_ptrs]
            
            self.next_edge[arrived_agents] = next_edges
            
            newly_buffered_agents = arrived_agents
            
        return newly_buffered_agents

    def _prepare_candidates_B(self, active_indices, statuses, newly_buffered_agents):
        # --- B. Handle Departures (Status 0 & 2) ---
        existing_mask = (statuses != 1) & (statuses != 3) 
        existing_candidates = active_indices[existing_mask]

        if newly_buffered_agents is not None:
            candidates = torch.cat([existing_candidates, newly_buffered_agents])
        else:
            candidates = existing_candidates
            
        return candidates

    def _handle_exits_C(self, candidates):
        # --- C. Handle Exits ---
        c_next_edges = self.next_edge[candidates]
        exit_mask = (c_next_edges == -1)
        
        if exit_mask.any():
            exiting_agents = candidates[exit_mask]
            
            # Free Capacity
            c_curr_edges_exit = self.current_edge[exiting_agents]
            valid_curr_mask = (c_curr_edges_exit >= 0)
            if valid_curr_mask.any():
                self.edge_occupancy -= torch.bincount(c_curr_edges_exit[valid_curr_mask], minlength=self.num_edges)
            
            self.status[exiting_agents] = 3
            self.wakeup_time[exiting_agents] = self.infinity 
            
            # Metrics
            start_times = self.start_time[exiting_agents]
            self.agent_metrics[exiting_agents, 1] = (self.current_step - start_times).float()
            
            candidates = candidates[~exit_mask]
            
        return candidates

    def _process_link_moves_D(self, candidates):
        if candidates.numel() == 0:
            return

        # --- D. Link Entry / Change ---
        target_edges = self.next_edge[candidates]
        current_edges = self.current_edge[candidates]
        c_arrival_times = self.arrival_time[candidates]
        
        # 1. Sort Candidates (Flow Check Grouping + FIFO)
        # Composite Key: (current_edge + 1) * SCALE + arrival_time
        # Use large scale to ensure primary sort key dominance
        sort_keys = (current_edges.long() + 1) * 1000000 + c_arrival_times.long()
        sort_idx = torch.argsort(sort_keys, stable=True)
        
        candidates_sorted = candidates[sort_idx]
        curr_sorted = current_edges[sort_idx]
        targ_sorted = target_edges[sort_idx]
        
        # 2. Check Flow Capacity (Outflow)
        unique_u, _, counts_u = torch.unique_consecutive(curr_sorted, return_counts=True, return_inverse=True)
        unique_starts = torch.zeros_like(unique_u)
        unique_starts[1:] = torch.cumsum(counts_u, dim=0)[:-1]
        item_group_starts = torch.repeat_interleave(unique_starts, counts_u)
        ranks = torch.arange(candidates_sorted.size(0), device=self.device) - item_group_starts
        
        limits = torch.zeros_like(curr_sorted, dtype=torch.float32)
        valid_mask = (curr_sorted >= 0)
        if valid_mask.any():
            limits[valid_mask] = self.step_edge_limits[curr_sorted[valid_mask]]
        limits[~valid_mask] = 999999 # Status 0: Infinite flow
        
        flow_pass_mask = (ranks < limits)
        candidates_s2 = candidates_sorted[flow_pass_mask]
        
        if candidates_s2.numel() == 0:
            return
            
        targ_s2 = targ_sorted[flow_pass_mask]
        stuck_s2 = self.stuck_since[candidates_s2]
        
        # 3. Check Storage Capacity (Inflow)
        sort_idx_2 = torch.argsort(targ_s2, stable=True)
        c_final = candidates_s2[sort_idx_2]
        t_final = targ_s2[sort_idx_2]
        stuck_final = stuck_s2[sort_idx_2]
        
        unique_v, _, counts_v = torch.unique_consecutive(t_final, return_counts=True, return_inverse=True)
        v_starts = torch.zeros_like(unique_v)
        v_starts[1:] = torch.cumsum(counts_v, dim=0)[:-1]
        item_v_starts = torch.repeat_interleave(v_starts, counts_v)
        inflow_ranks_final = torch.arange(c_final.size(0), device=self.device) - item_v_starts
        
        caps = self.storage_capacity[t_final]
        occs = self.edge_occupancy[t_final]
        avail = caps - occs
        
        is_stuck = (self.current_step - stuck_final) > self.stuck_threshold
        storage_pass = is_stuck | (inflow_ranks_final < avail)
        
        winners = c_final[storage_pass]
        
        if winners.numel() > 0:
             w_curr = self.current_edge[winners]
             w_next = self.next_edge[winners]
             
             # Update Occupancy
             valid_rem = (w_curr >= 0)
             if valid_rem.any():
                 self.edge_occupancy -= torch.bincount(w_curr[valid_rem], minlength=self.num_edges)
             self.edge_occupancy += torch.bincount(w_next, minlength=self.num_edges)
             
             # Update State
             self.status[winners] = 1 # Traveling
             self.current_edge[winners] = w_next
             
             # Increment Pointer (only if was on edge)
             self.path_ptr[winners[valid_rem]] += 1
             
             # Arrival Time
             ff_times = self.ff_travel_time_steps[w_next]
             
             # Modify for First Link Start
             # MATSim Logic : "vehicle enters traffic" on first link of their path, but don't traverse it.
             
             # w_curr (defined at start of block) holds previous edge index (if == -1, they are starting)
             is_start = (w_curr == -1)
             
             arrival_times = self.current_step + ff_times
             
             if is_start.any():
                 # Agents starting: arrival time = current step (instant arrival at end of link)
                 arrival_times[is_start] = self.current_step
             
             self.arrival_time[winners] = arrival_times
             
             # Wakeup Time
             self.wakeup_time[winners] = arrival_times

    def step(self):
        self.current_step += 1
        
        self._update_flow_limits()
        
        # 1. Unified Active Agent Search
        active_mask = (self.wakeup_time <= self.current_step)
        
        if not active_mask.any():
            return

        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        
        # Split by status
        statuses = self.status[active_indices]
        
        newly_buffered_agents = self._handle_arrivals_A(active_indices, statuses)
        
        candidates = self._prepare_candidates_B(active_indices, statuses, newly_buffered_agents)
            
        if candidates.numel() == 0:
            return
            
        candidates = self._handle_exits_C(candidates)
            
        self._process_link_moves_D(candidates)

    def stop_profiling(self):
        if self.profiler:
            self.profiler.disable()

    def print_stats(self, sort='cumtime', limit=20):
        if self.profiler:
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort)
            ps.print_stats(limit)
            print(s.getvalue())
        else:
            print("Profiling was not enabled.")

    def get_snapshot(self):
        """Returns current edge occupancy"""
        return self.edge_occupancy.clone()

    def get_metrics(self):
        """
        Return dict with:
        - arrived_count
        - en_route_count
        - avg_travel_time (arrived)
        - avg_travel_dist (arrived)
        """
        done_mask = (self.status == 3)
        # En route: Status 1 (Traveling) or 2 (Buffer)
        en_route_mask = (self.status == 1) | (self.status == 2)
        
        arrived_count = done_mask.sum().item()
        en_route_count = en_route_mask.sum().item()
        
        avg_time = 0.0
        avg_dist = 0.0
        
        if arrived_count > 0:
            avg_time = self.agent_metrics[done_mask, 1].mean().item()
            avg_dist = self.agent_metrics[done_mask, 0].mean().item()
            
        return {
            "arrived_count": arrived_count,
            "en_route_count": en_route_count,
            "avg_travel_time": avg_time,
            "avg_travel_dist": avg_dist
        }

    def get_leg_histogram(self):
        """
        Return histogram of path_pointer for en-route agents.
        """
        en_route_mask = (self.status == 1) | (self.status == 2)
        pointers = self.path_ptr[en_route_mask]
        hist = torch.histc(pointers.float(), bins=self.max_path_len, min=0, max=self.max_path_len-1)
        return hist
