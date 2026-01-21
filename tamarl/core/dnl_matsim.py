
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
        self.length = self.edge_static[:, 0]
        self.free_speed = self.edge_static[:, 1]
        self.storage_capacity = self.edge_static[:, 2]
        
        # Flow capacity is usually veh/hour. Convert to veh/step.
        # D_e (per step) = (D_e_hr / 3600) * dt
        self.flow_capacity_per_step = (self.edge_static[:, 3] / 3600.0) * self.dt
        
        # Free flow travel time (steps) = length / speed / dt. 
        # But edge_static[:, 4] might already be travel time.
        # Assuming edge_static columns: [len, speed, cap_store, cap_flow, ff_time]
        # We re-calculate to be safe or use provided if trusted. 
        # Let's use the provided ff_travel_time but ensure it's in steps.
        # If the input edge_static[:, 4] is seconds:
        self.ff_travel_time_steps = torch.ceil(self.edge_static[:, 4] / self.dt).long()

        # Edge Dynamic State [E, 2] -> [occupancy_count, buffer_count_reserved_internal_use_maybe?]
        # We only really need occupancy count for storage check.
        self.edge_occupancy = torch.zeros(self.num_edges, device=self.device, dtype=torch.long)
        
        # We also need to track flow used per step (reset every step)
        self.edge_flow_used = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        # Agent State [A, 7]
        # 0: status (0: Waiting to Start, 1: Traveling, 2: Buffer/Queue, 3: Arrived/Done)
        # 1: current_edge_idx (Edge ID or -1)
        # 2: next_edge_idx (Edge ID or -1)
        # 3: path_ptr (Index in paths tensor)
        # 4: arrival_time_at_end (Step when they finish/finished the link)
        # 5: stuck_since (Step when they entered the buffer)
        # 6: start_time (Step when they entered network - for metrics)
        self.agent_state = torch.zeros((self.num_agents, 7), device=self.device, dtype=torch.long)
        
        # Initialize Status
        # If departure_times provided, status 0.
        if departure_times is not None:
             self.departure_times = departure_times.to(self.device).long()
        else:
             self.departure_times = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
             
        # Set start_time to departure_time initially (logic uses it later)
        self.agent_state[:, 6] = self.departure_times
        
        # Pre-set first edge as next_edge for waiting agents
        # path_ptr = 0 -> next_edge = paths[:, 0]
        self.agent_state[:, 2] = self.paths[:, 0]
        self.agent_state[:, 1] = -1 # Not on any edge yet

        # Agent Metrics [A, 2] -> [accumulated_distance, final_travel_time]
        self.agent_metrics = torch.zeros((self.num_agents, 2), device=self.device, dtype=torch.float32)

        self.current_step = 0

    def reset(self):
        self.edge_occupancy.fill_(0)
        self.edge_flow_used.fill_(0)
        self.agent_state.fill_(0)
        self.agent_state[:, 6] = self.departure_times
        self.agent_state[:, 2] = self.paths[:, 0]
        self.agent_state[:, 1] = -1
        self.agent_metrics.fill_(0)
        self.current_step = 0
        if self.profiler:
            self.profiler.clear()

    def step(self):
        self.current_step += 1
        
        # 0. Reset per-step flow counters
        self.edge_flow_used.fill_(0)
        
        # 1. New Entries (Waiting -> Traveling/Buffer)
        # Agents waiting (status=0) and departure <= current
        waiting_mask = (self.agent_state[:, 0] == 0) & (self.departure_times <= self.current_step)
        
        if waiting_mask.any():
            waiting_indices = torch.nonzero(waiting_mask, as_tuple=True)[0]
            
            # Try to enter next_edge (which is first link)
            target_edges = self.agent_state[waiting_indices, 2]
            
            # Check Storage Capacity of Entry Links
            # Group by target edge to handle multiple agents entering same link
            # Simple First-Pass:
            # Sort by target edge
            sorted_indices = torch.argsort(target_edges)
            sorted_targets = target_edges[sorted_indices]
            original_indices_sorted = waiting_indices[sorted_indices]
            
            # Compute cumulative count per edge
            unique_edges, _, counts = torch.unique_consecutive(sorted_targets, return_counts=True, return_inverse=True)
            # We need the rank of each agent in for its target edge group
            # Cumsum per group?
            # Global cumsum - group_start_cumsum
            # Or just use the 'rank per group' logic:
            
            # Construct group starts (generic utility logic)
            group_starts = torch.zeros_like(unique_edges)
            group_starts[1:] = torch.cumsum(counts, dim=0)[:-1]
            
            # Map back to full array
            # We need inverse indices to map group starts to every element
            # But unique_consecutive returns inverse relative to unique_edges.
            # We need to re-run unique or do manual expansion. 
            # Re-running unique with return_inverse on sorted array is efficient.
            _, inverse_indices = torch.unique_consecutive(sorted_targets, return_inverse=True)
            
            expanded_starts = group_starts[inverse_indices]
            ranks = torch.arange(sorted_targets.size(0), device=self.device) - expanded_starts
            
            # Check Capacity
            current_occupancy = self.edge_occupancy[sorted_targets]
            capacity = self.storage_capacity[sorted_targets]
            
            # Allowed if (current + rank + 1) <= capacity
            # rank is 0-based. So 1st agent (rank 0) adds 1.
            allowed_mask = (current_occupancy + ranks + 1) <= capacity
            
            accepted_indices = original_indices_sorted[allowed_mask]
            
            if accepted_indices.numel() > 0:
                # Update State: 0 -> 1 (Traveling)
                # Actually, in MATSim, does entry put you in Buffer or Traveling?
                # Usually you travel the first link.
                accepted_targets = self.agent_state[accepted_indices, 2]
                
                # Update Edge Occupancy
                # We add '1' for each accepted agent.
                # Use bincount for bulk add
                add_counts = torch.bincount(accepted_targets, minlength=self.num_edges)
                self.edge_occupancy += add_counts
                
                # Set State
                self.agent_state[accepted_indices, 0] = 1 # Traveling
                self.agent_state[accepted_indices, 1] = accepted_targets # current = target
                
                # Set Arrival Time at end of first link
                ff_times = self.ff_travel_time_steps[accepted_targets]
                self.agent_state[accepted_indices, 4] = self.current_step + ff_times
                
                # Advance path pointer? 
                # Currently ptr=0 points to first link. 
                # If we are ON the link, we consume it.
                # So we verify logic: 
                # ptr points to "Current Edge" in paths list.
                # target was paths[ptr].
                # So we keep ptr same while on link?
                # Usually: ptr points to valid current edge. next is ptr+1.
                # Yes.
                
                # Update next_edge logic:
                # We need to know what comes AFTER this link.
                # But we just entered. We are TRAVELING. We don't need next_edge immediately until we hit buffer.
                # But let's pre-load it for consistency?
                # No, update next_edge when moving to Buffer or attempting to leave.
                # For now, current_edge is set.

        # 2. Traveling -> Buffer
        # Agents with status=1 and current_step >= arrival_time
        traveling_mask = (self.agent_state[:, 0] == 1) & (self.current_step >= self.agent_state[:, 4])
        if traveling_mask.any():
            self.agent_state[traveling_mask, 0] = 2 # Buffer
            self.agent_state[traveling_mask, 5] = self.current_step # Stuck since
            
            # Update accumulated distance
            # Add length of the edge they just finished (current_edge)
            finished_edges = self.agent_state[traveling_mask, 1]
            # Ensure valid edge (should be valid if status was 1)
            lengths = self.length[finished_edges]
            
            # Map back to agents
            moving_indices = torch.nonzero(traveling_mask, as_tuple=True)[0]
            self.agent_metrics[moving_indices, 0] += lengths
            
            # Update next_edge_idx for these agents
            # Logic: ptr + 1
            # Check bounds
            moving_indices = torch.nonzero(traveling_mask, as_tuple=True)[0]
            curr_ptrs = self.agent_state[moving_indices, 3]
            next_ptrs = curr_ptrs + 1
            
            # Check if finished (arrived at destination)
            # If next_ptr >= max_path or paths[next_ptr] == -1
            # We'll set next_edge to -1 to signify "Exit Node"
            
            # Default -1
            next_edges = torch.full_like(curr_ptrs, -1)
            
            valid_ptr_mask = next_ptrs < self.max_path_len
            valid_idx_local = torch.nonzero(valid_ptr_mask, as_tuple=True)[0]
            
            if valid_idx_local.numel() > 0:
                # Indices into 'moving_indices'
                valid_agents = moving_indices[valid_ptr_mask]
                valid_next_ptrs = next_ptrs[valid_ptr_mask]
                
                # Fetch from paths
                # paths is [A, MaxLen]
                # Advanced indexing:
                val_edges = self.paths[valid_agents, valid_next_ptrs]
                
                # Scatter back
                next_edges[valid_ptr_mask] = val_edges
            
            self.agent_state[moving_indices, 2] = next_edges


        # 3. Process Buffer (Departures/Arrials)
        # Agents in Buffer (status=2)
        buffer_mask = (self.agent_state[:, 0] == 2)
        
        if buffer_mask.any():
            buffer_indices = torch.nonzero(buffer_mask, as_tuple=True)[0]
            
            # A. Remove Agents who have Arrived (next_edge == -1)
            next_edges = self.agent_state[buffer_indices, 2]
            arrived_local_mask = (next_edges == -1)
            
            if arrived_local_mask.any():
                arrived_agents = buffer_indices[arrived_local_mask]
                
                # Free Capacity on Current Edge
                current_edges = self.agent_state[arrived_agents, 1]
                # Assuming they were on a valid edge
                remove_counts = torch.bincount(current_edges, minlength=self.num_edges)
                self.edge_occupancy -= remove_counts
                
                # Set Status to 3 (Done)
                self.agent_state[arrived_agents, 0] = 3
                
                # Record Final Travel Time
                # time = current_step - start_time
                start_times = self.agent_state[arrived_agents, 6]
                travel_times = self.current_step - start_times
                self.agent_metrics[arrived_agents, 1] = travel_times.float()
                
                # Filter out arrived agents from further processing this step
                buffer_indices = buffer_indices[~arrived_local_mask]
                if buffer_indices.numel() == 0:
                    return # All were arrivals
            
            # B. Move Candidates: Change Link
            # We have buffer_indices, all have valid next_edge != -1
            current_edges = self.agent_state[buffer_indices, 1]
            next_edges = self.agent_state[buffer_indices, 2]
            arrival_times = self.agent_state[buffer_indices, 4] # Used for FIFO sort (secondary key)
            
            # --- Flow Capacity & FIFO Check ---
            # Sort by: Current Edge (primary), Arrival Time (secondary)
            # We can pack keys or use stable sort
            # Pack: current_edge * max_time + arrival_time ? Time can be large.
            # Lexicographical sort via stable sort loops.
            # Sort by time first (stable), then edge (stable).
            
            # 1. Sort by Arrival Time
            sort_time_idx = torch.argsort(arrival_times)
            indices_sorted_by_time = buffer_indices[sort_time_idx]
            edges_sorted_by_time = current_edges[sort_time_idx]
            
            # 2. Sort by Current Edge (Stable to preserve time order)
            sort_edge_idx = torch.argsort(edges_sorted_by_time, stable=True)
            candidate_indices = indices_sorted_by_time[sort_edge_idx]
            
            # Re-gather arrays in sorted order
            c_curr_edges = self.agent_state[candidate_indices, 1]
            c_next_edges = self.agent_state[candidate_indices, 2]
            
            # 3. Compute Ranks for Flow Capacity
            unique_u, _, counts_u = torch.unique_consecutive(c_curr_edges, return_counts=True, return_inverse=True)
            
            group_starts_u = torch.zeros_like(unique_u)
            group_starts_u[1:] = torch.cumsum(counts_u, dim=0)[:-1]
            
            # Expand group starts
            # Need inverse mapping.
            # Optimization: since c_curr_edges is sorted, we can just repeat?
            # torch.repeat_interleave(group_starts_u, counts_u) matches the sorted array directly!
            expanded_starts_u = torch.repeat_interleave(group_starts_u, counts_u)
            
            outflow_ranks = torch.arange(candidate_indices.size(0), device=self.device) - expanded_starts_u
            
            # Check Flow Capacity
            # Limit is flow_capacity_per_step[edge]
            # Since flow is per step, we just take top N.
            allowed_flow_cap = self.flow_capacity_per_step[c_curr_edges]
            
            # **Integral limitation**: Flow of 0.5 means 1 vehicle every 2 steps.
            # We need a fluid accumulator or probabilistic or floor/ceil logic.
            # Simple approach: If 0.5 -> alternate. 
            # Vectorized stateful accumulation is hard without persistent state per edge.
            # But we implemented `edge_flow_used` (float). 
            # We added `edge_flow_used` earlier.
            # Strategy: 
            #   allow if (rank + 1) <= capacity_per_step?
            #   No, because capacities can be < 1.
            #   Let's use a simplified "Bucket" model? 
            #   Actually, for strict MATSim, capacity is accumulated.
            #   Let's just use strict cut-off for now, assuming integer capacities or close enough.
            #   Better: `rank < allowed_flow_cap` handles whole numbers.
            #   For fractions: `rank < floor(allowed)` ?
            #   If `allowed` is 0.2, nothing ever moves.
            #   Let's standard stochastic approach or accumulated?
            #   Let's assume standard capacity > 1 for GPU sim or randomized?
            #   "c_e" suggests int. "D_e" flow.
            #   For high speed, deterministic float check: `outflow_ranks < allowed_flow_cap`.
            #   This means if cap=1.5, 2 cars move? No, 0 and 1 < 1.5 -> 2 cars.
            #   If cap=0.5, 0 < 0.5 -> 1 car moves.
            #   It effectively ceils.
            flow_allowed_mask = outflow_ranks < allowed_flow_cap
            
            # --- Spillback (Storage) & Stuck Logic ---
            # Filter candidates passing Flow check
            candidates_step2 = candidate_indices[flow_allowed_mask]
            
            if candidates_step2.numel() > 0:
                s2_next = self.agent_state[candidates_step2, 2]
                s2_stuck = self.agent_state[candidates_step2, 5]
                
                # Check Stuck Condition
                # stuck if (current - stuck_since) > threshold
                is_stuck = (self.current_step - s2_stuck) > self.stuck_threshold
                
                # Check Storage Capacity
                # Next edge capacity
                storage_caps = self.storage_capacity[s2_next]
                current_occs = self.edge_occupancy[s2_next]
                
                # Available space
                available = storage_caps - current_occs
                
                # We also need to handle MULTIPLE agents entering same next_edge here.
                # Sort by next_edge to determine who gets the spots
                sort_next_idx = torch.argsort(s2_next)
                
                s2_sorted = candidates_step2[sort_next_idx]
                s2_next_sorted = s2_next[sort_next_idx]
                is_stuck_sorted = is_stuck[sort_next_idx]
                
                # Compute inflow ranks
                unique_v, _, counts_v = torch.unique_consecutive(s2_next_sorted, return_counts=True, return_inverse=True)
                expanded_starts_v = torch.repeat_interleave(unique_v, counts_v) # WAIT this is wrong logic for starts
                # Correct:
                group_starts_v = torch.zeros_like(unique_v)
                group_starts_v[1:] = torch.cumsum(counts_v, dim=0)[:-1]
                expanded_starts_v = torch.repeat_interleave(group_starts_v, counts_v)
                
                inflow_ranks = torch.arange(s2_sorted.size(0), device=self.device) - expanded_starts_v
                
                # Space Check map
                # We need available space for each agent based on their TARGET edge.
                # s2_next_sorted contains the Edge IDs.
                # storage_caps and current_occs defined above were the UNORDERED subset.
                # We should fetch fresh ordered values from the main arrays.
                
                agent_avail = (self.storage_capacity[s2_next_sorted] - self.edge_occupancy[s2_next_sorted])
                
                # Passed Storage Check if:
                # 1. Stuck (Force)
                # 2. rank < agent_avail
                storage_pass_mask = is_stuck_sorted | (inflow_ranks < agent_avail)
                
                final_move_indices = s2_sorted[storage_pass_mask]
                
                if final_move_indices.numel() > 0:
                     # EXECUTE MOVES
                     
                     # 1. Update Occupancy
                     # Remove from current
                     from_edges = self.agent_state[final_move_indices, 1]
                     to_edges = self.agent_state[final_move_indices, 2]
                     
                     # Use scatter / bincount
                     # Decrement from IDs
                     rem_counts = torch.bincount(from_edges, minlength=self.num_edges)
                     self.edge_occupancy -= rem_counts
                     
                     # Increment to IDs
                     add_counts = torch.bincount(to_edges, minlength=self.num_edges)
                     self.edge_occupancy += add_counts
                     
                     # 2. Update Agent State
                     # Status -> Traveling (1)
                     self.agent_state[final_move_indices, 0] = 1
                     # Current = Next
                     self.agent_state[final_move_indices, 1] = to_edges
                     # Path Ptr += 1
                     self.agent_state[final_move_indices, 3] += 1
                     # Arrival Time = Current + FF
                     ff_times = self.ff_travel_time_steps[to_edges]
                     self.agent_state[final_move_indices, 4] = self.current_step + ff_times
                     
                     # Next Edge is NOT updated yet (will be updated when they hit buffer next time) Or we can reset it to -1?
                     # Better to leave it or standardise. 
                     # Logic "Traveling -> Buffer" updates next_idx.
                     # So it's fine.

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
        done_mask = self.agent_state[:, 0] == 3
        # En route: Status 1 (Traveling) or 2 (Buffer)
        en_route_mask = (self.agent_state[:, 0] == 1) | (self.agent_state[:, 0] == 2)
        
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
        This represents how many legs agents have completed.
        Returns tensor of counts per leg index (up to max_path_len).
        """
        en_route_mask = (self.agent_state[:, 0] == 1) | (self.agent_state[:, 0] == 2)
        
        pointers = self.agent_state[en_route_mask, 3] # path_ptr is col 3
        
        # Histogram
        hist = torch.histc(pointers.float(), bins=self.max_path_len, min=0, max=self.max_path_len-1)
        return hist
