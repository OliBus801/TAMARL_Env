
import torch
import cProfile
import pstats
import io
from torch_scatter import scatter_add, scatter_max

class TorchDNL:
    def __init__(self, 
                 edge_static: torch.Tensor, 
                 paths: torch.Tensor,
                 alpha: float = 0.15,
                 beta: float = 4.0,
                 device: str = 'cuda',
                 departure_times: torch.Tensor = None,
                 stuck_threshold: int = 10,
                 enable_profiling: bool = False):
        """
        Initialize the TorchDNL simulation engine.

        Args:
            edge_static: Tensor [E, 5] -> [length, free_flow_speed, capacity_storage (c_e), capacity_flow (D_e), ff_travel_time]
            paths: Tensor [A, MaxPathLen] -> Pre-calculated path indices for each agent.
            alpha: Congestion parameter alpha.
            beta: Congestion parameter beta.
            device: Device to run on ('cuda' or 'cpu').
            departure_times: Tensor [A] -> Departure times for each agent.
            stuck_threshold: Number of steps an agent can be stuck before being rerouted.
            enable_profiling: If True, enables cProfile profiling for the simulation.
        """
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.stuck_threshold = stuck_threshold
        
        # Profiling Initialization
        self.profiler = None
        if enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        # Move inputs to device
        self.edge_static = edge_static.to(device)
        self.paths = paths.to(device)
        
        self.num_edges = self.edge_static.shape[0]
        self.num_agents = self.paths.shape[0]
        self.max_path_len = self.paths.shape[1]

        # Edge Static Unpacking for convenience
        # [length, free_flow_speed, c_e, D_e, ff_travel_time]
        self.c_e = self.edge_static[:, 2]
        self.D_e = self.edge_static[:, 3]
        self.ff_travel_time = self.edge_static[:, 4]

        # Calculate x_crit
        # x_crit,e = D_e * ff_e
        self.x_crit = self.D_e * self.ff_travel_time

        # Edge Dynamic Tensor [E, 2] -> [occupancy (x_e), current_travel_time (tau_e)]
        self.edge_dynamic = torch.zeros((self.num_edges, 2), device=self.device, dtype=torch.float32)

        # Agent State Tensor [A, 5] 
        # [pos_status (0:node, 1:edge, 2:done, -1:not_started), current_edge_idx, target_edge_idx, rem_time, path_pointer]
        self.agent_state = torch.zeros((self.num_agents, 5), device=self.device, dtype=torch.long)

        # Departure Times
        if departure_times is not None:
             self.departure_times = departure_times.to(self.device).long()
             future_mask = self.departure_times > 0
             self.agent_state[future_mask, 0] = -1
        else:
             self.departure_times = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)

        # Agent Metrics Tensor [A, 3]
        # [start_step, accumulated_distance, total_travel_time]
        self.agent_metrics = torch.zeros((self.num_agents, 3), device=self.device, dtype=torch.float32)
        self.agent_metrics[:, 0] = self.departure_times.float()
        
        self.current_step = 0

        
        # Initialize agents
        # Assume agents start at the beginning of their first link (if path is not empty)
        # pos_status = 0 (node/waiting) or 1 (edge) - Let's assume they enter the network at step 0?
        # User prompt implies "Agent State Tensor ... rem_time".
        # We need an explicit initialization logic. For now, init to 0. 
        # Typically agents might have start times, but prompt says "Simulate >100k agents ... per step".
        # Let's assume all agents are ready to enter or on the network.
        
        # Let's set initial path_pointer to 0
        # current_edge_idx = paths[:, 0]
        # But we need to handle "entering".
        # For simplicity based on prompt:
        # We will assume agents are initialized "on node" before first link, or "on link" 0?
        # Let's initialize appropriate columns.
        
        # float for rem_time, but tensor is long? Prompt says "rem_time". 
        # Steps are discrete, so long is fine.
        # But wait, edge_dynamic x_e is occupancy (count?), tau_e is travel time (steps?).
        
        pass

    def reset(self):
        self.edge_dynamic.fill_(0)
        self.agent_state.fill_(0)
        
        # Reset based on departure times
        if hasattr(self, 'departure_times'):
             future_mask = self.departure_times > 0
             self.agent_state[future_mask, 0] = -1
             self.agent_metrics.fill_(0)
             self.agent_metrics[:, 0] = self.departure_times.float()
        else:
             self.agent_metrics.fill_(0)
             
        self.current_step = 0
        # Re-initialize specific columns if needed
        
    def calculate_congestion(self):
        """
        Update tau_e for all links based on occupancy x_e.
        Formula: tau_e = ceil(ff_e * (1 + alpha * relu((x_e - x_crit)/(c_e - x_crit))^beta))
        Formula: tau_e = max(ff_e, ff_e * ((c_e - x_crit) / (c_e - x_e))
        Vectorized, fully dense operations for speed.
        """
        x_e = self.edge_dynamic[:, 0]
        
        # Avoid division by zero
        #epsilon = 1e-6
        
        # (x_e - x_crit) / (c_e - x_crit)
        # Note: if c_e <= x_crit, denominator is epsilon (very small) -> huge congestion if x_e > x_crit. (This is probably problematic)
        numerator = self.c_e - self.x_crit # TODO: Calculate this only once
        denominator = self.c_e - x_e
        
        ratio = numerator / denominator
        
        travel_time = torch.max(self.ff_travel_time, self.ff_travel_time * ratio)
        
        # Update travel times
        self.edge_dynamic[:, 1] = torch.ceil(travel_time)

    def release_agents(self):
        """
        Activates agents whose departure time has come.
        Transitions state from -1 (Not Started) to 0 (At Node).
        """
        # Find agents that are -1 and departure_time <= current_step
        if hasattr(self, 'departure_times'):
            mask = (self.agent_state[:, 0] == -1) & (self.departure_times <= self.current_step)
            if mask.any():
                self.agent_state[mask, 0] = 0 # Set to Ready at Node
                self.agent_state[mask, 3] = 0 # Reset rem_time (wait time starts)
            
    def move_agents(self):
        """
        Decrement rem_time for all active agents.
        - If rem_time > 0: Traveling (Status 1)
        - If rem_time <= 0: Waiting (Status 0 or Status 1 finished) -> Accumulates negative wait time
        """
        # Active agents: Status 0 or 1
        active_mask = (self.agent_state[:, 0] == 0) | (self.agent_state[:, 0] == 1)
        self.agent_state[active_mask, 3] -= 1

    def nodal_arbitration(self):
        """
        Arbitrate flows between links.
        Vectorized implementation using sorting to rank agents by target.
        """
        # 1. Identify agents ready to move (Candidates)
        #    - pos_status == 0 (waiting at start node)
        #    - pos_status == 1 & rem_time <= 0 (finished link, waiting to exit)
        #    Wait, if pos_status == 1 and rem_time <= 0, they are "ready".
        
        # Current logic:
        # We need to ensure target_edge_idx is up to date for candidates.
        # But target_edge_idx is static until they move?
        # Let's assume target_edge_idx tracks the *next* edge they want to enter.
        
        # Update target_edge_idx for agents who simply finished a link and haven't selected next target yet?
        # It's cleaner to handle "find next target" here or in a preparation phase.
        # Let's do it here.
        
        # Candidates: Agents who are ready to enter a new link.
        # This implies:
        #   Case A: pos_status == 0 (Waiting to start)
        #   Case B: pos_status == 1 and rem_time == 0
        
        candidates_mask = (self.agent_state[:, 0] == 0) | ((self.agent_state[:, 0] == 1) & (self.agent_state[:, 3] <= 0))
        
        if not candidates_mask.any():
            return None
            
        candidate_indices = torch.nonzero(candidates_mask, as_tuple=True)[0]
        
        # 2. Update/Verify Target Edge for Candidates
        #    For those finishing a link, we need to look at path_pointer.
        #    Actually, we should prefer updating valid targets *before* arbitration.
        #    Let's check if target_edge_idx is already correct.
        #    If `current_edge_idx` matches `paths[ptr]`, then `target` should be `paths[ptr+1]`.
        #    Let's rely on `state_transition` to set the `target_edge_idx` for the *next* step.
        #    So: at this point, `target_edge_idx` DOES contain the edge they want to enter.
        #    We just generally check validity (e.g. not -1).
        
        targets = self.agent_state[candidate_indices, 2]
        valid_targets_mask = targets != -1
        
        # Filter out Finished agents (target == -1)
        active_candidate_indices = candidate_indices[valid_targets_mask]
        if active_candidate_indices.numel() == 0:
            # Mark agents with target -1 as done?
            # Yes, if candidates have target -1, they exit the network.
            finished_indices = candidate_indices[~valid_targets_mask]
            if finished_indices.numel() > 0:
                self.agent_state[finished_indices, 0] = 2 # Done
                # If they were on a link, we need to release occupancy?
                # Yes, state_transition handles successful moves, this is "exit".
                # We handle exits immediately here?
                # "exit" acts like moving to "VOID".
                # For simplicity, we process exits in state_transition or here.
                # Let's handle 'exits' as a special "Accept".
                pass 

        active_targets = targets[valid_targets_mask]
        
        # ... Filter out Finished ...
        
        # 3. Calculate Supply (Capacity)
        x_e = self.edge_dynamic[:, 0]
        capacity_flow = self.D_e
        capacity_storage = self.c_e
        
        free_space = torch.min(capacity_flow, capacity_storage - x_e)
        free_space = torch.clamp(free_space, min=0)
        
        # 4. Anti-Gridlock: Identify Stuck Agents
        # Stuck if rem_time <= -stuck_threshold
        # We force these agents regardless of supply.
        
        rem_times = self.agent_state[active_candidate_indices, 3]
        is_stuck = rem_times <= -self.stuck_threshold
        
        forced_indices = active_candidate_indices[is_stuck]
        normal_candidates = active_candidate_indices[~is_stuck]
        
        # We must count how much supply forced agents consume
        forced_targets = active_targets[is_stuck]
        
        # Update supply (subtract forced usage)
        if forced_indices.numel() > 0:
            # Count forced per edge
            target_counts = torch.bincount(forced_targets, minlength=self.num_edges)
            free_space = torch.clamp(free_space - target_counts, min=0)
            
        # 5. Arbitrate Normal Candidates
        # Filter active targets for normal
        normal_targets = active_targets[~is_stuck]
        
        if normal_candidates.numel() == 0:
            return forced_indices
            
        # Shuffle normal candidates
        perm = torch.randperm(normal_candidates.numel(), device=self.device)
        shuffled_indices = normal_candidates[perm]
        shuffled_targets = normal_targets[perm]
        
        # Sort by target
        sorted_sort_idx = torch.argsort(shuffled_targets)
        sorted_indices = shuffled_indices[sorted_sort_idx]
        sorted_targets_final = shuffled_targets[sorted_sort_idx]
        
        # Calculate rank
        unique_edges, inverse, counts = torch.unique_consecutive(sorted_targets_final, return_counts=True, return_inverse=True)
        
        group_starts = torch.zeros_like(unique_edges)
        if unique_edges.numel() > 0:
             group_starts[1:] = counts.cumsum(0)[:-1]
        
        agent_group_starts = group_starts[inverse]
        ranks = torch.arange(sorted_indices.numel(), device=self.device) - agent_group_starts
        
        # Check against remaining supply
        supply = free_space[unique_edges] # Updated supply
        agent_supply = supply[inverse]
        accepted_bool = ranks < agent_supply
        
        accepted_normal_indices = sorted_indices[accepted_bool]
        
        # Combine
        if forced_indices.numel() > 0:
            return torch.cat([forced_indices, accepted_normal_indices])
        else:
            return accepted_normal_indices

    def state_transition(self, accepted_indices):
        """
        Update state for accepted agents.
        """
        if accepted_indices is None or accepted_indices.numel() == 0:
            return

        # Indices of agents moving
        # We need to know FROM where and TO where.
        
        # Current State
        current_edges = self.agent_state[accepted_indices, 1] # Might be -1 if starting
        target_edges = self.agent_state[accepted_indices, 2] # Valid edge
        
        # Updates
        # 1. Occupancy x_e
        #    - Increment target_edges
        #    - Decrement current_edges (if != -1)
        
        # We can use scatter_add for batch updates on x_e
        ones = torch.ones_like(target_edges, dtype=torch.float32)
        
        # Add to target
        if hasattr(self, 'edge_dynamic'):
             # scatter_add_(dim, index, src)
             # self.edge_dynamic[:, 0].scatter_add_(0, target_edges, ones) # loops? no, inplace.
             # but scatter_add_ is from torch? Yes torch.Tensor.scatter_add_ exists.
             # torch_scatter.scatter_add is functional.
             # Let's use functional for safety with collisions? No, in-place is fine.
             # Using torch_scatter if available is faster/safer?
             # Standard torch.scatter_add_ handles duplicate indices by summing.
             self.edge_dynamic[:, 0].scatter_add_(0, target_edges, ones)
        
        # Subtract from current (only if valid input link)
        valid_from_mask = current_edges != -1
        if valid_from_mask.any():
            from_edges = current_edges[valid_from_mask]
            # self.edge_dynamic[:, 0].scatter_add_(0, from_edges, -torch.ones_like(from_edges, dtype=torch.float32))
            # Just subtract
            self.edge_dynamic[:, 0].index_add_(0, from_edges, -torch.ones(from_edges.size(0), device=self.device))
            
            # Update accumulated distance
            # We add the length of the edge they just LEFT.
            # from_edges are the edges they completed.
            lengths = self.edge_static[from_edges, 0]
            
            # accepted_indices are the agents.
            # valid_from_mask filters accepted_indices.
            valid_agents = accepted_indices[valid_from_mask]
            self.agent_metrics[valid_agents, 1] += lengths
            
        # 2. Agent State Update
        #    - current_edge = target_edge
        #    - path_pointer += 1
        #    - target_edge = paths[pointer] (Look ahead)
        #    - pos_status = 1 (Moving on link)
        #    - rem_time = tau_e[new_current_edge]
        
        # Update current
        self.agent_state[accepted_indices, 1] = target_edges
        
        # Increment pointer
        self.agent_state[accepted_indices, 4] += 1
        
        # Update target based on new pointer
        new_ptrs = self.agent_state[accepted_indices, 4]
        # Safety check for path bounds?
        # Assuming paths is padded or length managed.
        # If ptr >= max_len, target = -1.
        
        # We need to handle agents that JUST finished their path.
        # If new_ptr >= MaxPathLen => Done?
        # Or if paths[ptr] is some sentinel? (-1).
        
        # Vectorized lookup
        # We can't easily handle variable lengths without mask.
        # Assuming paths has -1 for invalid/end.
        
        valid_ptr_mask = new_ptrs < self.max_path_len
        new_targets = torch.full_like(target_edges, -1)
        
        # Only look up valid pointers
        # Need to be careful with indexing: paths[accepted_indices, new_ptrs]
        # advanced indexing
        valid_idx = accepted_indices[valid_ptr_mask]
        valid_ptrs_val = new_ptrs[valid_ptr_mask]
        
        if valid_idx.numel() > 0:
            val_targets = self.paths[valid_idx, valid_ptrs_val]
            # Scatter back to new_targets?
            # actually we can just update agent_state directly with mask
            self.agent_state[valid_idx, 2] = val_targets
            
            # Note: For those where ptr >= max_len, target stays -1 (default above? no, initialized state).
            # Wait, we need to ensure target is -1 if invalid.
            # self.agent_state[accepted_indices[~valid_ptr_mask], 2] = -1
            if (~valid_ptr_mask).any():
                self.agent_state[accepted_indices[~valid_ptr_mask], 2] = -1

        # Set Status = 1 (Moving)
        self.agent_state[accepted_indices, 0] = 1
        
        # Set rem_time = tau[current_edge]
        # current_edge is accepted_indices' NEW current edge (target_edges)
        travel_times = self.edge_dynamic[target_edges, 1].long()
        self.agent_state[accepted_indices, 3] = travel_times
        
        # Handle Exits (Done agents)
        # We identified them in nodal_arbitration as having target == -1.
        # nodal_arbitration filtered them out of 'moves'.
        # We need to handle them separately?
        # See Step comment in nodal_arbitration.
        
    def process_exits(self):
        """
        Handle agents that have finished their current link and have no next target (-1).
        Set status to Done (2) and release resource.
        """
        # Candidates: Status 1 & rem <= 0 & target == -1
        # OR Status 0 & target == -1 (if started with empty path?)
        
        # Simple logical check:
        done_mask = (self.agent_state[:, 0] == 1) & \
                    (self.agent_state[:, 3] <= 0) & \
                    (self.agent_state[:, 2] == -1)
                    
        if done_mask.any():
            done_indices = torch.nonzero(done_mask, as_tuple=True)[0]
            
            # Release occupancy of current edge
            current_edges = self.agent_state[done_indices, 1]
            # Ensure valid edge
            valid_edges_mask = current_edges != -1
            if valid_edges_mask.any():
                edges_to_free = current_edges[valid_edges_mask]
                self.edge_dynamic[:, 0].index_add_(0, edges_to_free, -torch.ones(edges_to_free.size(0), device=self.device))
                
                # Update accumulated distance for the LAST edge
                # We need lengths of these edges.
                # edge_static col 0 is length
                lengths = self.edge_static[edges_to_free, 0]
                
                # Careful mapping back to agents.
                # done_indices are indices of agents.
                # valid_edges_mask filter on done_indices.
                valid_done_indices = done_indices[valid_edges_mask]
                self.agent_metrics[valid_done_indices, 1] += lengths

            # Update total travel time
            # time = current_step - start_step
            # We assume start_step is 0 for all for now, or recorded.
            # If start_step is recorded in agent_metrics[:, 0]
            start_steps = self.agent_metrics[done_indices, 0]
            travel_times = self.current_step - start_steps
            self.agent_metrics[done_indices, 2] = travel_times
            
            # Set status to Done (2)
            self.agent_state[done_indices, 0] = 2

    def step(self):
        self.current_step += 1
        
        # 1. Update State
        self.calculate_congestion()
        
        # 2. Release Agents (New Departures)
        self.release_agents()
        
        # 3. Move agents on links (decrement time)
        self.move_agents()
        
        # 3. Handle Exits (Agents finishing path)
        #    This frees up space for new entrants.
        self.process_exits()
        
        # 4. Arbitrate and Transition (Move agents to next link)
        accepted_indices = self.nodal_arbitration()
        self.state_transition(accepted_indices)

    def get_snapshot(self):
        return self.edge_dynamic[:, 0].clone()

    def get_metrics(self):
        """
        Return dict with:
        - arrived_count
        - en_route_count
        - avg_travel_time (arrived)
        - avg_travel_dist (arrived)
        """
        done_mask = self.agent_state[:, 0] == 2
        en_route_mask = self.agent_state[:, 0] == 1
        
        arrived_count = done_mask.sum().item()
        en_route_count = en_route_mask.sum().item()
        
        avg_time = 0.0
        avg_dist = 0.0
        
        if arrived_count > 0:
            avg_time = self.agent_metrics[done_mask, 2].mean().item()
            avg_dist = self.agent_metrics[done_mask, 1].mean().item()
            
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
        en_route_mask = (self.agent_state[:, 0] == 1) | (self.agent_state[:, 0] == 0)
        # Agents not started (0) are at leg 0? 
        # path_pointer starts at 0.
        
        pointers = self.agent_state[en_route_mask, 4]
        
        # Histogram
        # We know max path len.
        hist = torch.histc(pointers.float(), bins=self.max_path_len, min=0, max=self.max_path_len-1)
        return hist

    def stop_profiling(self):
        """Disable profiling if it was enabled."""
        if self.profiler:
            self.profiler.disable()

    def print_stats(self, sort='cumtime', limit=20):
        """
        Print profiling statistics.
        
        Args:
            sort: Key to sort statistics by (e.g. 'cumtime', 'tottime').
            limit: Number of lines to print.
        """
        if self.profiler:
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort)
            ps.print_stats(limit)
            print(s.getvalue())
        else:
            print("Profiling was not enabled.")
