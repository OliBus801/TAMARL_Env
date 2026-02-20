import torch
import cProfile
import pstats
import io
import math
import psutil
import os
import time

class TorchDNLGEMSim:
    def __init__(self, 
                 edge_static: torch.Tensor, 
                 paths: torch.Tensor,
                 net_topology: tuple = None,
                 device: str = 'cuda',
                 departure_times: torch.Tensor = None,
                 stuck_threshold: int = 10,
                 dt: float = 1.0,
                 enable_profiling: bool = False):
        """
        Initialize the TorchDNLGEMSim simulation engine (GEMSim implementation).
        
        Args:
            edge_static: Tensor [E, 5] -> [length, free_flow_speed, capacity_storage (c_e), capacity_flow (D_e per hour), ff_travel_time]
            paths: Tensor [A, MaxPathLen] -> Pre-calculated path indices for each agent.
            net_topology: Tuple (upstream_indices, upstream_offsets) encoding CSR adjacency for nodes -> upstream links.
            device: Device to run on ('cuda' or 'cpu').
            departure_times: Tensor [A] -> Departure times for each agent.
            stuck_threshold: Time in steps an agent waits in buffer before forcing entry.
            dt: Simulation time step in seconds.
            enable_profiling: If True, enables cProfile.
        """
        self.device = device
        self.stuck_threshold = stuck_threshold
        self.dt = dt
        self.length_veh = 7.5 # meters
        self.period = 3600.0 # 1 hour for capacity definition, but capacity is already scaled
        
        # Profiling
        self.profiler = None
        if enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # Move constants to device
        self.edge_static = edge_static.to(device)
        self.paths = paths.to(device).int()
        
        self.num_edges = self.edge_static.shape[0]
        self.num_agents = self.paths.shape[0]
        self.max_path_len = self.paths.shape[1]

        # Edge Attributes
        self.length = self.edge_static[:, 0].contiguous()
        self.free_speed = self.edge_static[:, 1].contiguous()
        self.storage_capacity = self.edge_static[:, 2].contiguous()
        # Flow capacity is veh/step
        self.flow_capacity_per_step = self.edge_static[:, 3].contiguous()
        self.ff_travel_time_steps = torch.ceil(self.edge_static[:, 4] / self.dt).long().contiguous()
        
        # Topology
        # We need to process nodes, so we need max_nodes or similar
        # net_topology = (upstream_indices, upstream_offsets)
        if net_topology is None:
             raise ValueError("net_topology is required for TorchDNLGEMSim")
        
        self.upstream_indices = net_topology[0].to(device).long()
        self.upstream_offsets = net_topology[1].to(device).long()
        self.num_nodes = self.upstream_offsets.shape[0] - 1
        
        # --- 1. Structure of Arrays (SoA) & Ring Buffers ---
        
        # Calculate Buffer Sizes
        # Spatial Buffer (N_l)
        # N_l = max(floor(L_link / L_veh), 1) * lanes (implicit in storage_capacity?)
        # Actually storage_capacity is already calculated as (length * lanes) / 7.5.
        # So we can just use storage_capacity as size. 
        # CAVEAT: storage_capacity is float, we need int size.
        self.in_queue_sizes = torch.ceil(self.storage_capacity).long()
        
        # Flow Buffer (N_f) - Output Queue
        # Let's stick to valid buffer size.
        # For simplicity and robustness, let's set out_queue size = in_queue size for now, 
        # or at least a reasonable minimum.
        self.out_queue_sizes = self.in_queue_sizes.clone() 

        # Add +5 to buffers for Gridlock "Squeeze-in"
        self.storage_capacity_int = self.in_queue_sizes.clone() # Keep original logical size
        self.in_queue_sizes += 5
        self.out_queue_sizes += 5
        
        self.in_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.long)
        self.in_queue_offsets[1:] = torch.cumsum(self.in_queue_sizes, dim=0)
        self.total_in_queue_size = self.in_queue_offsets[-1].item()
        
        self.out_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.long)
        self.out_queue_offsets[1:] = torch.cumsum(self.out_queue_sizes, dim=0)
        self.total_out_queue_size = self.out_queue_offsets[-1].item()
        
        # Global Ring Buffers
        self.in_queues = torch.full((self.total_in_queue_size,), -1, device=device, dtype=torch.int32)
        self.out_queues = torch.full((self.total_out_queue_size,), -1, device=device, dtype=torch.int32)
        
        # Link Dynamic State [E]
        # Head index (where we pop), Count (how many items)
        self.in_heads = torch.zeros(self.num_edges, device=device, dtype=torch.long) # Local offset 0..size
        self.in_counts = torch.zeros(self.num_edges, device=device, dtype=torch.int32)
        
        self.out_heads = torch.zeros(self.num_edges, device=device, dtype=torch.long)
        self.out_counts = torch.zeros(self.num_edges, device=device, dtype=torch.int32)
        
        # Agent State
        # 0: Waiting, 1: Traveling (in_queue), 2: Buffer (out_queue/transit), 3: Done
        self.status = torch.zeros(self.num_agents, device=self.device, dtype=torch.uint8)
        self.current_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.int32)
        self.next_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.int32)
        self.path_ptr = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.arrival_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.stuck_since = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.start_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.enter_link_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        
        # Initialize Status
        if departure_times is not None:
             self.departure_times = departure_times.to(self.device).int()
        else:
             self.departure_times = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
             
        self.start_time.copy_(self.departure_times)
        
        # Pre-set first edge as next_edge for waiting agents
        self.next_edge[:] = self.paths[:, 0]
        
        # Wakeup logic 
        self.wakeup_time = self.departure_times.clone()
        self.infinity = 2**30
        
        # Agent Metrics [A, 2] -> [accumulated_distance, final_travel_time]
        self.agent_metrics = torch.zeros((self.num_agents, 2), device=self.device, dtype=torch.float32)

        self.current_step = 0
        
        # Flow Accumulator
        self.flow_accumulator = torch.zeros(self.num_edges, device=device, dtype=torch.float32)

        # Gridlock Management
        self.last_link_exit_time = torch.zeros(self.num_edges, device=device, dtype=torch.int32) - 1000 # Init with past

    def reset(self):
        # Reset Logic
        self.in_queues.fill_(-1)
        self.out_queues.fill_(-1)
        self.in_heads.fill_(0)
        self.in_counts.fill_(0)
        self.out_heads.fill_(0)
        self.out_counts.fill_(0)
        
        self.status.fill_(0)
        self.current_edge.fill_(-1)
        self.next_edge[:] = self.paths[:, 0]
        self.path_ptr.fill_(0)
        self.arrival_time.fill_(0)
        self.stuck_since.fill_(0)
        self.start_time.copy_(self.departure_times)
        self.enter_link_time.fill_(0)
        
        self.agent_metrics.fill_(0)
        self.wakeup_time.copy_(self.departure_times)
        self.current_step = 0
        self.flow_accumulator.fill_(0)
        self.last_link_exit_time.fill_(-1000)

        if self.profiler:
            self.profiler.clear()

    def _accumulate_capacity(self):
        # Accumulate only if occupied
        # allowing negative accumulator.
        
        active_mask = self.in_counts > 0
        if active_mask.any():
            self.flow_accumulator[active_mask] += self.flow_capacity_per_step[active_mask]
            
            # Cap at max(1.0, flow_capacity) to prevent infinite banking
            caps = torch.maximum(torch.tensor(1.0, device=self.device), self.flow_capacity_per_step[active_mask])
            self.flow_accumulator[active_mask] = torch.minimum(self.flow_accumulator[active_mask], caps)

    def _process_links(self):
        # Logic to move agents from in_queue to out_queue
        
        # 0. Accumulate Capacity first
        self._accumulate_capacity()
        
        # Iterate all links
        # We can mask links that have items in in_queue
        active_links = torch.nonzero(self.in_counts > 0, as_tuple=True)[0]
        
        if active_links.numel() == 0:
            return

        # For these links, peek at head
        
        sizes = self.in_queue_sizes[active_links]
        wrapped_heads = self.in_heads[active_links] % sizes
        global_indices = self.in_queue_offsets[active_links] + wrapped_heads
        
        agent_ids = self.in_queues[global_indices]
        
        # Check T_link condition
        enter_times = self.enter_link_time[agent_ids]
        ff_times = self.ff_travel_time_steps[active_links]
        
        # Condition 1: Time passed
        can_leave_time = (self.current_step - enter_times) >= ff_times
        
        # Condition 2: Flow Capacity
        # User: "libérer directement un agent... et aboutir à -0.99". Implies check is > 0.
        can_leave_flow = self.flow_accumulator[active_links] > 0.0
        
        # Condition 3: Out Queue Space
        out_counts = self.out_counts[active_links]
        out_sizes = self.out_queue_sizes[active_links]
        can_leave_space = out_counts < out_sizes
        
        # Final mask
        move_mask = can_leave_time & can_leave_flow & can_leave_space
        
        if not move_mask.any():
            return

        moving_links = active_links[move_mask]
        moving_agents = agent_ids[move_mask]
        
        # Execute Move
        
        # 1. Remove from In-Queue
        self.in_counts[moving_links] -= 1
        self.in_heads[moving_links] += 1 # Head moves forward
        self.in_queues[global_indices[move_mask]] = -1 # Clear (optional)
        
        # 2. Add to Out-Queue
        # Tail = (Head + Count) % Size
        o_heads = self.out_heads[moving_links]
        o_counts = self.out_counts[moving_links]
        o_sizes = self.out_queue_sizes[moving_links]
        o_offsets = self.out_queue_offsets[moving_links]
        
        o_tails = (o_heads + o_counts) % o_sizes
        o_global_idx = o_offsets + o_tails
        
        self.out_queues[o_global_idx] = moving_agents.int()
        self.out_counts[moving_links] += 1
        
        # 3. Update Flow Accumulator
        # Deduct cost (1.0 veh)
        self.flow_accumulator[moving_links] -= 1.0
        
        # 4. Update Agent Status
        self.status[moving_agents] = 2 # Buffer
        self.stuck_since[moving_agents] = self.current_step
        
        # 5. Update Link Exit Time (for Stuck Detection)
        self.last_link_exit_time[moving_links] = self.current_step

    def _process_nodes(self):
        # Kernel 2: Intersections
        # Iterate NODES
        
        # 1. Find all links with out_count > 0
        active_out_links = torch.nonzero(self.out_counts > 0, as_tuple=True)[0]
        if active_out_links.numel() == 0:
            return

        # 2. Peek agents
        o_heads = self.out_heads[active_out_links]
        o_sizes = self.out_queue_sizes[active_out_links]
        o_offsets = self.out_queue_offsets[active_out_links]
        
        wrapped_heads = o_heads % o_sizes
        global_idx = o_offsets + wrapped_heads
        
        agents = self.out_queues[global_idx]
        
        # 3. Determine Targets
        # Agent -> Next Edge
        curr_ptrs = self.path_ptr[agents]
        next_ptrs = curr_ptrs + 1
        
        # Check bounds
        valid_idx_mask = next_ptrs < self.max_path_len
        
        # Look and check for -1 (End of path)
        next_edges = torch.full_like(curr_ptrs, -1)
        if valid_idx_mask.any():
            # Only lookup valid indices
            valid_subset_agents = agents[valid_idx_mask]
            valid_subset_ptr = next_ptrs[valid_idx_mask]
            next_edges[valid_idx_mask] = self.paths[valid_subset_agents, valid_subset_ptr.long()]
            
        # Exiting if out of bounds OR next_edge is -1
        exiting_mask = (next_edges == -1)
        continuing_mask = ~exiting_mask
        
        # --- Handle Exits ---
        if exiting_mask.any():
            exiting_agents = agents[exiting_mask]
            exiting_links = active_out_links[exiting_mask]
            
            # Remove from Out-Queue
            self.out_counts[exiting_links] -= 1
            self.out_heads[exiting_links] += 1
            self.out_queues[global_idx[exiting_mask]] = -1
            
            # Update Agent
            self.status[exiting_agents] = 3 # Done
            self.wakeup_time[exiting_agents] = self.infinity
            
            # Metrics
            start_times = self.start_time[exiting_agents]
            self.agent_metrics[exiting_agents, 1] = (self.current_step - start_times).float()
            
            # Distance update on exit
            lengths = self.length[exiting_links]
            self.agent_metrics[exiting_agents, 0] += lengths

        # --- Handle Transfers ---
        if continuing_mask.any():
            cont_agents = agents[continuing_mask]
            cont_links = active_out_links[continuing_mask]
            
            # next_edges already computed
            cont_next_edges = next_edges[continuing_mask]
            
            # Sort by `next_edge` to group conflicts
            sort_idx = torch.argsort(cont_next_edges)
            
            s_agents = cont_agents[sort_idx]
            s_from_links = cont_links[sort_idx]
            s_next_edges = cont_next_edges[sort_idx]
            
            # Compute available space in next_edges
            # In-Queue availability
            # We need unique next_edges and their storage availability
            unique_next, counts = torch.unique_consecutive(s_next_edges, return_counts=True)
            
            # Current fill
            current_in_counts = self.in_counts[unique_next]
            
            # Use LOGICAL capacity for standard candidates
            logical_sizes = self.storage_capacity_int[unique_next]
            physical_sizes = self.in_queue_sizes[unique_next] # +5 buffer
            
            avail_logical = logical_sizes - current_in_counts
            avail_physical = physical_sizes - current_in_counts # Should be >= 0 usually
            
            # Group starts
            group_starts = torch.zeros_like(unique_next)
            group_starts[1:] = torch.cumsum(counts, dim=0)[:-1]
            
            indices = torch.arange(s_agents.shape[0], device=self.device)
            item_group_starts = torch.repeat_interleave(group_starts, counts)
            ranks = indices - item_group_starts
            
            item_avails_log = torch.repeat_interleave(avail_logical, counts)
            item_avails_phy = torch.repeat_interleave(avail_physical, counts)
            
            # Check if passed (Standard Logic)
            passed = ranks < item_avails_log
            
            # Stuck Logic / Squeeze-in
            if (~passed).any():
                stuck_mask = ~passed
                stuck_agents = s_agents[stuck_mask]
                stuck_targets = s_next_edges[stuck_mask]
                
                # Check 1: Time threshold
                stuck_dur = self.current_step - self.stuck_since[stuck_agents]
                is_stuck_long = stuck_dur > self.stuck_threshold
                
                # Check 2: Target Link Stale? (No exit since threshold)
                last_exit = self.last_link_exit_time[stuck_targets]
                time_since_exit = self.current_step - last_exit
                is_target_stale = time_since_exit > self.stuck_threshold 
                
                # Check 3: Physical space available (Buffer + 5)
                has_phy_space = ranks[stuck_mask] < item_avails_phy[stuck_mask]
                
                # Combine
                force_entry = is_stuck_long & is_target_stale & has_phy_space
                
                passed[stuck_mask] = force_entry
                
            # Filter winners
            winners_mask = passed
            
            if winners_mask.any():
                w_agents = s_agents[winners_mask]
                w_from = s_from_links[winners_mask]
                w_to = s_next_edges[winners_mask]
                w_ranks = ranks[winners_mask]
                
                # Execute Transfer
                # 1. Provide physical slots (indices)
                t_heads = self.in_heads[w_to]
                t_counts = self.in_counts[w_to]
                t_sizes = self.in_queue_sizes[w_to]
                t_offsets = self.in_queue_offsets[w_to]
                
                # Calculate write positions
                write_indices = (t_heads + t_counts + w_ranks) % t_sizes
                global_write = t_offsets + write_indices
                
                self.in_queues[global_write] = w_agents.int()
                
                # Update Counts (Atomically / Aggregate)
                w_unique_to, w_counts_to = torch.unique_consecutive(w_to, return_counts=True)
                self.in_counts.index_add_(0, w_unique_to, w_counts_to.int())
                
                # 2. Pop from Upstream
                self.out_counts[w_from] -= 1
                self.out_heads[w_from] += 1
                
                # 3. Update Agent
                self.status[w_agents] = 1 # Traveling
                self.current_edge[w_agents] = w_to
                self.next_edge[w_agents] = -1 
                # Update path pointer
                self.path_ptr[w_agents] += 1
                
                # enter_link_time = now
                self.enter_link_time[w_agents] = self.current_step
                
                # Metrics
                lengths = self.length[w_from]
                self.agent_metrics[w_agents, 0] += lengths
                
    def _schedule_demand(self):
        # Previously _inject_departures
        # Check active agents
        active_mask = (self.status == 0) & (self.departure_times <= self.current_step)
        if not active_mask.any():
            return

        active_agents = torch.nonzero(active_mask, as_tuple=True)[0]
        
        # Target Edges
        first_edges = self.paths[active_agents, 0]
        
        # Check Capacity of First Edges
        sort_idx = torch.argsort(first_edges)
        s_agents = active_agents[sort_idx]
        s_edges = first_edges[sort_idx]
        
        unique_edges, counts = torch.unique_consecutive(s_edges, return_counts=True)
        
        # Cap check
        current_in = self.in_counts[unique_edges]
        
        # Use logical or physical? 
        # Injecting agents should respect LOGICAL capacity?
        # MATSim injects if storage available.
        # Let's use logical to simulate realism.
        max_sizes = self.storage_capacity_int[unique_edges]
        avail = max_sizes - current_in
        
        group_starts = torch.zeros_like(unique_edges)
        group_starts[1:] = torch.cumsum(counts, dim=0)[:-1]
        
        indices = torch.arange(s_agents.shape[0], device=self.device)
        item_group_starts = torch.repeat_interleave(group_starts, counts)
        ranks = indices - item_group_starts
        
        item_avails = torch.repeat_interleave(avail, counts)
        
        passed = ranks < item_avails
        
        if passed.any():
            w_agents = s_agents[passed]
            w_edges = s_edges[passed]
            w_ranks = ranks[passed]
            
            # Insert to In-Queue
            t_heads = self.in_heads[w_edges]
            t_counts = self.in_counts[w_edges]
            t_sizes = self.in_queue_sizes[w_edges] # Physical size
            t_offsets = self.in_queue_offsets[w_edges]
            
            write_curr = (t_heads + t_counts + w_ranks) % t_sizes
            global_write = t_offsets + write_curr
            
            self.in_queues[global_write] = w_agents.int()
            
            # Update Counts
            unique_w, counts_w = torch.unique_consecutive(w_edges, return_counts=True)
            self.in_counts.index_add_(0, unique_w, counts_w.int())
            
            # Update Agent
            self.status[w_agents] = 1 # Traveling
            self.current_edge[w_agents] = w_edges
            self.enter_link_time[w_agents] = self.current_step
            # path_ptr remains 0

    def step(self):
        self.current_step += 1
        
        # New Loop Order: ProcessLinks -> ProcessNodes -> ScheduleDemand
        
        # Process Links (Internal move + Accumulate Capacity)
        self._process_links()
        
        # Process Nodes (External move + Gridlock)
        self._process_nodes()
        
        # Schedule Demand (Departures)
        self._schedule_demand()

    def stop_profiling(self):
        if self.profiler:
            self.profiler.disable()

    def print_stats(self, sort='cumtime', limit=20):
        if self.profiler:
            print("\n--- 📊 Profiling Stats ---")
            print("\n--- ⏱️ Computational Time  ---")
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort)
            ps.print_stats(limit)
            print(s.getvalue())

            compute_time = ps.total_tt
            
            print("\n--- 🧠 Peak Memory Usage  ---")
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"RAM: {mem_mb:.2f} MB")
            
            vram_mb = 0
            if self.device == 'cuda':
                vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"VRAM: {vram_mb:.2f} MB")
            
            print("-----------------------")
        else:
            print("Profiling was not enabled.")
            compute_time = 0
            mem_mb = 0
            vram_mb = 0
        
        return compute_time, mem_mb, vram_mb
    
    def get_metrics(self):
        # Same as TorchDNLMATSim
        done_mask = (self.status == 3)
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
