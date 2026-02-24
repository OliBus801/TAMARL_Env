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
        Initialize the TorchDNLGEMSim simulation engine according to the GEMSim implementation (Saprykin et al., 2025).
        
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
        self.squeeze_margin = 5
        
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
        self.ff_travel_time_steps = torch.ceil(self.edge_static[:, 4] / self.dt).int().contiguous()
        
        # Topology
        if net_topology is None:
             raise ValueError("net_topology is required for TorchDNLGEMSim")
        
        self.upstream_indices = net_topology[0].to(device).long()
        self.upstream_offsets = net_topology[1].to(device).long()
        self.num_nodes = self.upstream_offsets.shape[0] - 1
        
        # --- 1. Structure of Arrays (SoA) & Ring Buffers ---
        
        # -- 1.1 Buffer Descriptors --
        # queue_offset : start index of the link's queue in the global ring buffer.
        # queue_size : size of the link's queue.

        # Spatial Buffer (N_l)
        self.in_queue_sizes = torch.clamp(self.storage_capacity.int(), min=1)
        
        # Capacity Buffer (N_f)
        self.out_queue_sizes = torch.clamp(self.flow_capacity_per_step.int(), min=1)

        # Add squeeze_margin to the buffer sizes for Gridlock "Squeeze-in"
        self.in_queue_sizes += self.squeeze_margin
        self.out_queue_sizes += self.squeeze_margin

        # Calculate queue_offset
        self.in_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.int32)
        self.in_queue_offsets[1:] = torch.cumsum(self.in_queue_sizes, dim=0)
        self.total_in_queue_size = self.in_queue_offsets[-1].item()
        
        self.out_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.int32)
        self.out_queue_offsets[1:] = torch.cumsum(self.out_queue_sizes, dim=0)
        self.total_out_queue_size = self.out_queue_offsets[-1].item()
        
        # -- 1.2 Dynamic State Buffer Descriptors --
        # q_cur : current index of the link's first occupied cell, relative to q_off.
        # q_cnt : number of occupied cells in the link's queue.
        self.in_cur = torch.zeros(self.num_edges, device=device, dtype=torch.short)
        self.in_cnt = torch.zeros(self.num_edges, device=device, dtype=torch.short)
        
        self.out_cur = torch.zeros(self.num_edges, device=device, dtype=torch.short)
        self.out_cnt = torch.zeros(self.num_edges, device=device, dtype=torch.short)

        # -- 1.3 Global Buffers --
        self.in_queues = torch.full((self.total_in_queue_size,), -1, device=device, dtype=torch.int32)
        self.out_queues = torch.full((self.total_out_queue_size,), -1, device=device, dtype=torch.int32)
        
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
        self.last_link_exit_time = torch.zeros(self.num_edges, device=device, dtype=torch.int32) - 1000

        # --- Pre-allocated scratch buffers ---
        self._scratch_flow_cap = torch.empty(self.num_edges, device=device, dtype=torch.float32)

    def reset(self):
        # Reset Logic
        self.in_queues.fill_(-1)
        self.out_queues.fill_(-1)
        self.in_cur.fill_(0)
        self.in_cnt.fill_(0)
        self.out_cur.fill_(0)
        self.out_cnt.fill_(0)
        
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

    @staticmethod
    def removeFront(counters, currents, sizes, link_ids):
        """
        Remove agents at head of queues of specified links
        """
        counters[link_ids] -= 1
        currents[link_ids] = (currents[link_ids] + 1) % sizes[link_ids].short() # (Eq. 5)

    @staticmethod
    def pushBack(counters, currents, sizes, offsets, queues, link_ids, agent_ids):
        """
        Push agents to tail of queues of active links
        """
        global_indices = offsets[link_ids] + ((currents[link_ids] + counters[link_ids]) % sizes[link_ids]) # (Eq. 4)
        queues[global_indices] = agent_ids
        counters[link_ids] += 1

    @staticmethod
    def pushBackAtomic(queues, offsets, currents, counts, sizes, link_ids, agent_ids, ranks):
        """
        Add agents to the tail of queues safely handling multiple agents entering the same link.
        Requires 'ranks' (0, 1, 2...) for agents entering the same link_id.
        link_ids MUST be sorted for the counter update to work correctly.
        """
        # 1. Calculate local index with arrival rank to avoid collisions
        write_indices = (currents[link_ids] + counts[link_ids] + ranks) % sizes[link_ids]
        
        # 2. Write to global index
        global_write_idx = offsets[link_ids] + write_indices
        queues[global_write_idx] = agent_ids
        
        # 3. Update counters — link_ids are sorted, so unique_consecutive is O(n_winners)
        unique_links, counts_to_add = torch.unique_consecutive(link_ids, return_counts=True)
        counts.index_add_(0, unique_links, counts_to_add.short())
    
    @staticmethod
    def getFrontAgent(currents, offsets, queues, link_ids):
        """
        Get agents at head of queues of active links
        """
        return queues[offsets[link_ids] + currents[link_ids]]

    @staticmethod
    def _compute_sorted_ranks(sorted_values, n, device):
        """
        Compute within-group ranks for a SORTED tensor.
        For sorted [A, A, B, B, B, C] -> returns [0, 1, 0, 1, 2, 0]
        O(n) — no cumsum over edges, works purely on the agent-sized tensors.
        """
        if n <= 1:
            return torch.zeros(n, device=device, dtype=torch.int32)
        # Detect group boundaries and compute group_id per element
        # group_id: [0,0,1,1,1,2] for [A,A,B,B,B,C]
        group_ids = torch.zeros(n, device=device, dtype=torch.int32)
        group_ids[1:] = (sorted_values[1:] != sorted_values[:-1]).int().cumsum_(0)
        # group_start[g] = first position of group g
        # boundary positions: 0, then positions where changes occur
        num_groups = group_ids[-1].item() + 1
        group_starts = torch.zeros(num_groups, device=device, dtype=torch.int32)
        if num_groups > 1:
            group_starts[1:] = (sorted_values[1:] != sorted_values[:-1]).nonzero(as_tuple=True)[0] + 1
        # rank = position - group_start[group_id]
        return torch.arange(n, device=device, dtype=torch.int32) - group_starts[group_ids]

    def _process_links(self):
        """
        Kernel 1: Links
        Logic to move appropriate agents from in_queue to out_queue
        (spatial buffer --> capacity buffer)
        Fused with capacity accumulation.
        """
        # Early exit: if no link has agents in its in_queue, skip entirely
        if not (self.in_cnt > 0).any():
            return

        # 0. Accumulate Capacity (fused)
        active_cap_mask = self.in_cnt > 0
        self.flow_accumulator[active_cap_mask] += self.flow_capacity_per_step[active_cap_mask]
        # Cap at max(1.0, flow_capacity) to prevent infinite banking
        caps = torch.maximum(torch.tensor(1.0, device=self.device), self.flow_capacity_per_step[active_cap_mask])
        torch.minimum(self.flow_accumulator[active_cap_mask], caps, out=caps)
        self.flow_accumulator[active_cap_mask] = caps

        while True:
            # 1. Get active links: in_queue not empty, out_queue not full, flow > 0
            mask = (self.in_cnt > 0) & (self.out_cnt < self.out_queue_sizes) & (self.flow_accumulator > 0)
            active_links = torch.nonzero(mask, as_tuple=True)[0]

            if active_links.numel() == 0:
                break

            # 2. Get agents at head of in_queues
            agent_ids = self.getFrontAgent(self.in_cur, self.in_queue_offsets, self.in_queues, active_links)

            # 3. Filter agents which can leave (travel time elapsed)
            ready_mask = (self.current_step - self.enter_link_time[agent_ids]) >= self.ff_travel_time_steps[active_links]

            if not ready_mask.any():
                break

            # 4. Apply mask
            moving_links = active_links[ready_mask]
            moving_agents = agent_ids[ready_mask]

            # 5. Execute move
            self.removeFront(self.in_cnt, self.in_cur, self.in_queue_sizes, moving_links)
            self.pushBack(self.out_cnt, self.out_cur, self.out_queue_sizes, self.out_queue_offsets, self.out_queues, moving_links, moving_agents)
            self.flow_accumulator[moving_links] -= 1.0

            # 6. Update Agent and Link Status
            self.status[moving_agents] = 2  # Status = Buffer
            self.stuck_since[moving_agents] = self.current_step
            self.last_link_exit_time[moving_links] = self.current_step

    def _process_nodes(self):
        """
        Kernel 2: Intersections
        Logic to move appropriate agents from out_queue of upstream links 
        to in_queue of downstream links
        (capacity buffer --> spatial buffer)
        
        Optimized: O(n_agents) ranking via sorted diff instead of O(n_edges) cumsum.
        """

        while True:
            # 1. Get all links with agents ready to leave
            active_links = torch.nonzero(self.out_cnt > 0, as_tuple=True)[0]

            if active_links.numel() == 0:
                break

            # 2. Get agents at head of out_queues and their next links
            agent_ids = self.getFrontAgent(self.out_cur, self.out_queue_offsets, self.out_queues, active_links)
            
            # Get next links for these agents
            curr_ptrs = self.path_ptr[agent_ids]
            next_ptrs = curr_ptrs + 1
            valid_idx_mask = next_ptrs < self.max_path_len
            next_links = torch.full_like(curr_ptrs, -1)
            if valid_idx_mask.any():
                valid_agents = agent_ids[valid_idx_mask]
                valid_ptrs = next_ptrs[valid_idx_mask]
                next_links[valid_idx_mask] = self.paths[valid_agents, valid_ptrs.long()]

            # 3. Handle Exits (agents reaching destination)
            exit_mask = next_links == -1

            if exit_mask.any():
                exiting_links = active_links[exit_mask]
                exiting_agents = agent_ids[exit_mask]

                self.removeFront(self.out_cnt, self.out_cur, self.out_queue_sizes, exiting_links)
                # scheduleNextActivity inlined
                self.status[exiting_agents] = 3
                self.wakeup_time[exiting_agents] = self.infinity
                self.agent_metrics[exiting_agents, 1] = (self.current_step - self.start_time[exiting_agents]).float()
                self.agent_metrics[exiting_agents, 0] += self.length[exiting_links]
            
            # 4. Handle Continuity / Transfers
            cont_mask = ~exit_mask

            if not cont_mask.any():
                continue

            cont_links = active_links[cont_mask]
            cont_agents = agent_ids[cont_mask]
            cont_targets = next_links[cont_mask]

            # Sort by target to group agents heading to the same link
            sort_idx = torch.argsort(cont_targets)
            s_targets = cont_targets[sort_idx]
            s_agents = cont_agents[sort_idx]
            s_links = cont_links[sort_idx]
            
            n = s_targets.shape[0]

            # O(n) ranking: compute within-group rank on sorted targets
            sorted_ranks = self._compute_sorted_ranks(s_targets, n, self.device)

            # Get available capacity (logical, without squeeze margin)
            available_capacity = self.in_queue_sizes[s_targets] - self.in_cnt[s_targets] - self.squeeze_margin

            winners_mask = sorted_ranks < available_capacity

            # 4.1 Handle stuck agents (gridlock resolution)
            stuck_mask = ~winners_mask

            if stuck_mask.any():
                stuck_agents = s_agents[stuck_mask]
                stuck_ranks = sorted_ranks[stuck_mask]
                stuck_avails = available_capacity[stuck_mask]

                # Is the agent stuck long enough?
                stuck_dur = self.current_step - self.stuck_since[stuck_agents]
                is_stuck_long = stuck_dur > self.stuck_threshold

                # Is there space in the physical buffer (logical + squeeze margin)?
                item_avails_phy = stuck_avails + self.squeeze_margin
                has_phy_space = stuck_ranks < item_avails_phy

                gridlock_mask = is_stuck_long & has_phy_space
                winners_mask[stuck_mask] = gridlock_mask

            # 5. Move "winning" agents
            if not winners_mask.any():
                break

            win_from = s_links[winners_mask]
            win_to = s_targets[winners_mask]
            win_agents_id = s_agents[winners_mask]
            win_ranks = sorted_ranks[winners_mask]

            # 6. Update queues (win_to is sorted since s_targets was sorted and winners is a submask)
            self.removeFront(self.out_cnt, self.out_cur, self.out_queue_sizes, win_from)
            self.pushBackAtomic(self.in_queues, self.in_queue_offsets, self.in_cur, self.in_cnt, self.in_queue_sizes, win_to, win_agents_id, win_ranks)
            
            # scheduleNextLink inlined
            self.status[win_agents_id] = 1  # Traveling
            self.current_edge[win_agents_id] = win_to
            self.path_ptr[win_agents_id] += 1
            self.enter_link_time[win_agents_id] = self.current_step
            self.agent_metrics[win_agents_id, 0] += self.length[win_from]
                
    def _schedule_demand(self):
        """Schedule departures for waiting agents."""
        active_mask = (self.status == 0) & (self.departure_times <= self.current_step)
        if not active_mask.any():
            return

        active_agents = torch.nonzero(active_mask, as_tuple=True)[0]
        
        # Target Edges
        first_edges = self.paths[active_agents, 0]
        
        # Sort by target edge to group agents
        sort_idx = torch.argsort(first_edges)
        s_agents = active_agents[sort_idx]
        s_edges = first_edges[sort_idx]
        n = s_agents.shape[0]
        
        # O(n) ranking on sorted edges
        sorted_ranks = self._compute_sorted_ranks(s_edges, n, self.device)
        
        # Available capacity per target
        avail = self.in_queue_sizes[s_edges] - self.in_cnt[s_edges]
        
        passed = sorted_ranks < avail
        
        if passed.any():
            w_agents = s_agents[passed]
            w_edges = s_edges[passed]
            w_ranks = sorted_ranks[passed]
            
            # Insert to In-Queue
            t_heads = self.in_cur[w_edges]
            t_counts = self.in_cnt[w_edges]
            t_sizes = self.in_queue_sizes[w_edges]
            t_offsets = self.in_queue_offsets[w_edges]
            
            write_curr = (t_heads + t_counts + w_ranks) % t_sizes
            global_write = t_offsets + write_curr
            
            self.in_queues[global_write] = w_agents.int()
            
            # Update Counts — w_edges is sorted
            unique_w, counts_w = torch.unique_consecutive(w_edges, return_counts=True)
            self.in_cnt.index_add_(0, unique_w, counts_w.short())
            
            # Update Agent
            self.status[w_agents] = 1  # Traveling
            self.current_edge[w_agents] = w_edges
            self.enter_link_time[w_agents] = self.current_step

    def step(self):
        # Step Order: ProcessLinks -> ProcessNodes -> ScheduleDemand
        # Then increment time
        
        # Process Links (Internal move + Accumulate Capacity)
        self._process_links()
        
        # Process Nodes (External move + Gridlock)
        self._process_nodes()
        
        # Schedule Demand (Departures)
        self._schedule_demand()

        self.current_step += 1

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
