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
        # We need to process nodes, so we need max_nodes or similar
        # net_topology = (upstream_indices, upstream_offsets)
        if net_topology is None:
             raise ValueError("net_topology is required for TorchDNLGEMSim")
        
        # We don't seem to use this even. 
        self.upstream_indices = net_topology[0].to(device).long()
        self.upstream_offsets = net_topology[1].to(device).long()
        self.num_nodes = self.upstream_offsets.shape[0] - 1
        
        # --- 1. Structure of Arrays (SoA) & Ring Buffers ---
        
        # -- 1.1 Buffer Descriptors --
        # Descriptors : Global Pointers (queue_offset, queue_size) | Tells us where each link's queue starts and ends in the global ring buffer.
        # queue_offset : start index of the link's queue in the global ring buffer.
        # queue_size : size of the link's queue.

        # Calculate queue_size --------------------
        # TODO: Remove calculation from benchmark_gemsim_dnl.py and calculate here instead.
        # Spatial Buffer (N_l)
        # N_l = max(floor(L_link / L_veh) * lanes, 1)
        # Actually storage_capacity is already calculated in benchmark_gemsim_dnl.py.
        # So we can just use storage_capacity as size. 
        self.in_queue_sizes = torch.clamp(self.storage_capacity.int(), min=1)
        
        # Capacity Buffer (N_f)
        # N_f = max(floor(q * dt / period), 1)
        # q is outflow_capacity per period (ex. veh/hour), dt is timeStepSize (sec), period is in seconds
        # Actually capacity_buffer is already calculated in benchmark_gemsim_dnl.py.  
        self.out_queue_sizes = torch.clamp(self.flow_capacity_per_step.int(), min=1)

        # Add squeeze_margin to the buffer sizes for Gridlock "Squeeze-in"
        # TODO : Find out what GEMSim does exactly. Or find better dynamic way to set squeeze_margin size.
        self.in_queue_sizes += self.squeeze_margin
        self.out_queue_sizes += self.squeeze_margin

        # Calculate queue_offset --------------------
        self.in_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.int32)
        self.in_queue_offsets[1:] = torch.cumsum(self.in_queue_sizes, dim=0)
        self.total_in_queue_size = self.in_queue_offsets[-1].item()
        
        self.out_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.int32)
        self.out_queue_offsets[1:] = torch.cumsum(self.out_queue_sizes, dim=0)
        self.total_out_queue_size = self.out_queue_offsets[-1].item()
        
        # -- 1.2 Dynamic State Buffer Descriptors --
        # States : Local Pointers (q_cur, q_cnt) | Tells us the dynamic state of each link's queue in their local ring buffer.
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
        # TODO: GEMSim has an activity-based model and also embedded AoS approach. 
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
        # TODO: Verify the -1000 initialization is okay.
        self.last_link_exit_time = torch.zeros(self.num_edges, device=device, dtype=torch.int32) - 1000 # Init with past

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
        """
        # 1. Calculate local index with arrival rank to avoid collisions
        write_indices = (currents[link_ids] + counts[link_ids] + ranks) % sizes[link_ids]
        
        # 2. Write to global index
        global_write_idx = offsets[link_ids] + write_indices
        queues[global_write_idx] = agent_ids
        
        # 3. Update counters securely (agglomerated operation)
        # Group by link and add the number of agents entering
        unique_links, counts_to_add = torch.unique_consecutive(link_ids, return_counts=True)
        counts.index_add_(0, unique_links, counts_to_add.short())
    
    @staticmethod
    def getFrontAgent(currents, offsets, queues, link_ids):
        """
        Get agents at head of queues of active links
        """
        return queues[offsets[link_ids] + currents[link_ids]]

    def _accumulate_capacity(self):
        # Accumulate only if occupied
        # allowing negative accumulator.
        
        active_mask = self.in_cnt > 0
        if active_mask.any():
            self.flow_accumulator[active_mask] += self.flow_capacity_per_step[active_mask]
            
            # Cap at max(1.0, flow_capacity) to prevent infinite banking
            caps = torch.maximum(torch.tensor(1.0, device=self.device), self.flow_capacity_per_step[active_mask])
            self.flow_accumulator[active_mask] = torch.minimum(self.flow_accumulator[active_mask], caps)

    def _process_links(self):
        """
        Kernel 1: Links
        Logic to move appropriate agents from in_queue to out_queue
        (spatial buffer --> capacity buffer)
        """
            
        def updateStatus(self, link_ids, agent_ids):
            """
            Update active agent status and active link exit time 
            """
            self.status[agent_ids] = 2 # Status = Buffer
            self.stuck_since[agent_ids] = self.current_step
            self.last_link_exit_time[link_ids] = self.current_step

        # 0. Accumulate Capacity
        self._accumulate_capacity()
        
        # 1. Get initial active links
        mask = (self.in_cnt > 0) & (self.out_cnt < self.out_queue_sizes) & (self.flow_accumulator > 0)
        active_links = torch.nonzero(mask, as_tuple=True)[0]

        while active_links.numel() > 0:
            # 2. Get agents at head of in_queues of active links 
            agent_ids = self.getFrontAgent(self.in_cur, self.in_queue_offsets, self.in_queues, active_links)

            # 3. Filter agents which can leave
            ready_mask = (self.current_step - self.enter_link_time[agent_ids]) >= self.ff_travel_time_steps[active_links]

            if not ready_mask.any():
                break # No agents to process

            # 4. Apply mask
            moving_links = active_links[ready_mask]
            moving_agents = agent_ids[ready_mask]

            # 5. Execute move
            self.removeFront(self.in_cnt, self.in_cur, self.in_queue_sizes, moving_links)
            self.pushBack(self.out_cnt, self.out_cur, self.out_queue_sizes, self.out_queue_offsets, self.out_queues, moving_links, moving_agents)
            self.flow_accumulator[moving_links] -= 1.0

            # 6. Update Agent and Link Status
            updateStatus(self, moving_links, moving_agents)
            
            # 7. Check conditions again ONLY for moving_links to continue looping
            still_active_mask = (self.in_cnt[moving_links] > 0) & \
                                (self.out_cnt[moving_links] < self.out_queue_sizes[moving_links]) & \
                                (self.flow_accumulator[moving_links] > 0)
            
            active_links = moving_links[still_active_mask]

    def _process_nodes(self):
        """
        Kernel 2: Intersections
        Logic to move appropriate agents from out_queue of upstream links 
        to in_queue of downstream links
        (capacity buffer --> spatial buffer)
        """

        def getNextLinks(agent_ids):
            # Agent -> Next Edge
            curr_ptrs = self.path_ptr[agent_ids]
            next_ptrs = curr_ptrs + 1
            
            valid_idx_mask = next_ptrs < self.max_path_len
            next_links = torch.full_like(curr_ptrs, -1)
            
            if valid_idx_mask.any():
                valid_agents = agent_ids[valid_idx_mask]
                valid_ptrs = next_ptrs[valid_idx_mask]
                next_links[valid_idx_mask] = self.paths[valid_agents, valid_ptrs.long()]
            return next_links

        def scheduleNextActivity(agents, exiting_links):
            """Finalize agent metrics and status upon reaching destination."""
            self.status[agents] = 3
            self.wakeup_time[agents] = self.infinity
            self.agent_metrics[agents, 1] = (self.current_step - self.start_time[agents]).float()
            self.agent_metrics[agents, 0] += self.length[exiting_links]

        def calculateArrivalRank(sorted_targets):
            # TODO: Make ArrivalRank random with probability based on link outflow
            unique_targets, inverse_indices, counts = torch.unique_consecutive(sorted_targets, return_inverse=True, return_counts=True)
            group_starts = torch.zeros_like(unique_targets, dtype=torch.int)
            if unique_targets.numel() > 0:
                group_starts[1:] = torch.cumsum(counts[:-1], dim=0)

            repeat_group_starts = group_starts[inverse_indices]
            
            # Calculate arrival rank
            indices = torch.arange(sorted_targets.shape[0], device=self.device)
            ranks = indices - repeat_group_starts
            return ranks, unique_targets, inverse_indices, counts

        def getAvailableCapacityLogical(self, targets, counts, inverse_indices):
            """
            Get available capacity for each link spatial buffer and spread it to match agent shape
            This doesn't include the squeeze_margin.
            """
            avail_space = self.in_queue_sizes[targets] - self.in_cnt[targets] - self.squeeze_margin
            return avail_space[inverse_indices]

        def resolveGridlock(stuck_agents, stuck_targets, stuck_ranks, stuck_item_avails):
            """
            Resolve agents that are stuck for too long if there's available space in the physical queue.
            Physical queue = Logicial queue + Security Buffer
            """
            
            # Is the agent stuck ?
            stuck_dur = self.current_step - self.stuck_since[stuck_agents]
            is_stuck_long = stuck_dur > self.stuck_threshold

            # Is there still space available in the physical buffer of the target link, including the squeeze margin ?
            item_avails_phy = stuck_item_avails + self.squeeze_margin
            has_phy_space = stuck_ranks < item_avails_phy

            return is_stuck_long & has_phy_space

        def scheduleNextLink(self, agents, from_links, to_links):
            """
            Handle agents entering a new link.
            """
            
            # Status = 1 (Traveling)
            self.status[agents] = 1

            # Update position on the graph
            self.current_edge[agents] = to_links
            
            # Update next link based on path
            self.path_ptr[agents] += 1

            # Update clock
            self.enter_link_time[agents] = self.current_step

            # Update agent metrics
            self.agent_metrics[agents, 0] += self.length[from_links]

        # 1. Get initial active links
        active_links = torch.nonzero(self.out_cnt > 0, as_tuple=True)[0]
        
        while active_links.numel() > 0:
            # 2. Get agents at head of out_queues of active links and their next links
            agent_ids = self.getFrontAgent(self.out_cur, self.out_queue_offsets, self.out_queues, active_links)
            next_links = getNextLinks(agent_ids)

            # 3. Handle Exits
            exit_mask = next_links == -1
            
            exiting_links = torch.empty(0, dtype=torch.long, device=self.device)

            if exit_mask.any():
                exiting_links = active_links[exit_mask]
                exiting_agents = agent_ids[exit_mask]

                self.removeFront(self.out_cnt, self.out_cur, self.out_queue_sizes, exiting_links)
                scheduleNextActivity(exiting_agents, exiting_links)
            
            # 4. Handle Continuity / Transfers
            cont_mask = ~exit_mask

            win_from = torch.empty(0, dtype=torch.long, device=self.device)
            if cont_mask.any():
                cont_links = active_links[cont_mask]
                cont_agents = agent_ids[cont_mask]
                cont_targets = next_links[cont_mask]

                # We MUST sort everything by cont_targets to group agents entering the same link
                sort_idx = torch.argsort(cont_targets)
                s_targets = cont_targets[sort_idx]
                s_agents = cont_agents[sort_idx]
                s_links = cont_links[sort_idx]

                ranks, unique_targets, inverse_indices, counts = calculateArrivalRank(s_targets)
                available_capacity = getAvailableCapacityLogical(self, unique_targets, counts, inverse_indices)

                winners_mask = ranks < available_capacity

                # 4.1 Handle stuck agents
                stuck_mask = ~winners_mask

                if stuck_mask.any():
                    stuck_agents = cont_agents[stuck_mask]
                    stuck_targets = cont_targets[stuck_mask]

                    # Find out who can force their way into the spatial buffer
                    gridlock_mask = resolveGridlock(self, stuck_agents, stuck_targets, ranks[stuck_mask], available_capacity[stuck_mask])
                    winners_mask[stuck_mask] = gridlock_mask

                # 5. Move "winning" agents
                if winners_mask.any():
                    win_from = s_links[winners_mask]
                    win_to = s_targets[winners_mask]
                    win_agents_id = s_agents[winners_mask]
                    win_ranks = ranks[winners_mask]

                    # 6. Update queues
                    self.removeFront(self.out_cnt, self.out_cur, self.out_queue_sizes, win_from)
                    self.pushBackAtomic(self.in_queues, self.in_queue_offsets, self.in_cur, self.in_cnt, self.in_queue_sizes, win_to, win_agents_id, win_ranks)
                    scheduleNextLink(self, win_agents_id, win_from, win_to)
            
            # 7. Update active links for next iteration
            if exiting_links.numel() == 0 and win_from.numel() == 0:
                break
                
            moved_links = torch.cat([exiting_links, win_from])
            still_active = self.out_cnt[moved_links] > 0
            active_links = moved_links[still_active]
                
    def _schedule_demand(self):
        # Previously _inject_departures
        # Check active agents
        active_mask = (self.wakeup_time <= self.current_step)
        if not active_mask.any():
            return

        active_agents = torch.nonzero(active_mask, as_tuple=True)[0]
        
        # Target Edges
        first_edges = self.paths[active_agents, 0]
        
        # Check Capacity of First Edges
        sort_idx = torch.argsort(first_edges)
        s_agents = active_agents[sort_idx]
        s_edges = first_edges[sort_idx]
        
        unique_edges, inverse_indices, counts = torch.unique_consecutive(s_edges, return_inverse=True, return_counts=True)
        
        # Cap check
        current_in = self.in_cnt[unique_edges]
        
        max_sizes = self.in_queue_sizes[unique_edges]
        avail = max_sizes - current_in
        
        group_starts = torch.zeros_like(unique_edges)
        if unique_edges.numel() > 0:
            group_starts[1:] = torch.cumsum(counts[:-1], dim=0)
        
        indices = torch.arange(s_agents.shape[0], device=self.device)
        item_group_starts = group_starts[inverse_indices]
        ranks = indices - item_group_starts
        
        item_avails = avail[inverse_indices]
        
        passed = ranks < item_avails
        
        if passed.any():
            w_agents = s_agents[passed]
            w_edges = s_edges[passed]
            w_ranks = ranks[passed]
            
            # Insert to In-Queue
            t_heads = self.in_cur[w_edges]
            t_counts = self.in_cnt[w_edges]
            t_sizes = self.in_queue_sizes[w_edges] # Physical size
            t_offsets = self.in_queue_offsets[w_edges]
            
            write_curr = (t_heads + t_counts + w_ranks) % t_sizes
            global_write = t_offsets + write_curr
            
            self.in_queues[global_write] = w_agents.int()
            
            # Update Counts
            unique_w, counts_w = torch.unique_consecutive(w_edges, return_counts=True)
            self.in_cnt.index_add_(0, unique_w, counts_w.short())
            
            # Update Agent
            self.status[w_agents] = 1 # Traveling
            self.current_edge[w_agents] = w_edges
            self.enter_link_time[w_agents] = self.current_step
            self.wakeup_time[w_agents] = self.infinity # Agent departed! Skip checking until done.
            # path_ptr remains 0

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
