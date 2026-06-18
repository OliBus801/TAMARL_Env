
import torch
import cProfile
import pstats
import io
import math
import psutil
import os

# Event type constants
EVT_ACTEND = 0
EVT_DEPARTURE = 1
EVT_ENTERS_TRAFFIC = 2
EVT_LEFT_LINK = 3
EVT_ENTERED_LINK = 4
EVT_LEAVES_TRAFFIC = 5
EVT_ARRIVAL = 6
EVT_ACTSTART = 7
EVT_STUCKANDABORT = 8
EVT_ENTERED_BUFFER = 9

EVENT_TYPE_NAMES = {
    EVT_ACTEND: 'actend',
    EVT_DEPARTURE: 'departure',
    EVT_ENTERS_TRAFFIC: 'enters_traffic',
    EVT_LEFT_LINK: 'left_link',
    EVT_ENTERED_LINK: 'entered_link',
    EVT_LEAVES_TRAFFIC: 'leaves_traffic',
    EVT_ARRIVAL: 'arrival',
    EVT_ACTSTART: 'actstart',
    EVT_STUCKANDABORT: 'stuckAndAbort',
    EVT_ENTERED_BUFFER: 'entered_buffer',
}

class TorchDNLMATSim:
    def __init__(self, 
                 edge_static: torch.Tensor, 
                 paths: torch.Tensor = None,
                 device: str = 'cuda',
                 departure_times: torch.Tensor = None,
                 edge_endpoints: torch.Tensor = None,
                 act_end_times: torch.Tensor = None,
                 act_durations: torch.Tensor = None,
                 num_legs: torch.Tensor = None,
                 stuck_threshold: int = 10,
                 dt: float = 1.0,
                 seed: int = None,
                 enable_profiling: bool = False,
                 track_events: bool = False,
                 first_edges: torch.Tensor = None,
                 destinations: torch.Tensor = None,
                 collect_link_tt: bool = False,
                 link_tt_interval: float = 300.0,
                 paths_flat: torch.Tensor = None,
                 path_offsets: torch.Tensor = None,
                 max_steps: int = None):
        """
        Initialize the TorchDNLMATSim simulation engine.
        
        Args:
            edge_static: Tensor [E, 5] -> [length, free_flow_speed, capacity_storage (c_e), capacity_flow (D_e per hour), ff_travel_time]
            paths: DEPRECATED. Dense tensor [A, MaxPathLen]. Use paths_flat + path_offsets instead.
            device: Device to run on ('cuda' or 'cpu').
            departure_times: Tensor [A] -> Departure times for each agent.
            edge_endpoints: Tensor [E, 2] -> [from_node, to_node] for connectivity validation.
            act_end_times: Tensor [A, MaxActs] -> Absolute end times for intermediate activities.
            act_durations: Tensor [A, MaxActs] -> Durations for intermediate activities.
            num_legs: Tensor [A] -> Total number of legs per agent.
            stuck_threshold: Time in steps an agent waits in buffer before forcing entry.
            dt: Simulation time step in seconds.
            enable_profiling: If True, enables cProfile.
            track_events: If True, records MATSim-style simulation events.
            first_edges: Tensor [A, MaxLegs] -> First edge for each leg (RL mode only).
            destinations: Tensor [A, MaxLegs] -> Destination node for each leg (RL mode only).
            paths_flat: Tensor [TotalEdges] -> 1D compact edge indices for all agents concatenated.
            path_offsets: Tensor [A+1] -> CSR-style offsets into paths_flat per agent.
            max_steps: If provided, pre-allocates link TT buffers for exactly this many steps,
                       avoiding repeated doubling during the simulation.
        """
        self.device = device
        self.stuck_threshold = stuck_threshold
        self.dt = dt
        self._max_steps = max_steps  # stored for link TT buffer sizing
        
        # Random number generator for probabilistic upstream link selection
        # Using a CPU generator, but need to monitor performance impact. -OB 
        self.rng = torch.Generator(device='cpu')
        self.seed = seed
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            self.rng.seed()
        
        # Profiling
        self.profiler = None
        if enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # Event tracking -- uses a pre-allocated GPU tensor buffer instead of Python list
        self.track_events = track_events

        # Test number of interactions
        self.interactions = 0
        self.stuck_count = 0

        # Edge connectivity (needed for path validation)
        if edge_endpoints is None:
            raise ValueError("edge_endpoints is mandatory for path validation.")
        self.edge_endpoints = edge_endpoints.to(device).int()

        # Move constants to device
        self.edge_static = edge_static.to(device)
        
        if paths is None and paths_flat is None:
            raise ValueError("Either 'paths' or 'paths_flat' + 'path_offsets' must be provided.")
            
        # Sparse CSR format (preferred): paths_flat + path_offsets
        if paths_flat is not None:
            self.paths_flat = paths_flat.to(device, dtype=torch.int32)
            self.path_offsets = path_offsets.to(device, dtype=torch.long)
            self.num_agents = self.path_offsets.shape[0] - 1
        else:
            # Legacy dense format: convert to CSR on the fly
            paths_dev = paths.to(device, dtype=torch.int32)
            self.num_agents = paths_dev.shape[0]
            flat_list = []
            offsets = [0]
            for i in range(self.num_agents):
                row = paths_dev[i]
                # Find last valid (non -1) element, keeping -2 separators
                valid_mask = row != -1
                if valid_mask.any():
                    last_valid = valid_mask.nonzero(as_tuple=True)[0][-1].item()
                    flat_list.append(row[:last_valid + 1])
                    offsets.append(offsets[-1] + last_valid + 1)
                else:
                    offsets.append(offsets[-1])
            if flat_list:
                self.paths_flat = torch.cat(flat_list).to(device, dtype=torch.int32)
            else:
                self.paths_flat = torch.empty(0, device=device, dtype=torch.int32)
            self.path_offsets = torch.tensor(offsets, device=device, dtype=torch.long)
            del paths_dev
            
        # Initialize event buffer now that self.num_agents is defined
        self._init_event_buffer()
            
        # Multi-leg definitions
        if num_legs is not None:
            self.num_legs = num_legs.to(device).long()
        else:
            self.num_legs = torch.ones(self.num_agents, device=device, dtype=torch.long)
            
        self.max_legs_count = self.num_legs.max().item() if self.num_agents > 0 else 1
        
        if act_end_times is not None:
            self.act_end_times = act_end_times.to(device).int()
        else:
            self.act_end_times = torch.full((self.num_agents, 0), -1, device=device, dtype=torch.int32)
            
        if act_durations is not None:
            self.act_durations = act_durations.to(device).int()
        else:
            self.act_durations = torch.full((self.num_agents, 0), -1, device=device, dtype=torch.int32)
            
        self.current_leg = torch.zeros(self.num_agents, device=device, dtype=torch.long)
        
        self.num_edges = self.edge_static.shape[0]

        # Edge Attributes
        self.length = self.edge_static[:, 0].contiguous()
        self.free_speed = self.edge_static[:, 1].contiguous()
        self.storage_capacity = self.edge_static[:, 2].contiguous()
        
        # Flow capacity is usually veh/hour. Convert to veh/step.
        # We set the flow_capacity to veh/timestep already
        self.flow_capacity_per_step = self.edge_static[:, 3]
        
        # Free flow travel time (steps)
        self.ff_travel_time_steps = torch.floor(self.edge_static[:, 4] / self.dt).long().contiguous()

        # Edge Dynamic State [E]
        self.edge_occupancy = torch.zeros(self.num_edges, device=self.device, dtype=torch.int32)
        
        # Flow Capacity Accumulator (MATSim lazy on-demand model)
        self.edge_capacity_accumulator = self.flow_capacity_per_step.clone().to(device=self.device, dtype=torch.float32)
        self.step_edge_limits = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        # OPT: Incremental buffer counts (avoids recomputing bincount on all agents)
        self.buffer_counts = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        self.collect_link_tt = collect_link_tt
        self.link_tt_interval = link_tt_interval
        if self.collect_link_tt:
            # Pre-allocate the exact number of intervals needed to cover the simulation,
            # avoiding the expensive doubling strategy during the run.
            # Formula: ceil(max_steps * dt / link_tt_interval) + 1 (safety margin)
            # Falls back to 288 (covers a full 24h simulation at 300s intervals) if unknown.
            if max_steps is not None and link_tt_interval > 0:
                import math
                num_intervals = math.ceil(max_steps * dt / link_tt_interval) + 1
            else:
                num_intervals = 288  # default: 86400s / 300s
            # Store full history on CPU using pinned memory for fast asynchronous transfers if CUDA is available
            pin_memory = torch.cuda.is_available()
            self.interval_tt_sum = torch.zeros((num_intervals, self.num_edges), device='cpu', dtype=torch.float32, pin_memory=pin_memory)
            self.interval_tt_count = torch.zeros((num_intervals, self.num_edges), device='cpu', dtype=torch.float32, pin_memory=pin_memory)
            # Double buffer current active interval on simulator device (e.g. GPU)
            self.current_interval_idx = 0
            self.current_interval_tt_sum = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)
            self.current_interval_tt_count = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        # Agent State - Structure of Arrays (SOA)
        # Status | 0: Waiting/Activity, 1: Traveling, 2: Buffer, 3: Done, 4: Exiter (waiting for dispatch)
        self.status = torch.zeros(self.num_agents, device=self.device, dtype=torch.uint8)
        # OPT: Use int64 (long) for edge/ptr indices to avoid repeated .long() conversions
        self.current_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.long)
        self.next_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.long)
        self.path_ptr = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
        self.arrival_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.stuck_since = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.start_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        
        # Initialize Status
        if departure_times is not None:
             self.departure_times = departure_times.to(self.device).int()
        else:
             self.departure_times = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
             
        self.start_time.copy_(self.departure_times)
        
        # Pre-set first edge as next_edge for waiting agents
        self.next_edge[:] = self.paths_flat[self.path_offsets[:-1]].long()
        
        # Leg Metrics [A, MaxLegs, 2] -> [accumulated_distance, final_travel_time]
        self.leg_metrics = torch.zeros((self.num_agents, self.max_legs_count, 2), device=self.device, dtype=torch.float32)
        # Leg departure times [A, MaxLegs] -> Actual step when agent departed for that leg
        self.leg_departure_times = torch.full((self.num_agents, self.max_legs_count), -1, device=self.device, dtype=torch.int32)

        # Optimization: Wakeup Time [A]
        # Unified scheduler: agents are only processed if current_step >= wakeup_time
        self.wakeup_time = self.departure_times.clone()
        self.infinity = 2**30

        # Track whether actend/departure events have been emitted (for track_events)
        self._departure_emitted = torch.zeros(self.num_agents, device=self.device, dtype=torch.bool)

        # OPT: Pre-allocate reusable scratch tensors
        self._edge_priority_scratch = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        self.current_step = 0
        self.active_agents_count = self.num_agents
        self.n_imputed_legs = 0

    def _init_event_buffer(self):
        """Initialize event buffer as a pre-allocated tensor (or None if not tracking)."""
        if self.track_events:
            # Scale the GPU chunk size based on population: ~20 events per agent per chunk
            # Bound between 500k and 5M to avoid massive initial allocations or tiny chunks
            estimated = max(500_000, self.num_agents * 20)
            self._event_chunk_size = min(estimated, 5_000_000)
            self._event_buffer = torch.empty((self._event_chunk_size, 4), device=self.device, dtype=torch.int32)
            self._event_count = 0
            self._cpu_events_blocks = []
        else:
            self._event_buffer = None
            self._event_count = 0
            self._cpu_events_blocks = []



    def reset(self):
        self.edge_occupancy.fill_(0)
        self.edge_capacity_accumulator.copy_(self.flow_capacity_per_step)
        self.step_edge_limits.fill_(0)
        self.buffer_counts.fill_(0)
        
        self.status.fill_(0)
        self.current_edge.fill_(-1)
        self.current_leg.fill_(0)
        self.next_edge[:] = self.paths_flat[self.path_offsets[:-1]].long()
            
        self.path_ptr.fill_(0)
        self.arrival_time.fill_(0)
        self.stuck_since.fill_(0)
        self.start_time.copy_(self.departure_times)
        
        self.leg_metrics.fill_(0)
        self.leg_departure_times.fill_(-1)
        self.wakeup_time.copy_(self.departure_times)
        self._departure_emitted.fill_(False)
        self.current_step = 0
        self.active_agents_count = self.num_agents
        self.stuck_count = 0
        self.n_imputed_legs = 0
        if self.seed is not None:
            self.rng.manual_seed(self.seed)
        self._init_event_buffer()
        
        if self.collect_link_tt:
            self.interval_tt_sum.fill_(0)
            self.interval_tt_count.fill_(0)
            self.current_interval_idx = 0
            self.current_interval_tt_sum.zero_()
            self.current_interval_tt_count.zero_()

        if self.profiler:
            self.profiler.clear()

    def _flush_events(self):
        """Move the current GPU event buffer to CPU RAM and reset counter."""
        if self._event_count > 0:
            self._cpu_events_blocks.append(self._event_buffer[:self._event_count].cpu().clone())
            self._event_count = 0

    def _record_events(self, event_type, agent_ids, edge_ids):
        """Record events using a pre-allocated tensor buffer. Flushes to CPU locally."""
        if not self.track_events:
            return
        n = agent_ids.size(0)
        if n == 0:
            return
            
        needed = self._event_count + n
        if needed > self._event_buffer.size(0):
            if n > self._event_buffer.size(0):
                # Exceptional burst: grow the GPU buffer
                new_cap = max(self._event_buffer.size(0) * 2, needed)
                new_buf = torch.empty((new_cap, 4), device=self.device, dtype=torch.int32)
                if self._event_count > 0:
                    new_buf[:self._event_count] = self._event_buffer[:self._event_count]
                self._event_buffer = new_buf
            else:
                self._flush_events()
                needed = self._event_count + n
        
        buf = self._event_buffer
        s = self._event_count
        buf[s:needed, 0] = self.current_step
        buf[s:needed, 1] = event_type
        buf[s:needed, 2] = agent_ids
        buf[s:needed, 3] = edge_ids
        self._event_count = needed

    # =========================================================================
    # Flow Accumulation (MATSim: updateFastFlowAccumulation)
    # =========================================================================
    def _update_all_flow_accumulation(self):
        """
        Global flow accumulation for ALL edges, once per step.
        """
        remaining = self.flow_capacity_per_step - self.buffer_counts
        
        self.edge_capacity_accumulator = torch.minimum(
            self.edge_capacity_accumulator + self.flow_capacity_per_step, 
            remaining
        )

    # =========================================================================
    # A. Process Nodes (Capacity Buffer -> Downstream Spatial Buffer)
    # =========================================================================
    def _process_nodes_A(self):
        """
        Move agents from upstream capacity buffer (status=2) to downstream 
        spatial buffer (status=1).
        
        Constraints for capacity -> downstream:
        - There is enough free-space (storage capacity) on downstream link
        - If waiting_time > gridlock_threshold, force entry regardless of space
        
        Events emitted: left_link, entered_link
        """
        buffer_mask = (self.status == 2) & (self.wakeup_time <= self.current_step)
        buffer_agents = torch.nonzero(buffer_mask, as_tuple=True)[0]
        if buffer_agents.numel() == 0:
            return

        b_next = self.next_edge[buffer_agents]

        # Process movers (next_edge != -1)
        move_mask = (b_next != -1)
        movers = buffer_agents[move_mask]
        if movers.numel() == 0:
            return

        m_next = b_next[move_mask]
        m_curr = self.current_edge[movers]

        # Dynamic connectivity check: to_node[curr] must equal from_node[next]
        to_of_curr = self.edge_endpoints[m_curr, 1]
        from_of_next = self.edge_endpoints[m_next, 0]
        connected = (to_of_curr == from_of_next)
        disconnected = ~connected

        if disconnected.any():
            stuck_agents = movers[disconnected]
            num_stuck = stuck_agents.numel()
            self.stuck_count += num_stuck

            # stuckAndAbort: remove from network
            self.status[stuck_agents] = 3
            self.active_agents_count -= num_stuck
            self.wakeup_time[stuck_agents] = self.infinity
            c_legs_stuck = self.current_leg[stuck_agents]
            dep_times = self.leg_departure_times[stuck_agents, c_legs_stuck]
            self.leg_metrics[stuck_agents, c_legs_stuck, 1] = (self.current_step - dep_times).float()
            # OPT: Update buffer_counts for agents leaving buffer
            self.buffer_counts.scatter_add_(0, m_curr[disconnected],
                -torch.ones(num_stuck, device=self.device))

            if self.track_events:
                self._record_events(EVT_STUCKANDABORT, stuck_agents, m_curr[disconnected])

            # Keep only connected movers
            movers = movers[connected]
            m_next = m_next[connected]
            m_curr = m_curr[connected]

        if movers.numel() == 0:
            return

        m_stuck = self.stuck_since[movers]

        # MATSim: emptyBufferAfterBufferRandomDistribution
        # Upstream links feeding the same node are randomly prioritized (weighted by capacity)
        # Then all agents from the selected upstream link are processed before the next.
        # 
        # Vectorized approach: assign a random priority per unique upstream link,
        # such that links with higher capacity have higher priority on average.
        # Sort movers by (downstream_link, upstream_random_priority) so the
        # randomly-selected link's agents come first in the storage capacity check.
        
        # Random priority per unique upstream link: rand() * capacity (weighted random)
        unique_curr = torch.unique(m_curr)
        rand_vals = torch.rand(unique_curr.size(0), generator=self.rng).to(self.device)
        weighted_rand = rand_vals * self.flow_capacity_per_step[unique_curr]
        
        # OPT: Reuse pre-allocated scratch tensor instead of allocating new one
        edge_priority = self._edge_priority_scratch
        edge_priority.zero_()
        edge_priority[unique_curr] = weighted_rand
        mover_priority = edge_priority[m_curr]

        # Sort by (downstream_link, -upstream_priority, stuck_time) 
        # Higher priority links come first (negate for ascending sort)
        # Within same upstream link, process agents in order (by agent id for stability)
        sort_keys = (m_next * 10000000000
                     - (mover_priority * 1000000).long() * 10000
                     + torch.arange(movers.size(0), device=self.device))
        sort_idx = torch.argsort(sort_keys, stable=True)

        movers_s = movers[sort_idx]
        next_s = m_next[sort_idx]
        curr_s = m_curr[sort_idx]
        stuck_s = m_stuck[sort_idx]

        # Storage capacity check
        # OPT: cummax-based group ranks (replaces unique_consecutive + repeat_interleave)
        n_movers = movers_s.size(0)
        positions = torch.arange(n_movers, device=self.device)
        boundary_pos = torch.zeros(n_movers, device=self.device, dtype=torch.long)
        if n_movers > 1:
            boundary_pos[1:] = (next_s[1:] != next_s[:-1]).long() * positions[1:]
        group_starts_v, _ = torch.cummax(boundary_pos, dim=0)
        inflow_ranks = positions - group_starts_v

        caps = self.storage_capacity[next_s]
        occs = self.edge_occupancy[next_s]
        avail = caps - occs

        is_stuck = (self.current_step - stuck_s) > self.stuck_threshold
        storage_pass = is_stuck | (inflow_ranks < avail)

        winners = movers_s[storage_pass]

        if winners.numel() > 0:
            w_curr = curr_s[storage_pass]
            w_next = next_s[storage_pass]

            # Events: left_link (leaving upstream link) + entered_link (entering downstream)
            if self.track_events:
                self._record_events(EVT_LEFT_LINK, winners, w_curr)
                self._record_events(EVT_ENTERED_LINK, winners, w_next)

            if self.collect_link_tt:
                ff_curr = self.ff_travel_time_steps[w_curr]
                delay = self.current_step - stuck_s[storage_pass]
                tt = (ff_curr + delay).float() * self.dt
                self.current_interval_tt_sum.scatter_add_(0, w_curr, tt)
                self.current_interval_tt_count.scatter_add_(0, w_curr, torch.ones_like(tt))

            # OPT: Update buffer_counts (agents leaving buffer)
            self.buffer_counts.scatter_add_(0, w_curr,
                -torch.ones(winners.size(0), device=self.device))

            # Update occupancy: add to downstream link
            self.edge_occupancy += torch.bincount(w_next, minlength=self.num_edges).to(self.edge_occupancy.dtype)

            # Update state
            self.status[winners] = 1  # Traveling (spatial buffer of downstream link)
            self.current_edge[winners] = w_next
            self.path_ptr[winners] += 1

            # Arrival time = current_step + freeflow_travel_time of downstream link
            ff_times = self.ff_travel_time_steps[w_next]
            arrival_times = (self.current_step + ff_times).int()
            self.arrival_time[winners] = arrival_times
            self.wakeup_time[winners] = arrival_times

    # =========================================================================
    # B. Accumulate Flow + Process Links (Spatial -> Capacity) + Handle Arrivals
    # =========================================================================
    def _process_links_B(self):
        """
        MATSim "moveLinks" phase:
        (a) Accumulate flow capacity on all edges.
        (b) Move agents from spatial buffer (status=1) to capacity buffer (status=2)
            if they satisfy freeflow_travel_time and flow capacity constraints.
        (c) Handle arrivals: agents in buffer with next_edge == -1 exit the network.
        """
        
        # Check if there are edges with spatial candidates
        spatial_mask = (self.status == 1) & (self.wakeup_time <= self.current_step)
        arrived_agents = torch.nonzero(spatial_mask, as_tuple=True)[0]
        if arrived_agents.numel() == 0:
            return

        # (a) Flow accumulation already done globally in step()

        # (b) moveQueueToBuffer: spatial -> capacity

        # Sort by (current_edge, arrival_time) for FIFO per-link ordering
        a_edges = self.current_edge[arrived_agents]
        a_arrival = self.arrival_time[arrived_agents]
        sort_keys = a_edges * 1000000 + a_arrival.long()
        sort_idx = torch.argsort(sort_keys, stable=True)

        arrived_sorted = arrived_agents[sort_idx]
        edges_sorted = a_edges[sort_idx]

        ptrs_sorted = self.path_ptr[arrived_sorted]
        next_ptrs_sorted = ptrs_sorted + 1

        # Determine exiter vs mover
        # Path-based: exiter if next path step is beyond agent's path length or is a sentinel
        agent_path_lens = self.path_offsets[arrived_sorted + 1] - self.path_offsets[arrived_sorted]
        is_exiter = (next_ptrs_sorted >= agent_path_lens)
        valid = ~is_exiter
        abs_ptrs = self.path_offsets[arrived_sorted[valid]] + next_ptrs_sorted[valid]
        path_next = self.paths_flat[abs_ptrs].long()
        is_exiter[valid] = (path_next == -1) | (path_next == -2)
        is_mover = ~is_exiter

        # --- Per-link grouping ---
        # OPT: cummax-based group starts (replaces unique_consecutive + repeat_interleave)
        n_arrived = arrived_sorted.size(0)
        positions_b = torch.arange(n_arrived, device=self.device)
        boundary_pos_b = torch.zeros(n_arrived, device=self.device, dtype=torch.long)
        if n_arrived > 1:
            boundary_pos_b[1:] = (edges_sorted[1:] != edges_sorted[:-1]).long() * positions_b[1:]
        item_starts, _ = torch.cummax(boundary_pos_b, dim=0)

        # --- FIFO blocking: a failing mover blocks all agents behind it ---
        # Grouped cumsum of is_mover → mover_rank (1-indexed per-link)
        mover_int = is_mover.int()
        global_cs_mover = torch.cumsum(mover_int, dim=0)
        offset_mover = global_cs_mover[item_starts] - mover_int[item_starts]
        mover_rank = global_cs_mover - offset_mover

        # Flow limits per agent (from their edge)
        # Subtract epsilon before ceil to avoid float32 precision artifacts
        # (e.g. -0.1+0.1 = 1.5e-8 in float32, ceil would give 1 instead of 0)
        flow_limits = torch.ceil(torch.clamp(self.edge_capacity_accumulator[edges_sorted] - 1e-6, min=0.0))

        # A mover fails if its per-link rank exceeds the flow limit
        mover_fails = is_mover & (mover_rank > flow_limits)

        # Grouped cumsum of failures → blocked if any prior mover on same link failed
        fails_int = mover_fails.int()
        global_cs_fails = torch.cumsum(fails_int, dim=0)
        offset_fails = global_cs_fails[item_starts] - fails_int[item_starts]
        blocked = (global_cs_fails - offset_fails) > 0

        processable = ~blocked
        proc_exit_mask = processable & is_exiter
        proc_win_mask = processable & is_mover

        # Common: remove all processable agents from link occupancy
        all_proc_edges = edges_sorted[processable]
        self.edge_occupancy -= torch.bincount(
            all_proc_edges, minlength=self.num_edges).to(self.edge_occupancy.dtype)

        # Common: add distance for traversed links (path_ptr > 0)
        all_proc_agents = arrived_sorted[processable]
        all_proc_ptrs = ptrs_sorted[processable]
        traversed = (all_proc_ptrs > 0)
        trav_agents = all_proc_agents[traversed]
        trav_c_legs = self.current_leg[trav_agents]
        self.leg_metrics[trav_agents, trav_c_legs, 0] += self.length[all_proc_edges[traversed]]

        # --- Exiters: leave the network (waiting for next leg dispatch) ---
        exiters = arrived_sorted[proc_exit_mask]

        if self.collect_link_tt and exiters.numel() > 0:
            exit_edges = edges_sorted[proc_exit_mask]
            ff_curr = self.ff_travel_time_steps[exit_edges]
            delay = self.current_step - self.stuck_since[exiters]
            tt = (ff_curr + delay).float() * self.dt
            self.current_interval_tt_sum.scatter_add_(0, exit_edges, tt)
            self.current_interval_tt_count.scatter_add_(0, exit_edges, torch.ones_like(tt))

        self.status[exiters] = 4

        if self.track_events:
            exit_edges = edges_sorted[proc_exit_mask]
            self._record_events(EVT_LEAVES_TRAFFIC, exiters, exit_edges)
            self._record_events(EVT_ARRIVAL, exiters, exit_edges)
            self._record_events(EVT_ACTSTART, exiters, exit_edges)

        # --- Winners: move to capacity buffer ---
        winners = arrived_sorted[proc_win_mask]
        w_edges = edges_sorted[proc_win_mask]

        # Consume flow capacity
        ones = torch.ones(winners.size(0), device=self.device)
        self.edge_capacity_accumulator.scatter_add_(0, w_edges, -ones)

        # Event: entered_buffer (spatial → capacity buffer)
        if self.track_events:
            self._record_events(EVT_ENTERED_BUFFER, winners, w_edges)

        # Status : Update to status, stuck_since, buffer_counts
        self.status[winners] = 2
        self.stuck_since[winners] = self.current_step
        self.buffer_counts.scatter_add_(0, w_edges, ones)

        # Compute next_edge
        self.next_edge[winners] = -1  # Default: end-of-path
        w_next_ptrs = next_ptrs_sorted[proc_win_mask]
        w_agent_path_lens = self.path_offsets[winners + 1] - self.path_offsets[winners]
        valid_ptr = w_next_ptrs < w_agent_path_lens
        valid_winners = winners[valid_ptr]
        valid_w_next_ptrs = w_next_ptrs[valid_ptr]
        abs_ptrs = self.path_offsets[valid_winners] + valid_w_next_ptrs
        self.next_edge[valid_winners] = self.paths_flat[abs_ptrs].long()

    # =========================================================================
    # C. Schedule Demand (New agents enter the network)
    # =========================================================================
    def _schedule_demand_C(self):
        """
        Phase 1: Process exiters (status=4). Transition to status=0 (next activity) 
                 or status=3 (done).
        Phase 2: Process waiting agents (status=0) whose departure time has arrived.
                 They enter the network directly into the capacity buffer of their 
                 first link (no spatial traversal required).
        """
        # --- Phase 1: Process Exiters (status=4) ---
        exiter_mask = (self.status == 4)
        exiters = torch.nonzero(exiter_mask, as_tuple=True)[0]
        
        if exiters.numel() > 0:
            c_legs = self.current_leg[exiters]
            n_legs = self.num_legs[exiters]
            
            has_more = (c_legs + 1) < n_legs
            
            # 1a. Agents entirely done
            done_agents = exiters[~has_more]
            if done_agents.numel() > 0:
                self.status[done_agents] = 3
                self.active_agents_count -= done_agents.numel()
                self.wakeup_time[done_agents] = self.infinity
                
                # Travel time for the final leg
                done_c_legs = self.current_leg[done_agents]
                dep_times = self.leg_departure_times[done_agents, done_c_legs]
                self.leg_metrics[done_agents, done_c_legs, 1] = (self.current_step - dep_times).float()
            
            # 1b. Agents with more legs
            cont_agents = exiters[has_more]
            if cont_agents.numel() > 0:
                c_legs_cont = c_legs[has_more]
                
                # Travel time for the completed intermediate leg
                dep_times_cont = self.leg_departure_times[cont_agents, c_legs_cont]
                self.leg_metrics[cont_agents, c_legs_cont, 1] = (self.current_step - dep_times_cont).float()
                
                # Fetch activity durations and end times (-1 if missing)
                end_t = self.act_end_times[cont_agents, c_legs_cont]
                dur_t = self.act_durations[cont_agents, c_legs_cont]
                
                # Priority: duration > end_time
                # "Un agent qui arrive à destination doit *au moins* attendre jusqu'à la fin de la duration."
                # "Si seulement end_time est spécifié, alors l'agent quitteras ... au end_time (ou après s'il arrive en retard)."
                wakeup = torch.full_like(end_t, self.infinity)
                
                has_end = end_t >= 0
                has_dur = dur_t >= 0
                
                both = has_end & has_dur
                only_end = has_end & ~has_dur
                only_dur = has_dur & ~has_end
                neither = ~has_end & ~has_dur
                
                # If both: wait at least duration. If end_time is later, wait until end_time.
                wakeup[both] = torch.maximum(end_t[both], (self.current_step + dur_t[both]).int())
                
                # If only end_time: wait until end_time, clamp to current_step if already late
                wakeup[only_end] = torch.clamp(end_t[only_end], min=self.current_step)
                
                # If only duration: wait exactly duration
                wakeup[only_dur] = (self.current_step + dur_t[only_dur]).int()
                
                # Neither
                wakeup[neither] = self.current_step
                
                # Transition to waiting/activity status
                self.wakeup_time[cont_agents] = wakeup
                self.status[cont_agents] = 0
                self.current_leg[cont_agents] += 1
                self._departure_emitted[cont_agents] = False # reset for next leg
                
                new_legs = self.current_leg[cont_agents]
                
                # Setup next_edge for the upcoming leg
                # In paths tensor, leg boundaries are separated by single -2
                # We just need to advance path_ptr by 2 (current is -2, next is start)
                self.path_ptr[cont_agents] += 2
                next_ptrs = self.path_ptr[cont_agents]
                # safe because next_ptrs < agent path len by definition of having more legs
                abs_ptrs = self.path_offsets[cont_agents] + next_ptrs
                self.next_edge[cont_agents] = self.paths_flat[abs_ptrs].long()

        # --- Phase 2: Process Waiting Agents (status=0) ---
        demand_mask = (self.status == 0) & (self.wakeup_time <= self.current_step)
        waiting_agents = torch.nonzero(demand_mask, as_tuple=True)[0]
        
        if waiting_agents.numel() == 0:
            return

        first_edges = self.next_edge[waiting_agents]

        # Record departure time for metrics (always, not just when tracking events).
        # Uses _departure_emitted flag to fire only once per leg.
        not_yet_emitted = ~self._departure_emitted[waiting_agents]
        departing_now = not_yet_emitted & (self.wakeup_time[waiting_agents] <= self.current_step)
        
        if departing_now.any():
            dep_agents = waiting_agents[departing_now]
            dep_edges = first_edges[departing_now]
            
            # Record departure time for the leg
            dep_c_legs = self.current_leg[dep_agents]
            self.leg_departure_times[dep_agents, dep_c_legs] = self.current_step
            
            self._departure_emitted[dep_agents] = True
            
            # Events: actend + departure
            if self.track_events:
                self._record_events(EVT_ACTEND, dep_agents, dep_edges)
                self._record_events(EVT_DEPARTURE, dep_agents, dep_edges)

        # Sort by (first_edge, departure_time) for FIFO per-link ordering
        dep_times = self.wakeup_time[waiting_agents]
        sort_keys = first_edges * 1000000 + dep_times.long()
        sort_idx = torch.argsort(sort_keys, stable=True)

        agents_sorted = waiting_agents[sort_idx]
        edges_sorted = first_edges[sort_idx]

        # Flow capacity check
        n_waiting = agents_sorted.size(0)
        positions_c = torch.arange(n_waiting, device=self.device)
        boundary_pos_c = torch.zeros(n_waiting, device=self.device, dtype=torch.long)
        if n_waiting > 1:
            boundary_pos_c[1:] = (edges_sorted[1:] != edges_sorted[:-1]).long() * positions_c[1:]
        group_starts_c, _ = torch.cummax(boundary_pos_c, dim=0)
        ranks = positions_c - group_starts_c

        flow_limits = torch.ceil(torch.clamp(self.edge_capacity_accumulator[edges_sorted] - 1e-6, min=0.0))
        flow_pass = (ranks < flow_limits)

        winners = agents_sorted[flow_pass]
        w_edges = self.next_edge[winners]

        # Consume flow capacity
        ones = torch.ones(winners.size(0), device=self.device)
        self.edge_capacity_accumulator.scatter_add_(0, w_edges, -ones)

        # Events: enters_traffic (only for agents that actually enter)
        if self.track_events:
            self._record_events(EVT_ENTERS_TRAFFIC, winners, w_edges)

        # Status : Update to status, current_edge, stuck_since, buffer_counts
        # Note : MATSim's behaviour is that agents enter capacity buffer directly on first link
        self.status[winners] = 2
        self.current_edge[winners] = w_edges
        self.stuck_since[winners] = self.current_step
        self.buffer_counts.scatter_add_(0, w_edges, ones)

        # Compute next_edge
        self.next_edge[winners] = -1  # Default: end-of-path
        next_ptrs = self.path_ptr[winners] + 1
        w_agent_path_lens = self.path_offsets[winners + 1] - self.path_offsets[winners]
        valid_ptr_mask = next_ptrs < w_agent_path_lens
        v_agents = winners[valid_ptr_mask]
        v_next_ptrs = next_ptrs[valid_ptr_mask]
        abs_ptrs = self.path_offsets[v_agents] + v_next_ptrs
        self.next_edge[v_agents] = self.paths_flat[abs_ptrs].long()

    # =========================================================================
    # Finalize agents that are still in-network when simulation ends
    # =========================================================================
    def finalize_stuck_agents(self):
        """Finalize metrics for agents still in-network at the end of simulation.

        Called after the simulation loop completes. For each agent:
        - If the agent has started a leg but not finished (status 0/1/2/4),
          impute travel_time = (current_step - departure_time) + FFTT of remaining edges.
        - If the agent is in status 0 (waiting) and hasn't started the leg,
          impute travel_time = FFTT of the entire leg, and set departure time to current_step.
        """
        if self.paths_flat is None:
            # Fallback to original logic if paths_flat is not available
            # (e.g. RL mode or empty paths)
            not_done = (self.status != 3)
            active_agents = torch.nonzero(not_done, as_tuple=True)[0]
            if active_agents.numel() > 0:
                c_legs = self.current_leg[active_agents]
                dep_times = self.leg_departure_times[active_agents, c_legs]
                started_mask = dep_times >= 0
                if started_mask.any():
                    started_agents = active_agents[started_mask]
                    started_c_legs = c_legs[started_mask]
                    started_dep = dep_times[started_mask]
                    self.leg_metrics[started_agents, started_c_legs, 1] = (
                        self.current_step - started_dep
                    ).float()
                    self.n_imputed_legs = int(started_mask.sum().item())
            return

        import numpy as np
        status_cpu = self.status.cpu().numpy()
        current_leg_cpu = self.current_leg.cpu().numpy()
        leg_dep_cpu = self.leg_departure_times.cpu().numpy()
        path_ptr_cpu = self.path_ptr.cpu().numpy()
        path_offsets_cpu = self.path_offsets.cpu().numpy()
        paths_flat_cpu = self.paths_flat.cpu().numpy()
        ff_travel_time_steps_cpu = self.ff_travel_time_steps.cpu().numpy()
        
        leg_metrics_cpu = self.leg_metrics.cpu().numpy()
        
        n_imputed = 0
        active_indices = np.nonzero(status_cpu != 3)[0]
        
        for i in active_indices:
            l = current_leg_cpu[i]
            dep_time = leg_dep_cpu[i, l]
            
            if dep_time >= 0:
                start_ptr = path_ptr_cpu[i] + 1
                elapsed = self.current_step - dep_time
            else:
                start_ptr = path_ptr_cpu[i]
                elapsed = 0
                
            start_idx = path_offsets_cpu[i] + start_ptr
            end_idx = path_offsets_cpu[i + 1]
            
            remaining_fftt = 0.0
            for idx in range(start_idx, end_idx):
                edge = paths_flat_cpu[idx]
                if edge == -2:
                    break
                if edge >= 0:
                    remaining_fftt += ff_travel_time_steps_cpu[edge]
                    
            imputed_tt = float(elapsed) + remaining_fftt
            leg_metrics_cpu[i, l, 1] = imputed_tt
            
            if dep_time < 0:
                leg_dep_cpu[i, l] = self.current_step
                
            n_imputed += 1
            
        self.leg_metrics.copy_(torch.from_numpy(leg_metrics_cpu).to(self.device))
        self.leg_departure_times.copy_(torch.from_numpy(leg_dep_cpu).to(self.device))
        self.n_imputed_legs = n_imputed

    # =========================================================================
    # Step
    # =========================================================================
    def step(self):
        # A. Process Nodes (Capacity -> Downstream Spatial)
        self._process_nodes_A()

        # OPT: Global flow accumulation once per step
        self._update_all_flow_accumulation()

        # B. Process Links (Spatial -> Capacity) + Handle Arrivals
        self._process_links_B()

        # C. Schedule Demand (New agents enter network)
        self._schedule_demand_C()

        # Check if the time interval changed
        if self.collect_link_tt:
            next_interval_idx = int(((self.current_step + 1) * self.dt) // self.link_tt_interval)
            if next_interval_idx > self.current_interval_idx:
                # Grow CPU tensors if necessary, keeping them in pinned memory if CUDA is available
                if next_interval_idx >= self.interval_tt_sum.size(0):
                    new_size = max(next_interval_idx + 10, self.interval_tt_sum.size(0) * 2)
                    pin_memory = torch.cuda.is_available()
                    new_sum = torch.zeros((new_size, self.num_edges), device='cpu', dtype=torch.float32, pin_memory=pin_memory)
                    new_count = torch.zeros((new_size, self.num_edges), device='cpu', dtype=torch.float32, pin_memory=pin_memory)
                    new_sum[:self.interval_tt_sum.size(0)] = self.interval_tt_sum
                    new_count[:self.interval_tt_count.size(0)] = self.interval_tt_count
                    self.interval_tt_sum = new_sum
                    self.interval_tt_count = new_count
                
                # Flush current GPU buffer to the CPU tensor asynchronously (non-blocking)
                self.interval_tt_sum[self.current_interval_idx].copy_(self.current_interval_tt_sum, non_blocking=True)
                self.interval_tt_count[self.current_interval_idx].copy_(self.current_interval_tt_count, non_blocking=True)
                
                # Zero out GPU buffers for the next interval
                self.current_interval_tt_sum.zero_()
                self.current_interval_tt_count.zero_()
                
                self.current_interval_idx = next_interval_idx

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
            # RAM
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"RAM: {mem_mb:.2f} MB")
            
            # VRAM
            vram_mb = 0
            if self.device == 'cuda':
                vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"VRAM: {vram_mb:.2f} MB")
            
            print("-----------------------")
        else:
            print("Profiling was not enabled.")
        
        if self.stuck_count > 0:
            print(f"⚠️  [WARNING] | {self.stuck_count} agents stuck/removed | Action: Verify paths/network")

        return compute_time, mem_mb, vram_mb

    def get_snapshot(self):
        """Returns current edge occupancy"""
        return self.edge_occupancy.clone()

    def get_metrics(self):
        """
        Return dict with:
        - arrived_count
        - en_route_count
        - avg_travel_time (completed legs)
        - avg_travel_dist (completed legs)
        """
        done_mask = (self.status == 3)
        # En route: Status 1 (Traveling) or 2 (Buffer)
        en_route_mask = (self.status == 1) | (self.status == 2)
        
        arrived_count = done_mask.sum().item()
        en_route_count = en_route_mask.sum().item()
        
        avg_time = 0.0
        avg_dist = 0.0
        
        # Valid legs have travel time > 0
        leg_m = self.leg_metrics.view(-1, 2)
        valid_legs = leg_m[:, 1] > 0
        
        if valid_legs.sum().item() > 0:
            avg_time = leg_m[valid_legs, 1].mean().item()
            avg_dist = leg_m[valid_legs, 0].mean().item()
            
        return {
            "arrived_count": arrived_count,
            "en_route_count": en_route_count,
            "avg_travel_time": avg_time,
            "avg_travel_dist": avg_dist,
            "n_imputed_legs": getattr(self, "n_imputed_legs", 0)
        }

    def get_leg_histogram(self):
        """
        Return histogram of path_pointer for en-route agents.
        """
        en_route_mask = (self.status == 1) | (self.status == 2)
        pointers = self.path_ptr[en_route_mask]
        if pointers.numel() == 0:
            return torch.zeros(1, device=self.device)
        max_ptr = int(pointers.max().item()) + 1
        hist = torch.histc(pointers.float(), bins=max_ptr, min=0, max=max_ptr-1)
        return hist

    def get_events(self):
        """
        Return the list of recorded events, sorted by time.
        Events are processed on CPU to avoid CUDA OOM for large scenarios.
        """
        if not self.track_events:
            return []
            
        self._flush_events()
        
        if len(self._cpu_events_blocks) == 0:
            return []
            
        # Concatenate all CPU blocks
        events_tensor = torch.cat(self._cpu_events_blocks, dim=0)
        
        # Sort on CPU to save VRAM
        sort_keys = events_tensor[:, 0].long() * 100 + events_tensor[:, 1].long()
        sort_idx = torch.argsort(sort_keys, stable=True)
        events_sorted = events_tensor[sort_idx]
        
        # Convert to list of tuples
        events_cpu = events_sorted.tolist()
        return [tuple(row) for row in events_cpu]

    def get_dynamic_link_travel_times(self) -> torch.Tensor:
        """Returns [num_intervals_active, num_edges] with average travel times."""
        if not self.collect_link_tt:
            return None
        
        # Flush the final (possibly incomplete) interval to CPU (blocking here to ensure data availability)
        self.interval_tt_sum[self.current_interval_idx].copy_(self.current_interval_tt_sum)
        self.interval_tt_count[self.current_interval_idx].copy_(self.current_interval_tt_count)
        
        max_active_interval = self.current_interval_idx
        if max_active_interval >= self.interval_tt_sum.size(0):
             max_active_interval = self.interval_tt_sum.size(0) - 1
             
        sum_tt = self.interval_tt_sum[:max_active_interval + 1]
        count_tt = self.interval_tt_count[:max_active_interval + 1]
        
        # Perform computation on CPU to conserve GPU VRAM
        avg_tt = sum_tt / count_tt.clamp(min=1.0)
        
        # Where count is 0, fallback to ff_travel_time
        ff_seconds = self.edge_static[:, 4].cpu().unsqueeze(0).expand_as(avg_tt)
        avg_tt = torch.where(count_tt > 0, avg_tt, ff_seconds)
        
        return avg_tt.to(self.device)
