
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
                 paths: torch.Tensor,
                 device: str = 'cuda',
                 departure_times: torch.Tensor = None,
                 edge_endpoints: torch.Tensor = None,
                 stuck_threshold: int = 10,
                 dt: float = 1.0,
                 seed: int = None,
                 enable_profiling: bool = False,
                 track_events: bool = False):
        """
        Initialize the TorchDNLMATSim simulation engine.
        
        Args:
            edge_static: Tensor [E, 5] -> [length, free_flow_speed, capacity_storage (c_e), capacity_flow (D_e per hour), ff_travel_time]
            paths: Tensor [A, MaxPathLen] -> Pre-calculated path indices for each agent.
            device: Device to run on ('cuda' or 'cpu').
            departure_times: Tensor [A] -> Departure times for each agent.
            edge_endpoints: Tensor [E, 2] -> [from_node, to_node] for connectivity validation. Optional.
            stuck_threshold: Time in steps an agent waits in buffer before forcing entry.
            dt: Simulation time step in seconds.
            enable_profiling: If True, enables cProfile.
            track_events: If True, records MATSim-style simulation events.
        """
        self.device = device
        self.stuck_threshold = stuck_threshold
        self.dt = dt
        
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
        self._init_event_buffer()

        # Test number of interactions
        self.interactions = 0
        self.stuck_count = 0

        # Edge connectivity (needed for path validation)
        if edge_endpoints is None:
            raise ValueError("edge_endpoints is mandatory for path validation.")
        self.edge_endpoints = edge_endpoints.to(device).int()

        # Move constants to device
        self.edge_static = edge_static.to(device)
        self.paths = paths.to(device).long()  # OPT: store as int64 to avoid .long() conversions
        
        self.num_edges = self.edge_static.shape[0]
        self.num_agents = self.paths.shape[0]
        self.max_path_len = self.paths.shape[1]

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
        self.flow_accumulator_last_updated = torch.zeros(self.num_edges, device=self.device, dtype=torch.int32)
        self.step_edge_limits = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        # OPT: Incremental buffer counts (avoids recomputing bincount on all agents)
        self.buffer_counts = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        # Agent State - Structure of Arrays (SOA)
        # Status | 0: Waiting, 1: Traveling, 2: Buffer, 3: Done
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
        self.next_edge[:] = self.paths[:, 0]
        
        # Agent Metrics [A, 2] -> [accumulated_distance, final_travel_time]
        self.agent_metrics = torch.zeros((self.num_agents, 2), device=self.device, dtype=torch.float32)

        # Optimization: Wakeup Time [A]
        # Unified scheduler: agents are only processed if current_step >= wakeup_time
        self.wakeup_time = self.departure_times.clone()
        self.infinity = 2**30

        # Track whether actend/departure events have been emitted (for track_events)
        self._departure_emitted = torch.zeros(self.num_agents, device=self.device, dtype=torch.bool)

        # OPT: Pre-allocate reusable scratch tensors
        self._edge_priority_scratch = torch.zeros(self.num_edges, device=self.device, dtype=torch.float32)

        self.current_step = 0

    def _init_event_buffer(self):
        """Initialize event buffer as a pre-allocated tensor (or None if not tracking)."""
        if self.track_events:
            # Pre-allocate for ~100k events, will grow as needed
            initial_cap = 100_000
            self._event_buffer = torch.empty((initial_cap, 4), device=self.device, dtype=torch.int32)
            self._event_count = 0
        else:
            self._event_buffer = None
            self._event_count = 0

    def reset(self):
        self.edge_occupancy.fill_(0)
        self.edge_capacity_accumulator.copy_(self.flow_capacity_per_step)
        self.flow_accumulator_last_updated.fill_(0)
        self.step_edge_limits.fill_(0)
        self.buffer_counts.fill_(0)
        
        self.status.fill_(0)
        self.current_edge.fill_(-1)
        self.next_edge[:] = self.paths[:, 0]
        self.path_ptr.fill_(0)
        self.arrival_time.fill_(0)
        self.stuck_since.fill_(0)
        self.start_time.copy_(self.departure_times)
        
        self.agent_metrics.fill_(0)
        self.wakeup_time.copy_(self.departure_times)
        self._departure_emitted.fill_(False)
        self.current_step = 0
        self.stuck_count = 0
        if self.seed is not None:
            self.rng.manual_seed(self.seed)
        self._init_event_buffer()
        
        if self.profiler:
            self.profiler.clear()

    def _record_events(self, event_type, agent_ids, edge_ids):
        """Record events using a pre-allocated tensor buffer. All data stays on device."""
        if not self.track_events:
            return
        n = agent_ids.size(0)
        if n == 0:
            return
        # Grow buffer if needed (doubling strategy)
        needed = self._event_count + n
        if needed > self._event_buffer.size(0):
            new_cap = max(self._event_buffer.size(0) * 2, needed)
            new_buf = torch.empty((new_cap, 4), device=self.device, dtype=torch.int32)
            if self._event_count > 0:
                new_buf[:self._event_count] = self._event_buffer[:self._event_count]
            self._event_buffer = new_buf
        
        buf = self._event_buffer
        buf[self._event_count:needed, 0] = self.current_step
        buf[self._event_count:needed, 1] = event_type
        buf[self._event_count:needed, 2] = agent_ids.int()
        buf[self._event_count:needed, 3] = edge_ids.int()
        self._event_count = needed

    # =========================================================================
    # Flow Accumulation (MATSim: updateFastFlowAccumulation)
    # =========================================================================
    def _update_flow_accumulation(self, edges):
        """
        Lazy on-demand flow accumulation for the specified edges.
        Matches MATSim's updateFastFlowAccumulation: computes elapsed time since
        last update, accumulates flow_per_step * elapsed, caps at remaining
        capacity (flow_per_step - buffer_count).
        
        Only updates edges where: last_updated < now AND acc < remaining.
        
        OPT: Uses pre-maintained self.buffer_counts instead of recomputing bincount.
        """
        unique_edges = torch.unique(edges)
        e = unique_edges.long()
        
        # OPT: Use pre-maintained buffer_counts instead of recomputing
        remaining = self.flow_capacity_per_step[e] - self.buffer_counts[e]
        
        # Condition: last_updated < now AND acc < remaining
        last_updated = self.flow_accumulator_last_updated[e]
        acc = self.edge_capacity_accumulator[e]
        should_update = (last_updated < self.current_step) & (acc < remaining)
        
        if should_update.any():
            upd_idx = e[should_update]
            upd_last = last_updated[should_update]
            upd_acc = acc[should_update]
            upd_remaining = remaining[should_update]
            
            elapsed = (self.current_step - upd_last).float()
            accumulated = elapsed * self.flow_capacity_per_step[upd_idx]
            new_acc = torch.minimum(upd_acc + accumulated, upd_remaining)
            
            self.edge_capacity_accumulator[upd_idx] = new_acc
            self.flow_accumulator_last_updated[upd_idx] = self.current_step

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
        # OPT: Skip nonzero + numel check instead of .any() (avoids GPU sync)
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
            self.wakeup_time[stuck_agents] = self.infinity
            start_times = self.start_time[stuck_agents]
            self.agent_metrics[stuck_agents, 1] = (self.current_step - start_times).float()
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
        unique_v, _, counts_v = torch.unique_consecutive(next_s, return_counts=True, return_inverse=True)
        v_starts = torch.zeros_like(unique_v)
        v_starts[1:] = torch.cumsum(counts_v, dim=0)[:-1]
        item_v_starts = torch.repeat_interleave(v_starts, counts_v)
        inflow_ranks = torch.arange(movers_s.size(0), device=self.device) - item_v_starts

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

            # OPT: Update buffer_counts (agents leaving buffer)
            self.buffer_counts.scatter_add_(0, w_curr,
                -torch.ones(winners.size(0), device=self.device))

            # Update occupancy: add to downstream link
            self.edge_occupancy += torch.bincount(w_next.int(), minlength=self.num_edges)

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
        # OPT: Compute nonzero directly, check numel (avoids .any() GPU sync)
        arrived_agents = torch.nonzero(spatial_mask, as_tuple=True)[0]
        if arrived_agents.numel() == 0:
            return

        # (a) Lazy flow accumulation: only update edges that have spatial candidates
        candidate_edges = self.current_edge[arrived_agents]
        self._update_flow_accumulation(candidate_edges)

        # (b) moveQueueToBuffer: spatial -> capacity

        # Sort by (current_edge, arrival_time) for FIFO per-link ordering
        a_edges = self.current_edge[arrived_agents]
        a_arrival = self.arrival_time[arrived_agents]
        sort_keys = a_edges * 1000000 + a_arrival.long()
        sort_idx = torch.argsort(sort_keys, stable=True)

        arrived_sorted = arrived_agents[sort_idx]
        edges_sorted = a_edges[sort_idx]

        # --- Shared computations (done ONCE for all candidates) ---
        ptrs_sorted = self.path_ptr[arrived_sorted]
        next_ptrs_sorted = ptrs_sorted + 1

        # Determine exiter vs mover: exiter if next path step is beyond path or padding
        is_exiter = (next_ptrs_sorted >= self.max_path_len)
        if not is_exiter.all():
            valid = ~is_exiter
            path_next = self.paths[arrived_sorted[valid], next_ptrs_sorted[valid]]
            is_exiter[valid] = (path_next == -1)
        is_mover = ~is_exiter

        # --- Per-link grouping ---
        unique_e, _, counts_e = torch.unique_consecutive(
            edges_sorted, return_counts=True, return_inverse=True)
        e_starts = torch.zeros_like(unique_e)
        e_starts[1:] = torch.cumsum(counts_e, dim=0)[:-1]
        item_starts = torch.repeat_interleave(e_starts, counts_e)

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

        has_exiters = proc_exit_mask.any()
        has_winners = proc_win_mask.any()

        if has_exiters or has_winners:
            # Common: remove all processable agents from link occupancy
            all_proc_edges = edges_sorted[processable]
            self.edge_occupancy -= torch.bincount(
                all_proc_edges.int(), minlength=self.num_edges)

            # Common: add distance for traversed links (path_ptr > 0)
            all_proc_agents = arrived_sorted[processable]
            all_proc_ptrs = ptrs_sorted[processable]
            traversed = (all_proc_ptrs > 0)
            if traversed.any():
                self.agent_metrics[all_proc_agents[traversed], 0] += \
                    self.length[all_proc_edges[traversed]]

        # --- Exiters: leave the network ---
        if has_exiters:
            exiters = arrived_sorted[proc_exit_mask]
            self.status[exiters] = 3
            self.wakeup_time[exiters] = self.infinity
            self.agent_metrics[exiters, 1] = \
                (self.current_step - self.start_time[exiters]).float()

            if self.track_events:
                exit_edges = edges_sorted[proc_exit_mask]
                self._record_events(EVT_LEAVES_TRAFFIC, exiters, exit_edges)
                self._record_events(EVT_ARRIVAL, exiters, exit_edges)
                self._record_events(EVT_ACTSTART, exiters, exit_edges)

        # --- Winners: move to capacity buffer ---
        if has_winners:
            winners = arrived_sorted[proc_win_mask]
            w_edges = edges_sorted[proc_win_mask]

            # Consume flow capacity (and mark consumption time for lazy accumulation)
            ones = torch.ones(winners.size(0), device=self.device)
            self.edge_capacity_accumulator.scatter_add_(0, w_edges, -ones)
            self.flow_accumulator_last_updated[w_edges] = self.current_step

            # Event: entered_buffer (spatial → capacity buffer)
            if self.track_events:
                self._record_events(EVT_ENTERED_BUFFER, winners, w_edges)

            self.status[winners] = 2
            self.stuck_since[winners] = self.current_step

            # OPT: Update buffer_counts (agents entering buffer)
            self.buffer_counts.scatter_add_(0, w_edges, ones)

            # Compute next_edge (reusing pre-computed next_ptrs_sorted)
            w_next_ptrs = next_ptrs_sorted[proc_win_mask]
            # OPT: Set to -1 first, then overwrite valid entries
            self.next_edge[winners] = -1
            valid_ptr = w_next_ptrs < self.max_path_len
            if valid_ptr.any():
                self.next_edge[winners[valid_ptr]] = self.paths[
                    winners[valid_ptr], w_next_ptrs[valid_ptr]]

    # =========================================================================
    # C. Schedule Demand (New agents enter the network)
    # =========================================================================
    def _schedule_demand_C(self):
        """
        Process waiting agents (status=0) whose departure time has arrived.
        They enter the network directly into the capacity buffer of their 
        first link (no spatial traversal required).
        
        Constraints:
        - Flow capacity of the first link must be available (FIFO order)
        
        Events emitted: actend, departure (at departure time), enters_traffic (when entering)
        """
        demand_mask = (self.status == 0) & (self.wakeup_time <= self.current_step)
        # OPT: Direct nonzero + numel check instead of .any()
        waiting_agents = torch.nonzero(demand_mask, as_tuple=True)[0]
        if waiting_agents.numel() == 0:
            return

        first_edges = self.next_edge[waiting_agents]

        # Events: actend + departure fire once per agent when their departure time has been reached.
        # Uses _departure_emitted flag to avoid re-emitting for agents that fail flow check
        # and are re-processed on subsequent steps.
        if self.track_events:
            not_yet_emitted = ~self._departure_emitted[waiting_agents]
            departing_now = not_yet_emitted & (self.departure_times[waiting_agents] <= self.current_step)
            if departing_now.any():
                dep_agents = waiting_agents[departing_now]
                dep_edges = first_edges[departing_now]
                self._record_events(EVT_ACTEND, dep_agents, dep_edges)
                self._record_events(EVT_DEPARTURE, dep_agents, dep_edges)
                self._departure_emitted[dep_agents] = True

        # Sort by (first_edge, departure_time) for FIFO per-link ordering
        dep_times = self.departure_times[waiting_agents]
        sort_keys = first_edges * 1000000 + dep_times.long()
        sort_idx = torch.argsort(sort_keys, stable=True)

        agents_sorted = waiting_agents[sort_idx]
        edges_sorted = first_edges[sort_idx]

        # Lazy flow accumulation for edges with waiting agents
        self._update_flow_accumulation(edges_sorted)

        # Flow capacity check
        unique_e, _, counts_e = torch.unique_consecutive(edges_sorted, return_counts=True, return_inverse=True)
        e_starts = torch.zeros_like(unique_e)
        e_starts[1:] = torch.cumsum(counts_e, dim=0)[:-1]
        item_starts = torch.repeat_interleave(e_starts, counts_e)
        ranks = torch.arange(agents_sorted.size(0), device=self.device) - item_starts

        flow_limits = torch.ceil(torch.clamp(self.edge_capacity_accumulator[edges_sorted] - 1e-6, min=0.0))
        flow_pass = (ranks < flow_limits)

        winners = agents_sorted[flow_pass]

        if winners.numel() == 0:
            return

        w_edges = self.next_edge[winners]

        # Consume flow capacity (and mark consumption time for lazy accumulation)
        ones = torch.ones(winners.size(0), device=self.device)
        self.edge_capacity_accumulator.scatter_add_(0, w_edges, -ones)
        self.flow_accumulator_last_updated[w_edges] = self.current_step

        # Events: enters_traffic (only for agents that actually enter)
        if self.track_events:
            self._record_events(EVT_ENTERS_TRAFFIC, winners, w_edges)

        # Enter capacity buffer directly (status=2, no spatial traversal)
        self.status[winners] = 2
        self.current_edge[winners] = w_edges
        self.stuck_since[winners] = self.current_step

        # OPT: Update buffer_counts (agents entering buffer)
        self.buffer_counts.scatter_add_(0, w_edges, ones)

        # Compute next_edge
        next_ptrs = self.path_ptr[winners] + 1
        # OPT: Set to -1 first, then overwrite valid entries
        self.next_edge[winners] = -1
        valid_ptr_mask = next_ptrs < self.max_path_len
        if valid_ptr_mask.any():
            v_agents = winners[valid_ptr_mask]
            v_next_ptrs = next_ptrs[valid_ptr_mask]
            self.next_edge[v_agents] = self.paths[v_agents, v_next_ptrs]

        # Note: occupancy NOT added here (buffer is not part of link occupancy)
        # Note: first link distance NOT added (agent didn't traverse spatial buffer)

    # =========================================================================
    # Step
    # =========================================================================
    def step(self):
        # A. Process Nodes (Capacity -> Downstream Spatial)
        self._process_nodes_A()

        # B. Accumulate Flow + Process Links (Spatial -> Capacity) + Handle Arrivals
        self._process_links_B()

        # C. Schedule Demand (New agents enter network)
        self._schedule_demand_C()

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

    def get_events(self):
        """
        Return the list of recorded events, sorted by time.
        Each event is a tuple: (time, event_type_id, agent_id, edge_id).
        Use EVENT_TYPE_NAMES to map type_id to string name.
        
        OPT: Events are stored as a GPU tensor, sorted with torch.argsort,
        then converted to list of tuples only at export time.
        """
        if self._event_buffer is None or self._event_count == 0:
            return []
        
        # Slice to actual events
        events_tensor = self._event_buffer[:self._event_count]
        
        # Sort by (time, event_type)
        sort_keys = events_tensor[:, 0].long() * 100 + events_tensor[:, 1].long()
        sort_idx = torch.argsort(sort_keys, stable=True)
        events_sorted = events_tensor[sort_idx]
        
        # Convert to list of tuples (on CPU)
        events_cpu = events_sorted.cpu().tolist()
        return [tuple(row) for row in events_cpu]
