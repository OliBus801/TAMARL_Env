import torch
import torch._inductor.config
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
                 enable_profiling: bool = False,
                 compile: str = None):
        """
        Initialize the TorchDNLGEMSim simulation engine (Saprykin et al., 2025).
        
        Args:
            edge_static: Tensor [E, 5] -> [length, free_flow_speed, capacity_storage, capacity_flow, ff_travel_time]
            paths: Tensor [A, MaxPathLen] -> Pre-calculated path indices for each agent.
            net_topology: Tuple (upstream_indices, upstream_offsets) CSR adjacency.
            device: 'cuda' or 'cpu'.
            departure_times: Tensor [A] -> Departure times.
            stuck_threshold: Steps before gridlock squeeze-in.
            dt: Time step seconds.
            enable_profiling: Enable cProfile (incompatible with compile).
            compile: None | 'default' | 'reduce-overhead' | 'max-autotune'
        """
        self.device = device
        self.stuck_threshold = stuck_threshold
        self.dt = dt
        self.length_veh = 7.5
        self.period = 3600.0
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
        self.flow_capacity_per_step = self.edge_static[:, 3].contiguous()
        self.ff_travel_time_steps = torch.ceil(self.edge_static[:, 4] / self.dt).int().contiguous()
        
        # Topology
        if net_topology is None:
             raise ValueError("net_topology is required for TorchDNLGEMSim")
        self.upstream_indices = net_topology[0].to(device).long()
        self.upstream_offsets = net_topology[1].to(device).long()
        self.num_nodes = self.upstream_offsets.shape[0] - 1
        
        # --- Ring Buffers ---
        self.in_queue_sizes = torch.clamp(self.storage_capacity.int(), min=1) + self.squeeze_margin
        self.out_queue_sizes = torch.clamp(self.flow_capacity_per_step.int(), min=1) + self.squeeze_margin

        self.in_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.int32)
        self.in_queue_offsets[1:] = torch.cumsum(self.in_queue_sizes, dim=0)
        self.total_in_queue_size = self.in_queue_offsets[-1].item()
        
        self.out_queue_offsets = torch.zeros(self.num_edges + 1, device=device, dtype=torch.int32)
        self.out_queue_offsets[1:] = torch.cumsum(self.out_queue_sizes, dim=0)
        self.total_out_queue_size = self.out_queue_offsets[-1].item()
        
        self.in_cur = torch.zeros(self.num_edges, device=device, dtype=torch.int)
        self.in_cnt = torch.zeros(self.num_edges, device=device, dtype=torch.int)
        self.out_cur = torch.zeros(self.num_edges, device=device, dtype=torch.int)
        self.out_cnt = torch.zeros(self.num_edges, device=device, dtype=torch.int)

        self.in_queues = torch.full((self.total_in_queue_size,), -1, device=device, dtype=torch.int32)
        self.out_queues = torch.full((self.total_out_queue_size,), -1, device=device, dtype=torch.int32)
        
        # Agent State
        self.status = torch.zeros(self.num_agents, device=self.device, dtype=torch.uint8)
        self.current_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.int32)
        self.next_edge = torch.full((self.num_agents,), -1, device=self.device, dtype=torch.int32)
        self.path_ptr = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.arrival_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.stuck_since = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.start_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.enter_link_time = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        
        if departure_times is not None:
             self.departure_times = departure_times.to(self.device).int()
        else:
             self.departure_times = torch.zeros(self.num_agents, device=self.device, dtype=torch.int32)
        self.start_time.copy_(self.departure_times)
        self.next_edge[:] = self.paths[:, 0]
        self.wakeup_time = self.departure_times.clone()
        self.infinity = 2**30
        self.agent_metrics = torch.zeros((self.num_agents, 2), device=self.device, dtype=torch.float32)
        self.current_step = torch.tensor(0, device=self.device, dtype=torch.int32)
        self.flow_accumulator = torch.zeros(self.num_edges, device=device, dtype=torch.float32)
        self.last_link_exit_time = torch.zeros(self.num_edges, device=device, dtype=torch.int32) - 1000

        # --- Pre-computed constants for CUDAGraph-friendly ops ---
        self._in_offsets = self.in_queue_offsets[:self.num_edges].contiguous()
        self._out_offsets = self.out_queue_offsets[:self.num_edges].contiguous()
        self._flow_cap_ceil = torch.maximum(torch.tensor(1.0, device=device), self.flow_capacity_per_step)
        self._zero_f = torch.tensor(0.0, device=device)
        self._one_f = torch.tensor(1.0, device=device)
        self._status_traveling = torch.tensor(1, device=device, dtype=torch.uint8)
        self._status_buffer = torch.tensor(2, device=device, dtype=torch.uint8)
        self._status_done = torch.tensor(3, device=device, dtype=torch.uint8)
        # Scratch buffer for edge→agent scatter in _process_links
        self._agent_scatter = torch.zeros(self.num_agents, device=device, dtype=torch.int32)

        # --- torch.compile ---
        if compile and not enable_profiling:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
            torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
            print(f"torch.compile enabled (mode={compile}). First steps will be slow (compilation)...")
            self._process_links = torch.compile(self._process_links, mode=compile, dynamic=True)
            self._process_nodes = torch.compile(self._process_nodes, mode=compile, dynamic=True)
            self._schedule_demand = torch.compile(self._schedule_demand, mode=compile, dynamic=True)

    def reset(self):
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
        self.current_step.zero_()
        self.flow_accumulator.fill_(0)
        self.last_link_exit_time.fill_(-1000)
        if self.profiler:
            self.profiler.clear()

    @staticmethod
    def _compute_sorted_ranks(sorted_values, n, device):
        """
        Compute within-group ranks for a SORTED tensor.
        [A, A, B, B, B, C] -> [0, 1, 0, 1, 2, 0]
        O(n), torch.compile compatible (no .item()).
        """
        if n <= 1:
            return torch.zeros(n, device=device, dtype=torch.int32)
        changes = sorted_values[1:] != sorted_values[:-1]
        group_ids = torch.zeros(n, device=device, dtype=torch.int32)
        group_ids[1:] = changes.int().cumsum_(0)
        starts = torch.zeros(n, device=device, dtype=torch.int32)
        boundary_idx = changes.nonzero(as_tuple=True)[0] + 1
        starts[boundary_idx] = boundary_idx.int()
        group_start_per_elem, _ = starts.cummax(dim=0)
        return torch.arange(n, device=device, dtype=torch.int32) - group_start_per_elem

    def _process_links(self):
        """
        Kernel 1: Links — CUDAGraph-optimized.
        ALL ops on fixed-size [num_edges] or [num_agents] tensors.
        No nonzero(), no dynamic indexing. Uses boolean masks + where().
        """
        if not (self.in_cnt > 0).any():
            return

        # Accumulate capacity — full-tensor [num_edges]
        active = self.in_cnt > 0
        self.flow_accumulator.add_(torch.where(active, self.flow_capacity_per_step, self._zero_f))
        self.flow_accumulator = torch.where(
            active,
            torch.minimum(self.flow_accumulator, self._flow_cap_ceil),
            self.flow_accumulator
        )

        while True:
            # 1. Eligible links mask — fixed [num_edges]
            mask = (self.in_cnt > 0) & (self.out_cnt < self.out_queue_sizes) & (self.flow_accumulator > 0)

            # 2. Front agent per edge — fixed [num_edges]
            front_agents = self.in_queues[(self._in_offsets + self.in_cur).long()]

            # 3. Ready check — fixed [num_edges]
            safe_front = front_agents.clamp(min=0)
            ready = mask & (front_agents >= 0) & (
                (self.current_step - self.enter_link_time[safe_front]) >= self.ff_travel_time_steps
            )

            if not ready.any():
                break

            # 4. removeFront — fixed [num_edges], no dynamic shapes
            self.in_cnt -= ready.int()
            self.in_cur = torch.where(ready, ((self.in_cur + 1) % self.in_queue_sizes.int()), self.in_cur)

            # 5. pushBack to out_queue — fixed [num_edges]
            out_pos = (self._out_offsets + ((self.out_cur + self.out_cnt) % self.out_queue_sizes)).long()
            existing = self.out_queues[out_pos]
            self.out_queues[out_pos] = torch.where(ready, front_agents, existing)
            self.out_cnt += ready.int()

            # 6. Flow update — fixed [num_edges]
            self.flow_accumulator -= torch.where(ready, self._one_f, self._zero_f)

            # 7. Link-level state — fixed [num_edges]
            self.last_link_exit_time = torch.where(ready, self.current_step, self.last_link_exit_time)

            # 8. Agent-level state via scatter — fixed [num_edges] → fixed [num_agents]
            self._agent_scatter.zero_()
            self._agent_scatter.scatter_add_(0, safe_front.long(), ready.int())
            moved = self._agent_scatter > 0
            self.status = torch.where(moved, self._status_buffer, self.status)
            self.stuck_since = torch.where(moved, self.current_step, self.stuck_since)

    def _process_nodes(self):
        """
        Kernel 2: Intersections — move agents from out_queue to in_queue of next link.
        Uses bincount for counter updates (fixed [num_edges] output).
        Sorting/ranking inherently requires dynamic shapes.
        """
        while True:
            active_links = torch.nonzero(self.out_cnt > 0, as_tuple=True)[0]
            if active_links.numel() == 0:
                break

            agent_ids = self.out_queues[(self._out_offsets[active_links] + self.out_cur[active_links]).long()]

            curr_ptrs = self.path_ptr[agent_ids]
            next_ptrs = curr_ptrs + 1
            valid_idx_mask = next_ptrs < self.max_path_len
            next_links = torch.full_like(curr_ptrs, -1)
            if valid_idx_mask.any():
                valid_agents = agent_ids[valid_idx_mask]
                next_links[valid_idx_mask] = self.paths[valid_agents, next_ptrs[valid_idx_mask].long()]

            # Handle Exits
            exit_mask = next_links == -1
            if exit_mask.any():
                exiting_links = active_links[exit_mask]
                exiting_agents = agent_ids[exit_mask]
                # removeFront via bincount — fixed [num_edges] output
                exit_dec = torch.bincount(exiting_links.long(), minlength=self.num_edges)
                self.out_cnt -= exit_dec.int()
                self.out_cur = (self.out_cur + exit_dec.int()) % self.out_queue_sizes.int()
                # Agent state
                self.status[exiting_agents] = 3
                self.wakeup_time[exiting_agents] = self.infinity
                self.agent_metrics[exiting_agents, 1] = (self.current_step - self.start_time[exiting_agents]).float()
                self.agent_metrics[exiting_agents, 0] += self.length[exiting_links]

            # Handle Transfers
            cont_mask = ~exit_mask
            if not cont_mask.any():
                continue

            cont_links = active_links[cont_mask]
            cont_agents = agent_ids[cont_mask]
            cont_targets = next_links[cont_mask]

            sort_idx = torch.argsort(cont_targets)
            s_targets = cont_targets[sort_idx]
            s_agents = cont_agents[sort_idx]
            s_links = cont_links[sort_idx]
            n = s_targets.shape[0]

            sorted_ranks = self._compute_sorted_ranks(s_targets, n, self.device)
            available_capacity = self.in_queue_sizes[s_targets] - self.in_cnt[s_targets] - self.squeeze_margin
            winners_mask = sorted_ranks < available_capacity

            # Gridlock resolution
            stuck_mask = ~winners_mask
            if stuck_mask.any():
                stuck_agents = s_agents[stuck_mask]
                stuck_dur = self.current_step - self.stuck_since[stuck_agents]
                is_stuck_long = stuck_dur > self.stuck_threshold
                item_avails_phy = available_capacity[stuck_mask] + self.squeeze_margin
                has_phy_space = sorted_ranks[stuck_mask] < item_avails_phy
                winners_mask[stuck_mask] = is_stuck_long & has_phy_space

            if not winners_mask.any():
                break

            win_from = s_links[winners_mask]
            win_to = s_targets[winners_mask]
            win_agents_id = s_agents[winners_mask]
            win_ranks = sorted_ranks[winners_mask]

            # removeFront via bincount — fixed [num_edges] output
            remove_dec = torch.bincount(win_from.long(), minlength=self.num_edges)
            self.out_cnt -= remove_dec.int()
            self.out_cur = (self.out_cur + remove_dec.int()) % self.out_queue_sizes.int()

            # pushBackAtomic — queue write (inherently dynamic) + bincount counter (fixed)
            write_indices = (self.in_cur[win_to] + self.in_cnt[win_to] + win_ranks) % self.in_queue_sizes[win_to]
            global_write_idx = self.in_queue_offsets[win_to] + write_indices
            self.in_queues[global_write_idx.long()] = win_agents_id
            add_dec = torch.bincount(win_to.long(), minlength=self.num_edges)
            self.in_cnt += add_dec.int()

            # Agent state
            self.status[win_agents_id] = 1
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
        first_edges = self.paths[active_agents, 0]
        
        sort_idx = torch.argsort(first_edges)
        s_agents = active_agents[sort_idx]
        s_edges = first_edges[sort_idx]
        n = s_agents.shape[0]
        
        sorted_ranks = self._compute_sorted_ranks(s_edges, n, self.device)
        avail = self.in_queue_sizes[s_edges] - self.in_cnt[s_edges]
        passed = sorted_ranks < avail
        
        if passed.any():
            w_agents = s_agents[passed]
            w_edges = s_edges[passed]
            w_ranks = sorted_ranks[passed]
            
            t_heads = self.in_cur[w_edges]
            t_counts = self.in_cnt[w_edges]
            t_sizes = self.in_queue_sizes[w_edges]
            t_offsets = self.in_queue_offsets[w_edges]
            
            write_curr = (t_heads + t_counts + w_ranks) % t_sizes
            global_write = t_offsets + write_curr
            self.in_queues[global_write.long()] = w_agents.int()
            
            # Counter update via bincount — fixed [num_edges]
            counts_delta = torch.bincount(w_edges.long(), minlength=self.num_edges)
            self.in_cnt += counts_delta.int()
            
            self.status[w_agents] = 1
            self.current_edge[w_agents] = w_edges
            self.enter_link_time[w_agents] = self.current_step

    def step(self):
        self._process_links()
        self._process_nodes()
        self._schedule_demand()
        self.current_step += 1

    def step_no_compile(self):
        """Same as step(), guaranteed not compiled (for profiling)."""
        self._process_links()
        self._process_nodes()
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
