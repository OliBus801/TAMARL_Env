"""
JAX-based Dynamic Network Loading (DNL) MATSim Simulator.

Functional rewrite of dnl_matsim.py (PyTorch) for jax.jit compilation.
All state is immutable (PyTree); step() takes state in, returns new state.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import chex

# Event type constants (same as dnl_matsim.py)
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

INFINITY = 2**30


# =============================================================================
# Data Structures (PyTrees)
# =============================================================================

@chex.dataclass
class SimState:
    """Mutable simulation state — a JAX PyTree."""
    # Agent state [A]
    status: jnp.ndarray           # uint8: 0=waiting, 1=traveling, 2=buffer, 3=done, 4=exiter
    current_edge: jnp.ndarray     # int32
    next_edge: jnp.ndarray        # int32
    path_ptr: jnp.ndarray         # int32
    arrival_time: jnp.ndarray     # int32
    stuck_since: jnp.ndarray      # int32
    start_time: jnp.ndarray       # int32
    wakeup_time: jnp.ndarray      # int32
    current_leg: jnp.ndarray      # int32
    departure_emitted: jnp.ndarray  # bool

    # Edge state [E]
    edge_occupancy: jnp.ndarray          # int32
    edge_capacity_accumulator: jnp.ndarray  # float32
    buffer_counts: jnp.ndarray           # float32

    # Metrics [A, MaxLegs, 2] -> [distance, travel_time]
    leg_metrics: jnp.ndarray      # float32
    leg_departure_times: jnp.ndarray  # int32

    # Simulation clock
    current_step: jnp.ndarray     # int32 scalar

    # RNG key
    rng_key: jnp.ndarray

    # Counters
    stuck_count: jnp.ndarray      # int32 scalar

    # --- Trajectory tracking (eval mode) [A, max_path_len] ---
    trajectory_edges: jnp.ndarray        # int32
    trajectory_enter_times: jnp.ndarray  # int32
    trajectory_exit_times: jnp.ndarray   # int32

    # Link travel time collection
    interval_tt_sum: jnp.ndarray    # float32 [max_intervals, E]
    interval_tt_count: jnp.ndarray  # float32 [max_intervals, E]


@chex.dataclass
class SimParams:
    """Static simulation parameters — passed to JIT as a PyTree of arrays.
    
    These do NOT change during a simulation episode.
    """
    # Edge attributes [E]
    length: jnp.ndarray              # float32
    free_speed: jnp.ndarray          # float32
    storage_capacity: jnp.ndarray    # float32
    flow_capacity_per_step: jnp.ndarray  # float32
    ff_travel_time_steps: jnp.ndarray    # int32
    edge_endpoints: jnp.ndarray      # int32 [E, 2]

    # Agent fixed data [A]
    departure_times: jnp.ndarray     # int32
    num_legs: jnp.ndarray            # int32

    # Paths [A, MaxPathLen] (non-RL mode only; -1 padded)
    paths: jnp.ndarray               # int32

    # Activity scheduling [A, MaxActs]
    act_end_times: jnp.ndarray       # int32
    act_durations: jnp.ndarray       # int32

    # RL mode data
    first_edges: jnp.ndarray         # int32 [A, MaxLegs]
    destinations: jnp.ndarray        # int32 [A, MaxLegs]

    # Dimensions (as arrays so they live in the PyTree)
    num_agents: jnp.ndarray          # int32 scalar
    num_edges: jnp.ndarray           # int32 scalar
    max_path_len: jnp.ndarray        # int32 scalar
    max_legs_count: jnp.ndarray      # int32 scalar
    stuck_threshold: jnp.ndarray     # int32 scalar
    dt: jnp.ndarray                  # float32 scalar
    link_tt_interval: jnp.ndarray    # float32 scalar
    max_intervals: jnp.ndarray       # int32 scalar


# =============================================================================
# Initialization
# =============================================================================

def create_params(
    edge_static,          # np [E, 5]
    edge_endpoints,       # np [E, 2]
    departure_times,      # np [A]
    paths=None,           # np [A, MaxPathLen] or None
    num_legs=None,        # np [A] or None
    act_end_times=None,   # np [A, MaxActs] or None
    act_durations=None,   # np [A, MaxActs] or None
    first_edges=None,     # np [A, MaxLegs] or None
    destinations=None,    # np [A, MaxLegs] or None
    stuck_threshold=10,
    dt=1.0,
    link_tt_interval=300.0,
    max_intervals=100,
) -> SimParams:
    """Create SimParams from numpy arrays (called once, outside JIT)."""
    edge_static = np.asarray(edge_static, dtype=np.float32)
    edge_endpoints = np.asarray(edge_endpoints, dtype=np.int32)
    departure_times = np.asarray(departure_times, dtype=np.int32)

    num_edges = edge_static.shape[0]
    num_agents = departure_times.shape[0]

    length = edge_static[:, 0]
    free_speed = edge_static[:, 1]
    storage_capacity = edge_static[:, 2]
    flow_capacity_per_step = edge_static[:, 3]
    ff_travel_time_steps = np.floor(edge_static[:, 4] / dt).astype(np.int32)

    if paths is not None:
        paths = np.asarray(paths, dtype=np.int32)
        max_path_len = paths.shape[1]
    else:
        paths = np.full((num_agents, 1), -1, dtype=np.int32)
        max_path_len = 0

    if num_legs is not None:
        num_legs = np.asarray(num_legs, dtype=np.int32)
    else:
        num_legs = np.ones(num_agents, dtype=np.int32)

    max_legs_count = int(num_legs.max()) if num_agents > 0 else 1

    if act_end_times is not None:
        act_end_times = np.asarray(act_end_times, dtype=np.int32)
    else:
        act_end_times = np.full((num_agents, 0), -1, dtype=np.int32)

    if act_durations is not None:
        act_durations = np.asarray(act_durations, dtype=np.int32)
    else:
        act_durations = np.full((num_agents, 0), -1, dtype=np.int32)

    if first_edges is not None:
        first_edges = np.asarray(first_edges, dtype=np.int32)
    else:
        first_edges = np.full((num_agents, max_legs_count), -1, dtype=np.int32)

    if destinations is not None:
        destinations = np.asarray(destinations, dtype=np.int32)
    else:
        destinations = np.full((num_agents, max_legs_count), -1, dtype=np.int32)

    return SimParams(
        length=jnp.array(length),
        free_speed=jnp.array(free_speed),
        storage_capacity=jnp.array(storage_capacity),
        flow_capacity_per_step=jnp.array(flow_capacity_per_step),
        ff_travel_time_steps=jnp.array(ff_travel_time_steps),
        edge_endpoints=jnp.array(edge_endpoints),
        departure_times=jnp.array(departure_times),
        num_legs=jnp.array(num_legs),
        paths=jnp.array(paths),
        act_end_times=jnp.array(act_end_times),
        act_durations=jnp.array(act_durations),
        first_edges=jnp.array(first_edges),
        destinations=jnp.array(destinations),
        num_agents=jnp.int32(num_agents),
        num_edges=jnp.int32(num_edges),
        max_path_len=jnp.int32(max_path_len),
        max_legs_count=jnp.int32(max_legs_count),
        stuck_threshold=jnp.int32(stuck_threshold),
        dt=jnp.float32(dt),
        link_tt_interval=jnp.float32(link_tt_interval),
        max_intervals=jnp.int32(max_intervals),
    )


def init_state(params: SimParams, seed: int = 0) -> SimState:
    """Create initial SimState from SimParams (called once, outside JIT)."""
    A = int(params.num_agents)
    E = int(params.num_edges)
    max_path_len = int(params.max_path_len)
    max_legs = int(params.max_legs_count)
    max_intervals = int(params.max_intervals)

    rl_mode = (max_path_len == 0)

    if not rl_mode:
        initial_next_edge = params.paths[:, 0]
    else:
        initial_next_edge = params.first_edges[:, 0]

    return SimState(
        status=jnp.zeros(A, dtype=jnp.uint8),
        current_edge=jnp.full(A, -1, dtype=jnp.int32),
        next_edge=initial_next_edge.astype(jnp.int32),
        path_ptr=jnp.zeros(A, dtype=jnp.int32),
        arrival_time=jnp.zeros(A, dtype=jnp.int32),
        stuck_since=jnp.zeros(A, dtype=jnp.int32),
        start_time=params.departure_times.copy(),
        wakeup_time=params.departure_times.copy(),
        current_leg=jnp.zeros(A, dtype=jnp.int32),
        departure_emitted=jnp.zeros(A, dtype=jnp.bool_),
        edge_occupancy=jnp.zeros(E, dtype=jnp.int32),
        edge_capacity_accumulator=params.flow_capacity_per_step.copy(),
        buffer_counts=jnp.zeros(E, dtype=jnp.float32),
        leg_metrics=jnp.zeros((A, max_legs, 2), dtype=jnp.float32),
        leg_departure_times=jnp.zeros((A, max_legs), dtype=jnp.int32),
        current_step=jnp.int32(0),
        rng_key=jax.random.PRNGKey(seed),
        stuck_count=jnp.int32(0),
        trajectory_edges=jnp.full((A, max(max_path_len, 1)), -1, dtype=jnp.int32),
        trajectory_enter_times=jnp.full((A, max(max_path_len, 1)), -1, dtype=jnp.int32),
        trajectory_exit_times=jnp.full((A, max(max_path_len, 1)), -1, dtype=jnp.int32),
        interval_tt_sum=jnp.zeros((max_intervals, E), dtype=jnp.float32),
        interval_tt_count=jnp.zeros((max_intervals, E), dtype=jnp.float32),
    )


def reset_state(params: SimParams, state: SimState) -> SimState:
    """Reset state for a new episode (pure function)."""
    return init_state(params, seed=int(state.rng_key[0]))


# =============================================================================
# Flow Accumulation
# =============================================================================

def _update_all_flow_accumulation(state: SimState, params: SimParams) -> SimState:
    """Global flow accumulation for ALL edges, once per step."""
    remaining = params.flow_capacity_per_step - state.buffer_counts
    new_acc = jnp.minimum(
        state.edge_capacity_accumulator + params.flow_capacity_per_step,
        remaining,
    )
    return state.replace(edge_capacity_accumulator=new_acc)


# =============================================================================
# A. Process Nodes (Capacity Buffer -> Downstream Spatial Buffer)
# =============================================================================

def _process_nodes_A(state: SimState, params: SimParams, is_eval: bool) -> SimState:
    """
    Move agents from upstream capacity buffer (status=2) to downstream
    spatial buffer (status=1).

    JAX rewrite: operates on ALL agents with masks instead of dynamic indexing.
    The sort-based priority logic is replaced with a per-agent random priority
    that achieves the same MATSim "emptyBufferAfterBufferRandomDistribution".
    """
    step = state.current_step
    A = state.status.shape[0]

    # --- Identify buffer agents ---
    is_buffer = (state.status == 2) & (state.wakeup_time <= step)
    has_next = (state.next_edge != -1)
    is_candidate = is_buffer & has_next  # [A] bool

    # Safe edge lookups (clamp -1 to 0 for indexing, mask will filter)
    curr_e = jnp.clip(state.current_edge, 0)
    next_e = jnp.clip(state.next_edge, 0)

    # --- Connectivity check ---
    to_of_curr = params.edge_endpoints[curr_e, 1]
    from_of_next = params.edge_endpoints[next_e, 0]
    connected = (to_of_curr == from_of_next)

    is_disconnected = is_candidate & ~connected
    is_mover = is_candidate & connected

    # --- Handle disconnected (stuck) agents ---
    new_status = jnp.where(is_disconnected, jnp.uint8(3), state.status)
    new_wakeup = jnp.where(is_disconnected, INFINITY, state.wakeup_time)
    stuck_count_delta = is_disconnected.sum()

    # Leg metrics for stuck agents
    c_legs_stuck = state.current_leg
    dep_times_stuck = state.leg_departure_times[jnp.arange(A), c_legs_stuck]
    tt_stuck = (step - dep_times_stuck).astype(jnp.float32)
    new_leg_metrics = state.leg_metrics.at[jnp.arange(A), c_legs_stuck, 1].add(
        jnp.where(is_disconnected, tt_stuck, 0.0)
    )

    # Buffer counts: remove disconnected agents from their current edge's buffer
    buffer_delta_disc = jnp.where(is_disconnected, -1.0, 0.0)
    new_buffer_counts = state.buffer_counts.at[curr_e].add(buffer_delta_disc)

    # --- Trajectory tracking for stuck agents (eval) ---
    new_traj_edges = state.trajectory_edges
    new_traj_enter = state.trajectory_enter_times
    new_traj_exit = state.trajectory_exit_times
    if is_eval:
        ptr = state.path_ptr
        # Record exit time for stuck agents on their current edge
        new_traj_exit = new_traj_exit.at[jnp.arange(A), ptr].set(
            jnp.where(is_disconnected, step, new_traj_exit[jnp.arange(A), ptr])
        )

    # --- Random priority for mover sorting ---
    # MATSim: upstream links feeding the same downstream node are randomly
    # prioritized by capacity. We assign a random key per agent, weighted
    # by the flow capacity of their current edge.
    key, subkey = jax.random.split(state.rng_key)
    rand_vals = jax.random.uniform(subkey, shape=(A,))
    weighted_priority = rand_vals * params.flow_capacity_per_step[curr_e]

    # --- Sort movers by (downstream_edge, -priority, agent_id) ---
    # Build a sort key. High priority = processed first (negate for ascending sort).
    # Only movers matter; non-movers get a huge sort key to be pushed to the end.
    sort_key = jnp.where(
        is_mover,
        state.next_edge * 10_000_000 - (weighted_priority * 1_000_000).astype(jnp.int32),
        jnp.int32(2_000_000_000),
    )
    sort_idx = jnp.argsort(sort_key, stable=True)

    # Gather sorted values
    next_sorted = state.next_edge[sort_idx]
    curr_sorted = state.current_edge[sort_idx]
    mover_sorted = is_mover[sort_idx]
    stuck_since_sorted = state.stuck_since[sort_idx]

    # --- Per-downstream-link inflow ranks (cummax trick) ---
    positions = jnp.arange(A)
    boundary = jnp.where(
        (positions > 0) & (next_sorted != jnp.roll(next_sorted, 1)) & mover_sorted,
        positions,
        0,
    )
    group_starts = jax.lax.associative_scan(jnp.maximum, boundary)
    inflow_ranks = positions - group_starts

    # Storage capacity check
    caps = params.storage_capacity[jnp.clip(next_sorted, 0)]
    occs = state.edge_occupancy[jnp.clip(next_sorted, 0)]
    avail = caps - occs

    time_in_buffer = step - stuck_since_sorted
    is_stuck = time_in_buffer > params.stuck_threshold
    storage_pass = mover_sorted & (is_stuck | (inflow_ranks < avail))

    # --- Apply winners (unsort back to agent order) ---
    # Create winner mask in original agent order
    winner_in_sorted = storage_pass
    # Scatter back to original indices
    inv_sort = jnp.argsort(sort_idx)
    winner_mask = winner_in_sorted[inv_sort]  # [A] bool in original order

    w_curr = state.current_edge  # [A]
    w_next = state.next_edge     # [A]

    # Remove winners from upstream buffer_counts
    buffer_delta_win = jnp.where(winner_mask, -1.0, 0.0)
    new_buffer_counts = new_buffer_counts.at[curr_e].add(buffer_delta_win)

    # Add winners to downstream occupancy
    occ_delta = jnp.where(winner_mask, 1, 0).astype(jnp.int32)
    new_occupancy = state.edge_occupancy.at[next_e].add(occ_delta)

    # Update agent state for winners
    new_status = jnp.where(winner_mask, jnp.uint8(1), new_status)
    new_current_edge = jnp.where(winner_mask, w_next, state.current_edge)
    new_path_ptr = jnp.where(winner_mask, state.path_ptr + 1, state.path_ptr)

    ff_times = params.ff_travel_time_steps[next_e]
    new_arrival = jnp.where(winner_mask, step + ff_times, state.arrival_time)
    new_wakeup = jnp.where(winner_mask, step + ff_times, new_wakeup)

    # --- Trajectory tracking for winners (eval) ---
    if is_eval:
        # Record exit from upstream link
        ptr_win = state.path_ptr  # ptr before increment
        new_traj_exit = new_traj_exit.at[jnp.arange(A), ptr_win].set(
            jnp.where(winner_mask, step, new_traj_exit[jnp.arange(A), ptr_win])
        )
        # Record entry into downstream link
        ptr_new = new_path_ptr
        new_traj_edges = new_traj_edges.at[jnp.arange(A), ptr_new].set(
            jnp.where(winner_mask, w_next, new_traj_edges[jnp.arange(A), ptr_new])
        )
        new_traj_enter = new_traj_enter.at[jnp.arange(A), ptr_new].set(
            jnp.where(winner_mask, step, new_traj_enter[jnp.arange(A), ptr_new])
        )

    # --- Link TT collection for winners ---
    interval_idx = (step * params.dt / params.link_tt_interval).astype(jnp.int32)
    interval_idx = jnp.clip(interval_idx, 0, params.max_intervals - 1)
    ff_curr = params.ff_travel_time_steps[curr_e]
    delay = step - state.stuck_since
    tt_val = (ff_curr + delay).astype(jnp.float32) * params.dt
    new_tt_sum = state.interval_tt_sum.at[interval_idx, curr_e].add(
        jnp.where(winner_mask, tt_val, 0.0)
    )
    new_tt_count = state.interval_tt_count.at[interval_idx, curr_e].add(
        jnp.where(winner_mask, 1.0, 0.0)
    )

    return state.replace(
        status=new_status,
        current_edge=new_current_edge,
        path_ptr=new_path_ptr,
        arrival_time=new_arrival,
        wakeup_time=new_wakeup,
        edge_occupancy=new_occupancy,
        buffer_counts=new_buffer_counts,
        leg_metrics=new_leg_metrics,
        stuck_count=state.stuck_count + stuck_count_delta,
        rng_key=key,
        trajectory_edges=new_traj_edges,
        trajectory_enter_times=new_traj_enter,
        trajectory_exit_times=new_traj_exit,
        interval_tt_sum=new_tt_sum,
        interval_tt_count=new_tt_count,
    )


# =============================================================================
# B. Process Links (Spatial -> Capacity) + Handle Arrivals
# =============================================================================

def _process_links_B(state: SimState, params: SimParams, is_eval: bool) -> SimState:
    """
    MATSim 'moveLinks' phase:
    (a) Move agents from spatial buffer (status=1) to capacity buffer (status=2)
        if they satisfy freeflow_travel_time and flow capacity constraints.
    (b) Handle arrivals: agents reaching end-of-path exit the network.

    JAX rewrite: fixed-size masks, no dynamic indexing.
    """
    step = state.current_step
    A = state.status.shape[0]
    E = params.length.shape[0]
    max_path_len = params.paths.shape[1]

    # --- Identify spatial candidates ---
    is_spatial = (state.status == 1) & (state.wakeup_time <= step)

    # Safe edge lookup
    curr_e = jnp.clip(state.current_edge, 0)
    agent_ids = jnp.arange(A)

    # --- Determine exiter vs mover ---
    ptrs = state.path_ptr
    next_ptrs = ptrs + 1

    # Path-based (non-RL): exiter if next path step is beyond path or sentinel
    beyond_path = (next_ptrs >= max_path_len)
    # Safe lookup of next path element
    safe_next_ptr = jnp.clip(next_ptrs, 0, jnp.maximum(max_path_len - 1, 0))
    path_next_val = params.paths[agent_ids, safe_next_ptr]
    is_sentinel = (path_next_val == -1) | (path_next_val == -2)
    is_exiter_path = beyond_path | is_sentinel

    # RL mode: exiter if to_node of current edge == destination
    curr_to_node = params.edge_endpoints[curr_e, 1]
    c_legs = state.current_leg
    max_legs_count = params.first_edges.shape[1]
    agent_dest = params.destinations[agent_ids, jnp.clip(c_legs, 0, max_legs_count - 1)]
    is_exiter_rl = (curr_to_node == agent_dest)

    # Select based on mode (max_path_len == 0 means RL mode)
    is_exiter = jnp.where(max_path_len > 0, is_exiter_path, is_exiter_rl)
    is_mover_type = ~is_exiter

    # --- Sort candidates by (edge, arrival_time) for FIFO ---
    sort_key = jnp.where(
        is_spatial,
        curr_e * 1_000_000 + state.arrival_time.astype(jnp.int32),
        jnp.int32(2_000_000_000),
    )
    sort_idx = jnp.argsort(sort_key, stable=True)

    edges_sorted = state.current_edge[sort_idx]
    spatial_sorted = is_spatial[sort_idx]
    is_mover_sorted = is_mover_type[sort_idx] & spatial_sorted
    is_exiter_sorted = is_exiter[sort_idx] & spatial_sorted

    # --- Per-link grouping (cummax trick) ---
    positions = jnp.arange(A)
    safe_edges_sorted = jnp.clip(edges_sorted, 0)
    boundary = jnp.where(
        (positions > 0) & (safe_edges_sorted != jnp.roll(safe_edges_sorted, 1)) & spatial_sorted,
        positions,
        0,
    )
    item_starts = jax.lax.associative_scan(jnp.maximum, boundary)

    # --- Mover ranks (grouped cumsum) ---
    mover_int = is_mover_sorted.astype(jnp.int32)
    global_cs_mover = jnp.cumsum(mover_int)
    offset_mover = global_cs_mover[item_starts] - mover_int[item_starts]
    mover_rank = global_cs_mover - offset_mover

    # Flow limits
    flow_limits = jnp.ceil(jnp.clip(
        state.edge_capacity_accumulator[safe_edges_sorted] - 1e-6, min=0.0
    ))

    # FIFO blocking: a failing mover blocks all behind on same link
    mover_fails = is_mover_sorted & (mover_rank > flow_limits)
    fails_int = mover_fails.astype(jnp.int32)
    global_cs_fails = jnp.cumsum(fails_int)
    offset_fails = global_cs_fails[item_starts] - fails_int[item_starts]
    blocked = (global_cs_fails - offset_fails) > 0

    processable = spatial_sorted & ~blocked
    proc_exit_sorted = processable & is_exiter_sorted
    proc_win_sorted = processable & is_mover_sorted

    # --- Unsort back to agent order ---
    inv_sort = jnp.argsort(sort_idx)
    proc_mask = processable[inv_sort]
    exit_mask = proc_exit_sorted[inv_sort]
    win_mask = proc_win_sorted[inv_sort]

    # --- Remove all processable agents from link occupancy ---
    occ_delta = jnp.where(proc_mask, -1, 0).astype(jnp.int32)
    new_occupancy = state.edge_occupancy.at[curr_e].add(occ_delta)

    # --- Distance accumulation for traversed links ---
    traversed = proc_mask & (ptrs > 0)
    trav_c_legs = state.current_leg
    new_leg_metrics = state.leg_metrics.at[agent_ids, trav_c_legs, 0].add(
        jnp.where(traversed, params.length[curr_e], 0.0)
    )

    # --- Exiters: leave the network (status -> 4) ---
    new_status = jnp.where(exit_mask, jnp.uint8(4), state.status)

    # --- Link TT for exiters ---
    interval_idx = (step * params.dt / params.link_tt_interval).astype(jnp.int32)
    interval_idx = jnp.clip(interval_idx, 0, params.max_intervals - 1)
    ff_exit = params.ff_travel_time_steps[curr_e]
    delay_exit = step - state.stuck_since
    tt_exit = (ff_exit + delay_exit).astype(jnp.float32) * params.dt
    new_tt_sum = state.interval_tt_sum.at[interval_idx, curr_e].add(
        jnp.where(exit_mask, tt_exit, 0.0)
    )
    new_tt_count = state.interval_tt_count.at[interval_idx, curr_e].add(
        jnp.where(exit_mask, 1.0, 0.0)
    )

    # --- Winners: move to capacity buffer (status -> 2) ---
    new_status = jnp.where(win_mask, jnp.uint8(2), new_status)
    new_stuck_since = jnp.where(win_mask, step, state.stuck_since)

    # Consume flow capacity
    cap_delta = jnp.where(win_mask, -1.0, 0.0)
    new_cap_acc = state.edge_capacity_accumulator.at[curr_e].add(cap_delta)

    # Buffer counts
    buf_delta = jnp.where(win_mask, 1.0, 0.0)
    new_buffer_counts = state.buffer_counts.at[curr_e].add(buf_delta)

    # Compute next_edge for winners
    new_next_edge = jnp.where(win_mask, -1, state.next_edge)
    # Path mode: look up next path element
    safe_next = jnp.clip(next_ptrs, 0, jnp.maximum(max_path_len - 1, 0))
    path_next = params.paths[agent_ids, safe_next]
    valid_ptr = (next_ptrs < max_path_len) & (max_path_len > 0)
    new_next_edge = jnp.where(win_mask & valid_ptr, path_next, new_next_edge)

    # --- Trajectory tracking (eval) ---
    new_traj_edges = state.trajectory_edges
    new_traj_enter = state.trajectory_enter_times
    new_traj_exit = state.trajectory_exit_times
    if is_eval:
        # Exiters: record exit time
        ptr_e = state.path_ptr
        new_traj_exit = new_traj_exit.at[agent_ids, ptr_e].set(
            jnp.where(exit_mask, step, new_traj_exit[agent_ids, ptr_e])
        )
        # Winners entering buffer: record entering buffer (exit from spatial)
        new_traj_exit = new_traj_exit.at[agent_ids, ptr_e].set(
            jnp.where(win_mask, step, new_traj_exit[agent_ids, ptr_e])
        )

    return state.replace(
        status=new_status,
        stuck_since=new_stuck_since,
        next_edge=new_next_edge,
        edge_occupancy=new_occupancy,
        edge_capacity_accumulator=new_cap_acc,
        buffer_counts=new_buffer_counts,
        leg_metrics=new_leg_metrics,
        interval_tt_sum=new_tt_sum,
        interval_tt_count=new_tt_count,
        trajectory_edges=new_traj_edges,
        trajectory_enter_times=new_traj_enter,
        trajectory_exit_times=new_traj_exit,
    )


# =============================================================================
# C. Schedule Demand (New agents enter the network)
# =============================================================================

def _schedule_demand_C(state: SimState, params: SimParams, is_eval: bool) -> SimState:
    """
    Phase 1: Process exiters (status=4) -> transition to activity or done.
    Phase 2: Process waiting agents (status=0) -> enter network.

    JAX rewrite: fully masked, no dynamic indexing.
    """
    step = state.current_step
    A = state.status.shape[0]
    agent_ids = jnp.arange(A)
    max_path_len = params.paths.shape[1]

    # ===================== Phase 1: Process Exiters (status=4) =====================
    is_exiter = (state.status == 4)

    c_legs = state.current_leg
    n_legs = params.num_legs
    has_more = (c_legs + 1) < n_legs

    # 1a. Agents entirely done (exiter & no more legs)
    is_done = is_exiter & ~has_more
    new_status = jnp.where(is_done, jnp.uint8(3), state.status)
    new_wakeup = jnp.where(is_done, INFINITY, state.wakeup_time)

    # Travel time for the final leg
    dep_times_done = state.leg_departure_times[agent_ids, c_legs]
    tt_done = (step - dep_times_done).astype(jnp.float32)
    new_leg_metrics = state.leg_metrics.at[agent_ids, c_legs, 1].add(
        jnp.where(is_done, tt_done, 0.0)
    )

    # 1b. Agents with more legs (exiter & has_more)
    is_cont = is_exiter & has_more

    # Travel time for the completed intermediate leg
    dep_times_cont = state.leg_departure_times[agent_ids, c_legs]
    tt_cont = (step - dep_times_cont).astype(jnp.float32)
    new_leg_metrics = new_leg_metrics.at[agent_ids, c_legs, 1].add(
        jnp.where(is_cont, tt_cont, 0.0)
    )

    # Activity scheduling (wakeup calculation)
    max_acts = params.act_end_times.shape[1] if params.act_end_times.ndim > 1 else 0
    safe_act_idx = jnp.clip(c_legs, 0, jnp.maximum(max_acts - 1, 0))

    # Fetch activity times (safe even if max_acts == 0 due to clip)
    end_t = jnp.where(max_acts > 0,
        params.act_end_times[agent_ids, safe_act_idx],
        jnp.int32(-1))
    dur_t = jnp.where(max_acts > 0,
        params.act_durations[agent_ids, safe_act_idx],
        jnp.int32(-1))

    has_end = end_t >= 0
    has_dur = dur_t >= 0
    both = has_end & has_dur
    only_end = has_end & ~has_dur
    only_dur = has_dur & ~has_end

    wakeup_cont = jnp.int32(step)  # default: immediate
    wakeup_cont = jnp.where(both, jnp.maximum(end_t, step + dur_t), wakeup_cont)
    wakeup_cont = jnp.where(only_end, jnp.maximum(end_t, step), wakeup_cont)
    wakeup_cont = jnp.where(only_dur, step + dur_t, wakeup_cont)

    new_wakeup = jnp.where(is_cont, wakeup_cont, new_wakeup)
    new_status = jnp.where(is_cont, jnp.uint8(0), new_status)
    new_current_leg = jnp.where(is_cont, c_legs + 1, state.current_leg)
    new_dep_emitted = jnp.where(is_cont, False, state.departure_emitted)

    # Setup next_edge for the upcoming leg
    new_legs = new_current_leg
    new_path_ptr = state.path_ptr
    new_next_edge = state.next_edge

    # Path mode: advance path_ptr by 2 (skip -2 sentinel)
    new_path_ptr = jnp.where(is_cont & (max_path_len > 0),
        state.path_ptr + 2, new_path_ptr)
    safe_ptr = jnp.clip(new_path_ptr, 0, jnp.maximum(max_path_len - 1, 0))
    path_next = params.paths[agent_ids, safe_ptr]
    new_next_edge = jnp.where(is_cont & (max_path_len > 0), path_next, new_next_edge)

    # RL mode: use first_edges for the new leg
    safe_leg = jnp.clip(new_legs, 0, params.first_edges.shape[1] - 1)
    rl_first = params.first_edges[agent_ids, safe_leg]
    new_next_edge = jnp.where(is_cont & (max_path_len == 0), rl_first, new_next_edge)

    # ===================== Phase 2: Process Waiting Agents (status=0) =====================
    is_waiting = (new_status == 0) & (new_wakeup <= step)
    first_edges_w = new_next_edge

    # Record departure time (once per leg)
    not_emitted = ~new_dep_emitted
    departing_now = is_waiting & not_emitted

    new_leg_dep_times = state.leg_departure_times.at[agent_ids, new_current_leg].set(
        jnp.where(departing_now, step, state.leg_departure_times[agent_ids, new_current_leg])
    )
    new_dep_emitted = jnp.where(departing_now, True, new_dep_emitted)

    # --- Sort waiting agents by (first_edge, departure_time) for FIFO ---
    w_edges = jnp.clip(new_next_edge, 0)
    sort_key_w = jnp.where(
        is_waiting,
        w_edges * 1_000_000 + new_wakeup.astype(jnp.int32),
        jnp.int32(2_000_000_000),
    )
    sort_idx_w = jnp.argsort(sort_key_w, stable=True)

    edges_sorted_w = new_next_edge[sort_idx_w]
    waiting_sorted = is_waiting[sort_idx_w]

    # Per-link ranks
    positions_w = jnp.arange(A)
    safe_edges_w = jnp.clip(edges_sorted_w, 0)
    boundary_w = jnp.where(
        (positions_w > 0) & (safe_edges_w != jnp.roll(safe_edges_w, 1)) & waiting_sorted,
        positions_w,
        0,
    )
    group_starts_w = jax.lax.associative_scan(jnp.maximum, boundary_w)
    ranks_w = positions_w - group_starts_w

    # Flow capacity check
    flow_limits_w = jnp.ceil(jnp.clip(
        state.edge_capacity_accumulator[safe_edges_w] - 1e-6, min=0.0
    ))
    flow_pass = waiting_sorted & (ranks_w < flow_limits_w)

    # Unsort
    inv_sort_w = jnp.argsort(sort_idx_w)
    enter_mask = flow_pass[inv_sort_w]

    # Consume flow capacity
    cap_delta_w = jnp.where(enter_mask, -1.0, 0.0)
    new_cap_acc = state.edge_capacity_accumulator.at[w_edges].add(cap_delta_w)

    # Update state: enter capacity buffer (status -> 2)
    new_status = jnp.where(enter_mask, jnp.uint8(2), new_status)
    new_current_edge_final = jnp.where(enter_mask, new_next_edge, state.current_edge)
    # Apply exiter/cont current_edge changes too (they didn't change current_edge)
    # current_edge stays as-is for exiters/cont
    new_stuck_since = jnp.where(enter_mask, step, state.stuck_since)
    new_buffer_counts = state.buffer_counts.at[w_edges].add(
        jnp.where(enter_mask, 1.0, 0.0)
    )

    # Next edge for entering agents
    new_next_edge_final = jnp.where(enter_mask, -1, new_next_edge)
    # Path mode: look up next path element
    enter_next_ptr = new_path_ptr + 1
    safe_enter_ptr = jnp.clip(enter_next_ptr, 0, jnp.maximum(max_path_len - 1, 0))
    path_next_enter = params.paths[agent_ids, safe_enter_ptr]
    valid_enter = (enter_next_ptr < max_path_len) & (max_path_len > 0)
    new_next_edge_final = jnp.where(enter_mask & valid_enter, path_next_enter, new_next_edge_final)

    # --- Trajectory tracking for entering agents (eval) ---
    new_traj_edges = state.trajectory_edges
    new_traj_enter = state.trajectory_enter_times
    new_traj_exit = state.trajectory_exit_times
    if is_eval:
        ptr_enter = new_path_ptr
        new_traj_edges = new_traj_edges.at[agent_ids, ptr_enter].set(
            jnp.where(enter_mask, new_next_edge, new_traj_edges[agent_ids, ptr_enter])
        )
        new_traj_enter = new_traj_enter.at[agent_ids, ptr_enter].set(
            jnp.where(enter_mask, step, new_traj_enter[agent_ids, ptr_enter])
        )

    return state.replace(
        status=new_status,
        current_edge=new_current_edge_final,
        next_edge=new_next_edge_final,
        path_ptr=new_path_ptr,
        wakeup_time=new_wakeup,
        current_leg=new_current_leg,
        departure_emitted=new_dep_emitted,
        stuck_since=new_stuck_since,
        edge_capacity_accumulator=new_cap_acc,
        buffer_counts=new_buffer_counts,
        leg_metrics=new_leg_metrics,
        leg_departure_times=new_leg_dep_times,
        trajectory_edges=new_traj_edges,
        trajectory_enter_times=new_traj_enter,
        trajectory_exit_times=new_traj_exit,
    )


# =============================================================================
# Step (main entry point, JIT-compiled)
# =============================================================================

@functools.partial(jax.jit, static_argnames=['is_eval'])
def jit_step(state: SimState, params: SimParams, is_eval: bool = False) -> SimState:
    """One simulation tick. Compiled separately for is_eval=True/False."""
    # A. Process Nodes (Capacity -> Downstream Spatial)
    state = _process_nodes_A(state, params, is_eval)

    # Global flow accumulation once per step
    state = _update_all_flow_accumulation(state, params)

    # B. Process Links (Spatial -> Capacity) + Handle Arrivals
    state = _process_links_B(state, params, is_eval)

    # C. Schedule Demand (New agents enter network)
    state = _schedule_demand_C(state, params, is_eval)

    # Increment clock
    state = state.replace(current_step=state.current_step + 1)
    return state


# =============================================================================
# Post-Processing (CPU, outside JIT)
# =============================================================================

def reconstruct_events(state: SimState, params: SimParams, dt: float = 1.0):
    """
    Reconstruct MATSim-style event list from dense trajectory matrices.
    Called AFTER the simulation loop, on CPU.

    Returns:
        List of (time, event_type, agent_id, edge_id) tuples, sorted by time.
    """
    traj_edges = np.asarray(state.trajectory_edges)      # [A, max_path_len]
    traj_enter = np.asarray(state.trajectory_enter_times) # [A, max_path_len]
    traj_exit = np.asarray(state.trajectory_exit_times)   # [A, max_path_len]
    departure_times = np.asarray(params.departure_times)

    events = []
    num_agents = traj_edges.shape[0]

    for a in range(num_agents):
        dep_time = int(departure_times[a])
        first_valid = -1

        for p in range(traj_edges.shape[1]):
            edge = int(traj_edges[a, p])
            if edge == -1:
                break

            enter_t = int(traj_enter[a, p])
            exit_t = int(traj_exit[a, p])

            if first_valid == -1:
                first_valid = p
                # actend + departure (at departure time)
                events.append((dep_time, EVT_ACTEND, a, edge))
                events.append((dep_time, EVT_DEPARTURE, a, edge))
                # enters_traffic
                events.append((enter_t, EVT_ENTERS_TRAFFIC, a, edge))

            if p > first_valid:
                # entered_link
                events.append((enter_t, EVT_ENTERED_LINK, a, edge))

            if exit_t >= 0:
                if p > first_valid:
                    # left_link on previous edge
                    prev_edge = int(traj_edges[a, p - 1]) if p > 0 else edge
                    events.append((exit_t, EVT_LEFT_LINK, a, prev_edge))

                # Check if this is the last valid edge
                next_p = p + 1
                is_last = (next_p >= traj_edges.shape[1]) or (traj_edges[a, next_p] == -1)
                if is_last:
                    events.append((exit_t, EVT_LEAVES_TRAFFIC, a, edge))
                    events.append((exit_t, EVT_ARRIVAL, a, edge))
                    events.append((exit_t, EVT_ACTSTART, a, edge))

    # Sort by (time, event_type)
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def get_metrics(state: SimState):
    """Return dict with arrived_count, en_route_count, avg_travel_time, avg_travel_dist."""
    status = np.asarray(state.status)
    leg_m = np.asarray(state.leg_metrics)

    done_mask = (status == 3)
    en_route_mask = (status == 1) | (status == 2)

    arrived_count = int(done_mask.sum())
    en_route_count = int(en_route_mask.sum())

    flat = leg_m.reshape(-1, 2)
    valid = flat[:, 1] > 0

    avg_time = float(flat[valid, 1].mean()) if valid.sum() > 0 else 0.0
    avg_dist = float(flat[valid, 0].mean()) if valid.sum() > 0 else 0.0

    return {
        "arrived_count": arrived_count,
        "en_route_count": en_route_count,
        "avg_travel_time": avg_time,
        "avg_travel_dist": avg_dist,
    }


# =============================================================================
# JaxDNL Wrapper Class
# =============================================================================

class JaxDNL:
    """
    Convenience wrapper around the functional JAX DNL simulator.

    Holds params (immutable) and state (replaced each step).
    Provides an OOP-like interface similar to TorchDNLMATSim for easy integration.
    """

    def __init__(self, params: SimParams, seed: int = 0):
        self.params = params
        self.state = init_state(params, seed=seed)
        self._seed = seed

    def reset(self):
        self.state = init_state(self.params, seed=self._seed)

    def step(self, is_eval: bool = False):
        self.state = jit_step(self.state, self.params, is_eval=is_eval)

    @property
    def current_step(self):
        return int(self.state.current_step)

    @current_step.setter
    def current_step(self, value):
        self.state = self.state.replace(current_step=jnp.int32(value))

    def get_metrics(self):
        return get_metrics(self.state)

    def get_events(self):
        return reconstruct_events(self.state, self.params, dt=float(self.params.dt))

    def get_snapshot(self):
        return self.state.edge_occupancy

    def get_dynamic_link_travel_times(self):
        """Returns [num_intervals_active, num_edges] with average travel times."""
        step = int(self.state.current_step)
        dt = float(self.params.dt)
        interval = float(self.params.link_tt_interval)
        max_active = int((step * dt) // interval)
        max_active = min(max_active, int(self.params.max_intervals) - 1)

        sum_tt = np.asarray(self.state.interval_tt_sum[:max_active + 1])
        count_tt = np.asarray(self.state.interval_tt_count[:max_active + 1])

        avg_tt = np.where(count_tt > 0, sum_tt / np.maximum(count_tt, 1.0), 0.0)

        # Fallback to free-flow where no data
        ff_seconds = np.asarray(self.params.ff_travel_time_steps).astype(np.float32) * dt
        ff_expanded = np.broadcast_to(ff_seconds, avg_tt.shape)
        avg_tt = np.where(count_tt > 0, avg_tt, ff_expanded)

        return avg_tt
