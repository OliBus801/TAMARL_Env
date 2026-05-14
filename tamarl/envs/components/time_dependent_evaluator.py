"""Time-Dependent Path Evaluator.

Reconstructs experienced link travel times from the data collected during
a completed TorchDNLMATSim simulation and uses them to evaluate the Top-K 
candidate paths for every OD pair in an O(A × K × MaxPathLen) vectorised
PyTorch forward pass.

Typical usage (inside AgentLevelWrapper.step or a post-episode callback):

    from tamarl.envs.components.time_dependent_evaluator import TimeDependentEvaluator

    # Build once, reuse across episodes (same env / same candidate routes)
    evaluator = TimeDependentEvaluator(
        candidate_routes=env.candidate_routes,   # [NumOD, K, MaxRouteLen]
        first_edges=env.first_edges_all_legs,    # [TotalLegs]
        od_indices=env.od_indices_all_legs,      # [TotalLegs]
        link_tt_interval=env.bandit.link_tt_interval,
        device=env._device,
    )

    # After every simulation, evaluate all K paths for each leg:
    path_costs, best_k = evaluator.evaluate(
        dnl=env.bandit.dnl,
        departure_times=env.bandit.scenario.departure_times,
        od_indices=env.od_indices_all_legs,
    )
    # path_costs : [TotalLegs, K]  – TD travel time (seconds) for each path
    # best_k     : [TotalLegs]     – index of the cheapest path
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers (no state)
# ──────────────────────────────────────────────────────────────────────────────

def compute_n_curves(
    events_tensor: torch.Tensor,  # [NumEvents, 4] -> [step, type, agent, edge]
    num_steps: int,
    num_edges: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Sparse N-Curves A(t) and D(t) from simulation events.
    
    Args:
        events_tensor: Raw event buffer [N, 4].
        num_steps: Total steps in the simulation.
        num_edges: Number of edges in the network.
        
    Returns:
        A_shifted_sorted: 1D tensor of arrivals shifted by edge*num_steps.
        A_offsets: 1D tensor of size [num_edges + 1] with offsets.
        D_times_sorted: 1D tensor of departure times.
        D_offsets: 1D tensor of size [num_edges + 1] with offsets.
    """
    if events_tensor.numel() == 0:
        empty_A = torch.empty(0, device=device, dtype=torch.long)
        empty_D = torch.empty(0, device=device, dtype=torch.long)
        zeros_off = torch.zeros(num_edges + 1, device=device, dtype=torch.long)
        return empty_A, zeros_off, empty_D, zeros_off.clone()

    times = events_tensor[:, 0].long().clamp(max=num_steps - 1)
    event_types = events_tensor[:, 1].long()
    edges = events_tensor[:, 3].long().clamp(min=0, max=num_edges - 1)
    
    # Arrivals: EVT_DEPARTURE (1), EVT_ENTERED_LINK (4)
    mask_A = (event_types == 1) | (event_types == 4)
    A_times = times[mask_A]
    A_edges = edges[mask_A]
    
    # Departures: EVT_LEFT_LINK (3), EVT_LEAVES_TRAFFIC (5), EVT_STUCKANDABORT (8)
    mask_D = (event_types == 3) | (event_types == 5) | (event_types == 8)
    D_times = times[mask_D]
    D_edges = edges[mask_D]
    
    # Sort arrivals by edge then time
    A_shifted = A_edges * num_steps + A_times
    A_shifted_sorted, A_sort_idx = torch.sort(A_shifted)
    
    A_counts = torch.bincount(A_edges[A_sort_idx], minlength=num_edges)
    A_offsets = torch.zeros(num_edges + 1, device=device, dtype=torch.long)
    A_offsets[1:] = torch.cumsum(A_counts, dim=0)
    
    # Sort departures by edge then time
    D_shifted = D_edges * num_steps + D_times
    D_shifted_sorted, D_sort_idx = torch.sort(D_shifted)
    D_times_sorted = D_times[D_sort_idx]
    
    D_counts = torch.bincount(D_edges[D_sort_idx], minlength=num_edges)
    D_offsets = torch.zeros(num_edges + 1, device=device, dtype=torch.long)
    D_offsets[1:] = torch.cumsum(D_counts, dim=0)
    
    return A_shifted_sorted, A_offsets, D_times_sorted, D_offsets


def evaluate_paths_time_dependent(
    A_shifted_sorted: torch.Tensor,
    A_offsets: torch.Tensor,
    D_times_sorted: torch.Tensor,
    D_offsets: torch.Tensor,
    routes: torch.Tensor,           # [Batch, K, MaxRouteLen] edge indices, -1 = padding
    first_edges: torch.Tensor,      # [Batch] first-edge index per leg
    departure_times_step: torch.Tensor,# [Batch] departure time in steps
    dt: float,                      # simulation timestep in seconds
    num_edges: int,
    num_steps: int,
    ff_travel_time_steps: torch.Tensor, # [NumEdges] free-flow travel time
) -> torch.Tensor:
    """Evaluate K candidate routes for each element of a batch in O(Batch×K×L).

    Propagates a virtual traveller through each path using Newell's N-Curves.

    Args:
        A_shifted_sorted, A_offsets, D_times_sorted, D_offsets: Sparse N-Curves.
        routes:             [Batch, K, MaxRouteLen] – edge IDs, -1 = padding sentinel.
        first_edges:        [Batch] – the mandatory first edge.
        departure_times_step: [Batch] – departure time in steps.
        dt:                 Simulation time step in seconds.
        num_edges:          Total edges in the network.
        num_steps:          Total steps in the simulation.
        ff_travel_time_steps: [NumEdges] free flow travel time per edge in steps.

    Returns:
        path_costs: [Batch, K] – total travel time (s) for each (agent, path).
    """
    device = A_shifted_sorted.device
    Batch, K, MaxLen = routes.shape

    # Current simulated clock in steps for every (batch, k) pair
    # Shape: [Batch, K]
    current_step = departure_times_step.unsqueeze(1).expand(Batch, K).long().clone()
    
    # helper for processing one step on an edge
    def _traverse(edges_mask, edges_b_k, cur_steps_b_k):
        flat_mask = edges_mask.reshape(-1)
        valid_indices = torch.nonzero(flat_mask, as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return torch.zeros_like(cur_steps_b_k)
            
        flat_edges = edges_b_k.reshape(-1)[valid_indices].clamp(min=0, max=num_edges - 1)
        flat_steps = cur_steps_b_k.reshape(-1)[valid_indices].clamp(min=0, max=num_steps - 1)
        
        # Arrival number n = A[e, t]
        queries = flat_edges * num_steps + flat_steps
        global_idx = torch.searchsorted(A_shifted_sorted, queries, right=True)
        n_vals = global_idx - A_offsets[flat_edges]
        
        num_departures = D_offsets[flat_edges + 1] - D_offsets[flat_edges]
        valid_mask = (n_vals > 0) & (n_vals <= num_departures)
        
        t_exit = torch.zeros_like(flat_steps)
        # Fallback: if arriving on completely empty link, assume free-flow travel time
        t_exit[n_vals == 0] = flat_steps[n_vals == 0] + ff_travel_time_steps[flat_edges[n_vals == 0]]
        t_exit[n_vals > num_departures] = num_steps - 1
        
        if valid_mask.any():
            idx = D_offsets[flat_edges[valid_mask]] + n_vals[valid_mask] - 1
            t_exit[valid_mask] = D_times_sorted[idx]
            
        t_exit = t_exit.clamp(max=num_steps - 1)
        step_tt_steps = t_exit - flat_steps
        
        # Make sure travel time is not negative, minimum is free flow travel time
        min_tt = ff_travel_time_steps[flat_edges]
        step_tt_steps = torch.maximum(step_tt_steps, min_tt)
        
        # Scatter back to [Batch, K]
        result = torch.zeros(Batch * K, device=device, dtype=torch.long)
        result[valid_indices] = step_tt_steps
        return result.view(Batch, K)

    # ── First edge (mandatory, same for all K paths of a leg) ────────────────
    fe_expanded = first_edges.unsqueeze(1).expand(Batch, K).long()
    fe_mask = fe_expanded >= 0
    tt_first = _traverse(fe_mask, fe_expanded, current_step)
    current_step += tt_first

    # ── Route edges (vary per K path) ────────────────────────────────────────
    for step in range(MaxLen):
        edges = routes[:, :, step]  # [Batch, K]
        valid_mask = (edges >= 0)   # [Batch, K]
        if not valid_mask.any():
            break

        step_tt = _traverse(valid_mask, edges, current_step)
        current_step += step_tt

    # Total travel time = final time − departure time
    path_costs = (current_step - departure_times_step.unsqueeze(1)).float() * dt  # [Batch, K]
    return path_costs





# ──────────────────────────────────────────────────────────────────────────────
# Stateful convenience class
# ──────────────────────────────────────────────────────────────────────────────

class TimeDependentEvaluator:
    """Stateful wrapper around the two low-level helpers above.

    Build once (same candidate routes for the entire training run) and call
    ``evaluate()`` after every episode.

    Attributes:
        candidate_routes: [NumOD, K, MaxRouteLen] – the wrapper's route tensor.
        first_edges:      [TotalLegs] – first edge for each leg.
        od_indices:       [TotalLegs] – OD index for each leg.
        link_tt_interval: Width of each time-bin in seconds.
        device:           Torch device.
    """

    def __init__(
        self,
        candidate_routes: torch.Tensor,  # [NumOD, K, MaxRouteLen]
        first_edges: torch.Tensor,        # [TotalLegs]
        od_indices: torch.Tensor,         # [TotalLegs]
        link_tt_interval: float = 300.0,
        device: str = "cpu",
        leg_agent_map: Optional[torch.Tensor] = None,  # [TotalLegs] → agent idx
    ):
        self.candidate_routes = candidate_routes.to(device)
        self.first_edges = first_edges.to(device)
        self.od_indices = od_indices.to(device)
        self.link_tt_interval = link_tt_interval
        self.device = device
        # [TotalLegs] mapping each leg to its parent agent index.
        # None means TotalLegs == A (single-leg shortcut).
        self._leg_agent_map = leg_agent_map.to(device) if leg_agent_map is not None else None

    # ------------------------------------------------------------------
    def evaluate(
        self,
        dnl,  # TorchDNLMATSim instance (after a completed simulation)
        departure_times: torch.Tensor,  # [A] scenario departure times (steps)
        od_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the time-dependent path evaluation.

        Args:
            dnl:            A *completed* TorchDNLMATSim instance.
            departure_times:[A] departure times in simulation **steps**
                            (will be converted to seconds using ``dnl.dt``).
            od_indices:     Optional override for the leg→OD mapping.
                            Defaults to ``self.od_indices``.

        Returns:
            path_costs: [TotalLegs, K] – TD travel time (s) per path.
            best_k:     [TotalLegs]    – index of the cheapest path per leg.
        """

        od_idx = od_indices if od_indices is not None else self.od_indices

        # ── Build A(t) and D(t) N-Curves from events ────
        if not dnl.track_events:
            raise ValueError("TimeDependentEvaluator requires dnl.track_events=True to build N-Curves.")
            
        dnl._flush_events()
        if len(dnl._cpu_events_blocks) > 0:
            events_tensor = torch.cat(dnl._cpu_events_blocks, dim=0).to(self.device)
        else:
            events_tensor = torch.empty((0, 4), device=self.device, dtype=torch.long)
            
        # Add 1 to current_step to ensure enough capacity (steps are 0-indexed)
        num_steps = max(dnl.current_step + 1, 1)
        A_shifted_sorted, A_offsets, D_times_sorted, D_offsets = compute_n_curves(
            events_tensor, num_steps, dnl.num_edges, self.device)

        # ── Per-leg routes and departure times ────────────────────────
        # routes: [TotalLegs, K, MaxRouteLen]
        routes = self.candidate_routes[od_idx]  # [TotalLegs, K, MaxRouteLen]

        # Map agent departure times (steps) to [TotalLegs].
        dep_all = departure_times.long()  # [A] steps
        if dep_all.shape[0] == routes.shape[0]:
            # Single-leg scenario shortcut
            dep = dep_all.to(self.device)
        else:
            # Multi-leg: expand using the stored leg→agent index map
            dep = dep_all[self._leg_agent_map].to(self.device)  # [TotalLegs]

        fe = self.first_edges.to(self.device)     # [TotalLegs]

        # ── Evaluate ──────────────────────────────────────────────────
        path_costs = evaluate_paths_time_dependent(
            A_shifted_sorted=A_shifted_sorted,
            A_offsets=A_offsets,
            D_times_sorted=D_times_sorted,
            D_offsets=D_offsets,
            routes=routes,
            first_edges=fe,
            departure_times_step=dep,
            dt=dnl.dt,
            num_edges=dnl.num_edges,
            num_steps=num_steps,
            ff_travel_time_steps=dnl.ff_travel_time_steps,
        )  # [TotalLegs, K]

        # Mask invalid paths (will have cost = 0 from zero first-edge edge index)
        # We do not have the action_masks here, so just return raw costs.
        # The caller can apply masks if needed.
        best_k = path_costs.argmin(dim=1)  # [TotalLegs]

        return path_costs, best_k

    # ------------------------------------------------------------------
    @classmethod
    def from_wrapper(cls, env) -> "TimeDependentEvaluator":
        """Convenience constructor from an AgentLevelWrapper instance.

        Args:
            env: An ``AgentLevelWrapper`` instance.

        Returns:
            A ``TimeDependentEvaluator`` ready to use with ``env.bandit.dnl``.
        """
        # Build leg → agent index map from env.leg_to_agent
        # leg_to_agent is a list of (agent_idx, leg_in_agent_idx) tuples
        leg_agent_indices = torch.tensor(
            [agent_idx for agent_idx, _ in env.leg_to_agent],
            dtype=torch.long,
        )
        return cls(
            candidate_routes=env.candidate_routes,
            first_edges=env.first_edges_all_legs,
            od_indices=env.od_indices_all_legs,
            link_tt_interval=env.bandit.link_tt_interval,
            device=env._device,
            leg_agent_map=leg_agent_indices,
        )
