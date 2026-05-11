"""Time-Dependent Path Evaluator.

Reconstructs experienced link travel times from the data collected during
a completed TorchDNLMATSim simulation (via ``collect_link_tt=True``) and
uses them to evaluate the Top-K candidate paths for every OD pair in an
O(A × K × MaxPathLen) vectorised PyTorch forward pass.

No modification to dnl_matsim.py is required.  The only pre-condition is
that the DNL was run with ``collect_link_tt=True`` so that
``dnl.interval_tt_sum`` and ``dnl.interval_tt_count`` are populated.

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

def compute_avg_travel_times(
    interval_tt_sum: torch.Tensor,   # [NumBins, NumEdges]
    interval_tt_count: torch.Tensor, # [NumBins, NumEdges]
    ff_travel_times_s: torch.Tensor, # [NumEdges]  free-flow TT in seconds
) -> torch.Tensor:
    """Consolidate raw bin accumulators into average link travel times.

    Bins with zero observations fall back to the free-flow travel time
    (MATSim behaviour: ``CongesteTravelTimeCalculator`` fallback).

    Args:
        interval_tt_sum:    Accumulated sum of travel times per [bin, edge].
        interval_tt_count:  Number of observations per [bin, edge].
        ff_travel_times_s:  Free-flow travel time in *seconds* per edge.

    Returns:
        avg_tt: [NumBins, NumEdges] – average experienced travel time (s).
    """
    avg_tt = interval_tt_sum / interval_tt_count.clamp(min=1.0)

    # Fill empty bins with free-flow travel time
    empty = (interval_tt_count == 0)
    # Broadcast ff over bins
    ff_expanded = ff_travel_times_s.unsqueeze(0).expand_as(avg_tt)
    avg_tt[empty] = ff_expanded[empty]

    return avg_tt  # [NumBins, NumEdges]


def evaluate_paths_time_dependent(
    avg_tt: torch.Tensor,           # [NumBins, NumEdges]
    routes: torch.Tensor,           # [Batch, K, MaxRouteLen]  edge indices, -1 = padding
    first_edges: torch.Tensor,      # [Batch]  first-edge index per leg
    departure_times_s: torch.Tensor,# [Batch]  departure time in seconds
    bin_size_s: float,              # time-bin width in seconds
) -> torch.Tensor:
    """Evaluate K candidate routes for each element of a batch in O(Batch×K×L).

    The function propagates a virtual traveller through each path from the
    agent's departure time, looking up the *experienced* average travel time
    from the bin that corresponds to the simulated entry time on each link.

    Args:
        avg_tt:             [NumBins, NumEdges] – output of ``compute_avg_travel_times``.
        routes:             [Batch, K, MaxRouteLen] – edge IDs, -1 = padding sentinel.
        first_edges:        [Batch] – the mandatory first edge (from the scenario).
        departure_times_s:  [Batch] – departure time in seconds.
        bin_size_s:         Width of each time-bin in seconds.

    Returns:
        path_costs: [Batch, K] – total travel time (s) for each (agent, path).
    """
    num_bins = avg_tt.shape[0]
    device = avg_tt.device

    Batch, K, MaxLen = routes.shape
    assert departure_times_s.shape[0] == Batch, (
        f"departure_times_s must have shape [{Batch}] (one per row in routes), "
        f"got {list(departure_times_s.shape)}"
    )

    # Current simulated clock for every (batch, k) pair – starts at departure
    # Shape: [Batch, K]
    current_time = departure_times_s.unsqueeze(1).expand(Batch, K).float().clone()

    # ── First edge (mandatory, same for all K paths of a leg) ────────────────
    # Lookup its TT at departure-time bin
    dep_bins = (departure_times_s.float() / bin_size_s).long().clamp(max=num_bins - 1)  # [Batch]
    fe_tt = avg_tt[dep_bins, first_edges.clamp(min=0)]                                  # [Batch]
    current_time += fe_tt.unsqueeze(1)  # advance for all K paths equally

    # ── Route edges (vary per K path) ────────────────────────────────────────
    for step in range(MaxLen):
        edges = routes[:, :, step]  # [Batch, K]

        valid_mask = (edges >= 0)   # [Batch, K]
        if not valid_mask.any():
            break

        # Current bin index per (agent, path)
        cur_bins = (current_time / bin_size_s).long().clamp(max=num_bins - 1)  # [Batch, K]

        # Flatten for advanced indexing, then scatter back
        flat_bins  = cur_bins[valid_mask]           # [M]
        flat_edges = edges[valid_mask].long()       # [M]  safe: padding filtered by mask

        step_tt = torch.zeros(Batch, K, device=device)
        step_tt[valid_mask] = avg_tt[flat_bins, flat_edges]

        current_time = current_time + step_tt

    # Total travel time = final time − departure time
    path_costs = current_time - departure_times_s.unsqueeze(1).float()  # [Batch, K]
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
            dnl:            A *completed* TorchDNLMATSim instance with
                            ``collect_link_tt=True``.
            departure_times:[A] departure times in simulation **steps**
                            (will be converted to seconds using ``dnl.dt``).
            od_indices:     Optional override for the leg→OD mapping.
                            Defaults to ``self.od_indices``.

        Returns:
            path_costs: [TotalLegs, K] – TD travel time (s) per path.
            best_k:     [TotalLegs]    – index of the cheapest path per leg.

        Raises:
            RuntimeError: If the DNL was not run with ``collect_link_tt=True``.
        """
        if not dnl.collect_link_tt:
            raise RuntimeError(
                "DNL must be run with collect_link_tt=True to use "
                "TimeDependentEvaluator."
            )

        od_idx = od_indices if od_indices is not None else self.od_indices

        # ── Build avg_tt matrix from the simulation's accumulators ────
        num_intervals_used = int((dnl.current_step * dnl.dt) // self.link_tt_interval) + 1
        num_intervals_used = min(num_intervals_used, dnl.interval_tt_sum.shape[0])

        tt_sum   = dnl.interval_tt_sum[:num_intervals_used]    # [UsedBins, E]
        tt_count = dnl.interval_tt_count[:num_intervals_used]  # [UsedBins, E]

        # Free-flow TT in seconds
        ff_tt_s = (dnl.ff_travel_time_steps.float() * dnl.dt).to(self.device)  # [E]

        avg_tt = compute_avg_travel_times(tt_sum, tt_count, ff_tt_s)            # [UsedBins, E]

        # ── Per-leg routes and departure times ────────────────────────
        # routes: [TotalLegs, K, MaxRouteLen]
        routes = self.candidate_routes[od_idx]  # [TotalLegs, K, MaxRouteLen]

        # Map agent departure times (steps → seconds) to [TotalLegs].
        # departure_times is [A]; we expand it to [TotalLegs] using
        # the leg→agent mapping stored in self._leg_agent_map.
        # If TotalLegs == A (all single-leg agents), this is a no-op.
        dep_all_s = departure_times.float() * dnl.dt  # [A] seconds
        if dep_all_s.shape[0] == routes.shape[0]:
            # Single-leg scenario shortcut
            dep_s = dep_all_s.to(self.device)
        else:
            # Multi-leg: expand using the stored leg→agent index map
            dep_s = dep_all_s[self._leg_agent_map].to(self.device)  # [TotalLegs]

        fe = self.first_edges.to(self.device)     # [TotalLegs]

        # ── Evaluate ──────────────────────────────────────────────────
        path_costs = evaluate_paths_time_dependent(
            avg_tt=avg_tt,
            routes=routes,
            first_edges=fe,
            departure_times_s=dep_s,
            bin_size_s=self.link_tt_interval,
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
