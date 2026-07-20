"""Time-Dependent Path Evaluator.

Reconstructs experienced link travel times from the data collected during
a completed TorchDNL simulation and uses them to evaluate the Top-K
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

from typing import Optional

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers (no state)
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_paths_time_dependent(
    dynamic_tt: torch.Tensor,  # [NumIntervals, NumEdges] average travel time in seconds
    link_tt_interval: float,  # width of interval in seconds
    routes: torch.Tensor,  # [Batch, K, MaxRouteLen] edge indices, -1 = padding
    first_edges: torch.Tensor,  # [Batch] first-edge index per leg
    departure_times_sec: torch.Tensor,  # [Batch] departure time in seconds
    num_edges: int,
) -> torch.Tensor:
    """Evaluate K candidate routes for each element of a batch in O(Batch×K×L).

    Propagates a virtual traveller through each path using interval-based link travel times.

    Args:
        dynamic_tt:         [NumIntervals, NumEdges] average travel time in seconds.
        link_tt_interval:   Width of the time bin in seconds.
        routes:             [Batch, K, MaxRouteLen] – edge IDs, -1 = padding sentinel.
        first_edges:        [Batch] – the mandatory first edge.
        departure_times_sec: [Batch] – departure time in seconds.
        num_edges:          Total edges in the network.

    Returns:
        path_costs: [Batch, K] – total travel time (s) for each (agent, path).
    """
    device = dynamic_tt.device
    Batch, K, MaxLen = routes.shape
    NumIntervals = dynamic_tt.size(0)

    # Current simulated clock in seconds for every (batch, k) pair
    current_time_sec = departure_times_sec.unsqueeze(1).expand(Batch, K).clone()

    # helper for processing one step on an edge
    def _traverse(edges_mask, edges_b_k, cur_time_b_k):
        flat_mask = edges_mask.reshape(-1)
        valid_indices = torch.nonzero(flat_mask, as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return torch.zeros_like(cur_time_b_k)

        flat_edges = edges_b_k.reshape(-1)[valid_indices].clamp(min=0, max=num_edges - 1)
        flat_times = cur_time_b_k.reshape(-1)[valid_indices]

        # Find which interval the agent arrives at the link
        interval_idx = (flat_times / link_tt_interval).long().clamp(max=NumIntervals - 1)

        step_tt_sec = dynamic_tt[interval_idx, flat_edges]

        result = torch.zeros(Batch * K, device=device, dtype=torch.float32)
        result[valid_indices] = step_tt_sec
        return result.view(Batch, K)

    # ── First edge (mandatory, same for all K paths of a leg) ────────────────
    fe_expanded = first_edges.unsqueeze(1).expand(Batch, K).long()
    fe_mask = fe_expanded >= 0
    tt_first = _traverse(fe_mask, fe_expanded, current_time_sec)
    current_time_sec += tt_first

    # ── Route edges (vary per K path) ────────────────────────────────────────
    for step in range(MaxLen):
        edges = routes[:, :, step]  # [Batch, K]
        valid_mask = edges >= 0  # [Batch, K]
        if not valid_mask.any():
            break

        step_tt = _traverse(valid_mask, edges, current_time_sec)
        current_time_sec += step_tt

    # Total travel time = final time − departure time
    path_costs = current_time_sec - departure_times_sec.unsqueeze(1)  # [Batch, K]
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
        routes_flat_csr: torch.Tensor,  # [TotalActualEdges] int32
        routes_offsets_csr: torch.Tensor,  # [NumOD * K + 1] int64
        K: int,  # number of candidate routes per OD
        first_edges: torch.Tensor,  # [TotalLegs]
        od_indices: torch.Tensor,  # [TotalLegs]
        link_tt_interval: float = 300.0,
        device: str = "cpu",
        leg_agent_map: torch.Tensor | None = None,  # [TotalLegs] → agent idx
    ):
        self.routes_flat_csr = routes_flat_csr.to(device)
        self.routes_offsets_csr = routes_offsets_csr.to(device)
        self.K = K
        self.first_edges = first_edges.to(device)
        self.od_indices = od_indices.to(device)
        self.link_tt_interval = link_tt_interval
        self.device = device
        self._leg_agent_map = leg_agent_map.to(device) if leg_agent_map is not None else None

    # ------------------------------------------------------------------
    def evaluate(
        self,
        dnl,  # TorchDNL instance (after a completed simulation)
        departure_times: torch.Tensor,  # [A] scenario departure times (steps)
        od_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the time-dependent path evaluation.

        Args:
            dnl:            A *completed* TorchDNL instance.
            departure_times:[A] departure times in simulation **steps**
                            (will be converted to seconds using ``dnl.dt``).
            od_indices:     Optional override for the leg→OD mapping.
                            Defaults to ``self.od_indices``.

        Returns:
            path_costs: [TotalLegs, K] – TD travel time (s) per path.
            best_k:     [TotalLegs]    – index of the cheapest path per leg.
        """

        od_idx = od_indices if od_indices is not None else self.od_indices

        # ── Fetch interval-based average link travel times ───────────
        if not dnl.collect_link_tt:
            raise ValueError(
                "TimeDependentEvaluator requires dnl.collect_link_tt=True to evaluate paths."
            )

        dynamic_tt = dnl.get_dynamic_link_travel_times()
        if dynamic_tt is None:
            # Fallback to free flow travel time if simulation hasn't run
            dynamic_tt = dnl.edge_static[:, 4].unsqueeze(0)

        Batch = od_idx.shape[0]
        CHUNK_SIZE = 16384
        all_path_costs = []

        # Map agent departure times (steps) to [TotalLegs].
        dep_all_steps = departure_times.to(self.device)
        if dep_all_steps.shape[0] == Batch:
            dep_steps = dep_all_steps
        else:
            dep_steps = dep_all_steps[self._leg_agent_map]
        dep_sec_all = dep_steps.float() * dnl.dt
        fe_all = self.first_edges.to(self.device)

        # ── Process in chunks to avoid OOM ────────────────────────────
        for start_idx in range(0, Batch, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, Batch)
            chunk_size = end_idx - start_idx

            od_chunk = od_idx[start_idx:end_idx]
            dep_sec_chunk = dep_sec_all[start_idx:end_idx]
            fe_chunk = fe_all[start_idx:end_idx]

            # ── Reconstruct minimal-width route tensor for this chunk ─────
            route_rows_2d = od_chunk.unsqueeze(1) * self.K + torch.arange(
                self.K, device=self.device, dtype=torch.long
            ).unsqueeze(0)
            route_rows_flat = route_rows_2d.reshape(-1)
            starts = self.routes_offsets_csr[route_rows_flat]
            ends = self.routes_offsets_csr[route_rows_flat + 1]
            lens = (ends - starts).long()
            max_len = int(lens.max().item()) if lens.numel() > 0 and lens.max() > 0 else 0

            routes_chunk = torch.full(
                (chunk_size * self.K, max(max_len, 1)), -1, device=self.device, dtype=torch.int32
            )
            if max_len > 0:
                total_e = int(lens.sum().item())
                if total_e > 0:
                    bk_of_e = torch.repeat_interleave(
                        torch.arange(chunk_size * self.K, device=self.device, dtype=torch.long),
                        lens,
                    )
                    cs_bk = torch.zeros(
                        chunk_size * self.K + 1, device=self.device, dtype=torch.long
                    )
                    cs_bk[1:] = torch.cumsum(lens, dim=0)
                    rank = (
                        torch.arange(total_e, device=self.device, dtype=torch.long) - cs_bk[bk_of_e]
                    )
                    routes_chunk[bk_of_e, rank] = self.routes_flat_csr[starts[bk_of_e] + rank]
            routes_chunk = routes_chunk.view(chunk_size, self.K, max(max_len, 1))

            # ── Evaluate chunk ────────────────────────────────────────────
            chunk_costs = evaluate_paths_time_dependent(
                dynamic_tt=dynamic_tt,
                link_tt_interval=self.link_tt_interval,
                routes=routes_chunk,
                first_edges=fe_chunk,
                departure_times_sec=dep_sec_chunk,
                num_edges=dnl.num_edges,
            )
            all_path_costs.append(chunk_costs)

            # Free intermediates
            del route_rows_2d, route_rows_flat, starts, ends, lens, routes_chunk

        # ── Concatenate and compute best_k ────────────────────────────
        path_costs = torch.cat(all_path_costs, dim=0)  # [TotalLegs, K]
        best_k = path_costs.argmin(dim=1)  # [TotalLegs]

        return path_costs, best_k

    # ------------------------------------------------------------------
    @classmethod
    def from_wrapper(cls, env) -> TimeDependentEvaluator:
        """Convenience constructor from a wrapper instance (any formulation).

        Args:
            env: An ``AgentLevelWrapper``, ``ODLevelWrapper``, or
                 ``CentralizedLevelWrapper`` instance.

        Returns:
            A ``TimeDependentEvaluator`` ready to use with ``env.bandit.dnl``.
        """
        leg_agent_indices = torch.tensor(
            [agent_idx for agent_idx, _ in env.leg_to_agent],
            dtype=torch.long,
        )
        return cls(
            routes_flat_csr=env.routes_flat_csr,
            routes_offsets_csr=env.routes_offsets_csr,
            K=env.K,
            first_edges=env.first_edges_all_legs,
            od_indices=env.od_indices_all_legs,
            link_tt_interval=env.bandit.link_tt_interval,
            device=env._device,
            leg_agent_map=leg_agent_indices,
        )
