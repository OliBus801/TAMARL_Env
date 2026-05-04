"""Cost tracking and analytics for the HAB (Hierarchical Advisory Behavior) agent.

Provides post-episode metrics used by HAB's three sub-agents:
  - Per-OD mean travel costs and cost variance (μ_i) for the Main Agent's α.
  - Slack reward decomposition (Δc_base + ϖ·Δc_slack) for the Main Agent's reward.
  - Inefficient-path detection (ζ_p, Equation 25) for the Size Advisor.

All calculations use the experienced dynamic link travel times obtained from
``dnl.get_dynamic_link_travel_times()`` rather than free-flow estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tamarl.core.dnl_matsim import TorchDNLMATSim


@dataclass
class ODMetrics:
    """Post-episode metrics for a single OD pair."""
    mean_cost: float = 0.0
    cost_variance: float = 0.0           # μ_i — cost variance weighted by flow
    mean_travel_time: float = 0.0        # t̄_i^k — mean TT in seconds
    path_costs: Dict[int, float] = field(default_factory=dict)  # path_idx → cost
    n_agents: int = 0


class CostTracker:
    """Computes post-episode analytics required by the HAB agent hierarchy.

    This tracker is stateless across episodes: call :meth:`get_od_metrics`
    at the end of each episode to compute fresh metrics from the DNL state.

    Args:
        od_pairs: [N, 2] int32 numpy array — (origin_node, dest_node) per agent.
        paths_per_od: dict mapping (o, d) → list of edge-index paths.
        edge_endpoints: [E, 2] numpy array of (from_node, to_node).
        dt: simulation timestep in seconds.
    """

    def __init__(
        self,
        od_pairs: np.ndarray,
        paths_per_od: Dict[Tuple[int, int], List[List[int]]],
        edge_endpoints: np.ndarray,
        dt: float = 1.0,
    ):
        self._od_pairs = od_pairs.copy()
        self._paths_per_od = paths_per_od
        self._edge_endpoints = edge_endpoints
        self._dt = dt

        # Pre-compute unique OD keys and agent-to-OD mapping
        self._agent_od_keys: List[Tuple[int, int]] = [
            (int(od_pairs[i, 0]), int(od_pairs[i, 1]))
            for i in range(od_pairs.shape[0])
        ]
        self._unique_ods = list(
            {k for k in self._agent_od_keys}
        )

        # Previous episode's per-OD mean cost (for Δc̄_improve)
        self._prev_od_mean_cost: Dict[Tuple[int, int], float] = {}

    # ──────────────────────────────────────────────────────────────────
    #  Main entry point: compute all metrics for one episode
    # ──────────────────────────────────────────────────────────────────

    def get_od_metrics(
        self,
        dnl: TorchDNLMATSim,
    ) -> Dict[Tuple[int, int], ODMetrics]:
        """Compute per-OD metrics from the completed episode.

        Reads ``dnl.leg_metrics`` for agent-level travel times and aggregates
        them by OD pair.

        Returns:
            Dict mapping (origin, dest) → ODMetrics.
        """
        leg_metrics_np = dnl.leg_metrics.cpu().numpy()  # [N, MaxLegs, 2]
        status_np = dnl.status.cpu().numpy()
        num_legs_np = dnl.num_legs.cpu().numpy()

        # Collect per-agent total travel time (across all legs)
        n_agents = dnl.num_agents
        agent_total_tt = np.zeros(n_agents, dtype=np.float64)

        for i in range(n_agents):
            tt_sum = 0.0
            for leg in range(int(num_legs_np[i])):
                tt = leg_metrics_np[i, leg, 1]
                if tt > 0:
                    tt_sum += tt
                elif status_np[i] != 3:
                    # En-route: estimate as current_step - departure
                    dep = dnl.leg_departure_times[i, leg].item()
                    tt_sum += max(0.0, dnl.current_step - dep)
            agent_total_tt[i] = tt_sum * self._dt

        # Group by OD pair
        od_agents: Dict[Tuple[int, int], List[int]] = {}
        for i in range(n_agents):
            key = self._agent_od_keys[i]
            od_agents.setdefault(key, []).append(i)

        results: Dict[Tuple[int, int], ODMetrics] = {}
        for od_key, agents in od_agents.items():
            tts = agent_total_tt[agents]
            n = len(agents)
            mean_cost = float(tts.mean()) if n > 0 else 0.0

            # Cost variance μ_i = Σ f_b (c_b - c̄_i)²
            # f_b = 1/n (uniform flow weight)
            if n > 1:
                variance = float(np.mean((tts - mean_cost) ** 2))
            else:
                variance = 0.0

            results[od_key] = ODMetrics(
                mean_cost=mean_cost,
                cost_variance=variance,
                mean_travel_time=mean_cost,
                n_agents=n,
            )

        return results

    # ──────────────────────────────────────────────────────────────────
    #  Advisor reward: per-OD cost improvement
    # ──────────────────────────────────────────────────────────────────

    def compute_cost_improvement(
        self,
        current_metrics: Dict[Tuple[int, int], ODMetrics],
    ) -> Dict[Tuple[int, int], float]:
        """Compute Δc̄_improve = c̄_b^{k-1} - c̄_b^k for each OD pair.

        A positive value means the cost decreased (improvement).
        """
        improvements: Dict[Tuple[int, int], float] = {}
        for od_key, metrics in current_metrics.items():
            prev_cost = self._prev_od_mean_cost.get(od_key, metrics.mean_cost)
            improvements[od_key] = prev_cost - metrics.mean_cost
        return improvements

    def store_episode_costs(
        self,
        current_metrics: Dict[Tuple[int, int], ODMetrics],
    ):
        """Store current episode costs as baseline for next episode's Δc̄."""
        self._prev_od_mean_cost = {
            od_key: m.mean_cost for od_key, m in current_metrics.items()
        }

    # ──────────────────────────────────────────────────────────────────
    #  Main Agent: slack reward decomposition
    # ──────────────────────────────────────────────────────────────────

    def calculate_slack_reward(
        self,
        dnl: TorchDNLMATSim,
        chosen_path_indices: np.ndarray,
        prev_path_costs: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None,
        slack_weight: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the base and slack reward components for each agent.

        Args:
            dnl: the DNL after the episode.
            chosen_path_indices: [N] array of the path index each agent chose.
            prev_path_costs: previous episode's per-OD per-path costs (optional).
            slack_weight: ϖ — weight for the slack (external) component.

        Returns:
            delta_base: [N] improvement on the chosen path.
            delta_slack: [N] weighted improvement on non-chosen paths.
        """
        tt_matrix = dnl.get_dynamic_link_travel_times()
        if tt_matrix is None:
            n = dnl.num_agents
            return np.zeros(n), np.zeros(n)

        tt_np = tt_matrix.cpu().numpy()  # [intervals, E]

        n_agents = dnl.num_agents
        delta_base = np.zeros(n_agents, dtype=np.float64)
        delta_slack = np.zeros(n_agents, dtype=np.float64)

        # Compute current path costs using interval-averaged TT
        # Use interval 0 for simplicity (single-interval approximation)
        avg_tt = tt_np.mean(axis=0)  # [E] average across intervals

        for i in range(n_agents):
            od_key = self._agent_od_keys[i]
            paths = self._paths_per_od.get(od_key, [])
            if not paths:
                continue

            chosen_idx = int(chosen_path_indices[i])

            # Cost of each path
            path_costs = []
            for p_idx, path_edges in enumerate(paths):
                cost = sum(avg_tt[e] for e in path_edges) if path_edges else 0.0
                path_costs.append(cost)

            if chosen_idx >= len(path_costs):
                continue

            # Base: improvement on the chosen path
            if prev_path_costs and od_key in prev_path_costs:
                prev_chosen = prev_path_costs[od_key].get(chosen_idx, path_costs[chosen_idx])
                delta_base[i] = prev_chosen - path_costs[chosen_idx]
            else:
                delta_base[i] = 0.0

            # Slack: improvement on non-chosen paths (external utility)
            other_costs = [c for j, c in enumerate(path_costs) if j != chosen_idx]
            if other_costs and prev_path_costs and od_key in prev_path_costs:
                prev_others = [
                    prev_path_costs[od_key].get(j, path_costs[j])
                    for j in range(len(path_costs)) if j != chosen_idx
                ]
                avg_improvement = np.mean(
                    [p - c for p, c in zip(prev_others, other_costs)]
                )
                delta_slack[i] = slack_weight * avg_improvement
            else:
                delta_slack[i] = 0.0

        return delta_base, delta_slack

    # ──────────────────────────────────────────────────────────────────
    #  Size Advisor: inefficient path detection (Equation 25)
    # ──────────────────────────────────────────────────────────────────

    def detect_inefficient_paths(
        self,
        dnl: TorchDNLMATSim,
        margin: float = 0.2,
    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Identify inefficient paths per OD pair using experienced travel times.

        A path is considered inefficient (ζ_p = 1) if its cost exceeds
        ``min_cost * (1 + margin)``.

        Uses ``dnl.get_dynamic_link_travel_times()`` to compute path costs.

        Args:
            dnl: the DNL after the episode.
            margin: relative margin above min cost to classify as inefficient.

        Returns:
            Dict mapping (o, d) → (n_valid_paths, n_total_paths).
            n_valid = n_total - Σζ_p.
        """
        tt_matrix = dnl.get_dynamic_link_travel_times()
        if tt_matrix is None:
            # Fallback: all paths are valid
            return {
                od: (len(self._paths_per_od.get(od, [])),
                     len(self._paths_per_od.get(od, [])))
                for od in self._unique_ods
            }

        tt_np = tt_matrix.cpu().numpy()
        avg_tt = tt_np.mean(axis=0)  # [E]

        results: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for od_key in self._unique_ods:
            paths = self._paths_per_od.get(od_key, [])
            if not paths:
                results[od_key] = (0, 0)
                continue

            # Compute cost for each path
            costs = []
            for path_edges in paths:
                cost = sum(avg_tt[e] for e in path_edges) if path_edges else float('inf')
                costs.append(cost)

            min_cost = min(costs)
            threshold = min_cost * (1.0 + margin)

            n_valid = sum(1 for c in costs if c <= threshold)
            n_total = len(paths)
            results[od_key] = (n_valid, n_total)

        return results

    def get_current_path_costs(
        self,
        dnl: TorchDNLMATSim,
    ) -> Dict[Tuple[int, int], Dict[int, float]]:
        """Compute the cost of each path for each OD pair.

        Uses interval-averaged travel times from the DNL.

        Returns:
            Dict mapping (o, d) → {path_idx: cost}.
        """
        tt_matrix = dnl.get_dynamic_link_travel_times()
        if tt_matrix is None:
            return {}

        tt_np = tt_matrix.cpu().numpy()
        avg_tt = tt_np.mean(axis=0)

        results: Dict[Tuple[int, int], Dict[int, float]] = {}
        for od_key in self._unique_ods:
            paths = self._paths_per_od.get(od_key, [])
            if not paths:
                continue
            costs = {}
            for p_idx, path_edges in enumerate(paths):
                costs[p_idx] = float(
                    sum(avg_tt[e] for e in path_edges) if path_edges else 0.0
                )
            results[od_key] = costs
        return results
