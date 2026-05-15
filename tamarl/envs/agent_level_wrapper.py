"""AgentLevelWrapper — gymnasium.vector.VectorEnv bridge for DTABanditEnv.

Presents the DTA one-shot simulation as a vectorised Gym environment
where each sub-environment is one vehicle choosing among K candidate
routes *for leg 0*.  SB3 and other libraries see ``num_envs = A``
independent agents with ``Discrete(K)`` action spaces sharing a single
policy.

Multi-leg agents are supported: the RL action chooses the leg-0 route,
while subsequent legs automatically use the shortest path (k=0).

Flow:
    1. __init__  → loads scenario, enumerates top-k paths per OD
                   (for every unique OD across all legs),
                   builds ``candidate_routes_leg0 [Num_OD0, K, MaxRoute0]``,
                   and fixed routes for leg 1+ .
    2. reset()   → returns empty observations + info.
    3. step(actions)
         a. Assemble the full multi-leg paths tensor.
         b. Run:   ``bandit.reset(paths); rewards = bandit.step()``
         c. Return: obs, rewards, terminated (all True), truncated, info.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.components.path_enumerator import get_or_compute_top_k_paths
from tamarl.envs.components.metrics import compute_empirical_nash_metrics_tensor
from tamarl.envs.components.time_dependent_evaluator import TimeDependentEvaluator


class AgentLevelWrapper:
    """Vectorised Gym env: one leg = one sub-env choosing a route.

    This wrapper treats every leg of every vehicle as an independent sub-environment
    in the Gymnasium VectorEnv. For example, a vehicle with 2 legs will be
    represented as two distinct "agents" in the vectorized batch.

    Attributes:
        bandit: The underlying DTABanditEnv engine.
        candidate_routes: [Num_Unique_OD, K, MaxRouteLen] padded with -1.
        od_indices_all_legs: [TotalLegs] int tensor mapping each leg to its OD row.
        K: Number of candidate routes per OD pair.
        num_envs: Total number of legs across all vehicles.
    """

    metadata = {"render_modes": [], "autoreset_mode": 0}

    def __init__(
        self,
        bandit: DTABanditEnv,
        top_k: int = 3,
        feedback_type: str = "full",
    ):
        self.bandit = bandit
        self.K = top_k
        self._device = bandit._device
        self.feedback_type = feedback_type
        
        scenario = self.bandit.scenario
        A = self.bandit.num_agents
        timestep = bandit._timestep

        edge_eps = scenario.edge_endpoints.numpy()  # [E, 2]
        edge_static = scenario.edge_static.numpy()

        # ── Collect info for ALL legs of ALL agents ──────────────────
        self.leg_to_agent = []      # list of (agent_idx, leg_in_agent_idx)
        leg_origins = []
        leg_dests = []
        self.leg_first_edges = []
        
        num_legs_np = scenario.num_legs.numpy()
        fe_np = scenario.first_edges.numpy()
        dest_np = scenario.destinations.numpy()

        for i in range(A):
            for leg in range(num_legs_np[i]):
                fe = int(fe_np[i, leg])
                dest = int(dest_np[i, leg])
                if fe >= 0:
                    orig = int(edge_eps[fe, 1])
                    leg_origins.append(orig)
                    leg_dests.append(dest)
                    self.leg_first_edges.append(fe)
                    self.leg_to_agent.append((i, leg))

        self.num_envs = len(self.leg_to_agent)

        # ── GPU index tensors for vectorized reward extraction (Opt-1) ─
        # Pre-computed once; used in step() to replace the Python loop.
        self._leg_agent_idx = torch.tensor(
            [i for i, _ in self.leg_to_agent], dtype=torch.long
        )  # [TotalLegs]  (moved to device after _device is confirmed)
        self._leg_leg_idx = torch.tensor(
            [j for _, j in self.leg_to_agent], dtype=torch.long
        )  # [TotalLegs]

        # ── Enumerate top-K paths for all unique ODs ─────────────────
        unique_od, inverse_od = np.unique(
            np.stack([leg_origins, leg_dests], axis=1), 
            axis=0, return_inverse=True
        )
        self.od_indices_all_legs = torch.tensor(inverse_od, dtype=torch.long, device=self._device)
        self.first_edges_all_legs = torch.tensor(self.leg_first_edges, dtype=torch.long, device=self._device)

        # Free-flow times for path enumeration
        ff_times = torch.floor(
            scenario.edge_static[:, 4] / timestep
        ).numpy().astype(np.float64)

        paths_dict = get_or_compute_top_k_paths(
            scenario_dir=bandit._scenario_path,
            num_nodes=scenario.num_nodes,
            edge_endpoints=edge_eps,
            ff_times=ff_times,
            od_pairs=unique_od.astype(np.int32),
            k=top_k,
        )

        # ── Build candidate routes [Num_Unique_OD, K, MaxRouteLen] ───
        max_route_len = 0
        for paths_list in paths_dict.values():
            for p in paths_list:
                max_route_len = max(max_route_len, len(p))
        max_route_len = max(max_route_len, 1)

        num_unique_od = len(unique_od)
        cand_np = np.full((num_unique_od, top_k, max_route_len), -1, dtype=np.int32)
        masks_np = np.zeros((num_unique_od, top_k), dtype=bool)

        for od_idx in range(num_unique_od):
            od_key = (int(unique_od[od_idx, 0]), int(unique_od[od_idx, 1]))
            paths_list = paths_dict.get(od_key, [])
            if not paths_list:
                continue
            
            # Identify a representative first_edge for this OD to compute initial costs
            # (We find the first leg in our list that matches this OD index)
            first_leg_idx = np.where(inverse_od == od_idx)[0][0]
            fe_id = self.leg_first_edges[first_leg_idx]
            fe_ff = ff_times[fe_id]

            for k_idx in range(top_k):
                p_idx = min(k_idx, len(paths_list) - 1)
                path = paths_list[p_idx]
                for e_idx, edge_id in enumerate(path):
                    cand_np[od_idx, k_idx, e_idx] = edge_id
                
                # Action is valid if it's one of the unique paths found
                masks_np[od_idx, k_idx] = (k_idx < len(paths_list))

        self.candidate_routes = torch.tensor(
            cand_np, dtype=torch.long, device=self._device
        )  # [Num_Unique_OD, K, MaxRouteLen]
        
        self.action_masks = torch.tensor(
            masks_np, dtype=torch.bool, device=self._device
        )  # [Num_Unique_OD, K]

        # ── Compute FFTT matrix for Path-Based Empirical Regret ──────
        self.fftt_matrix = np.zeros((num_unique_od, top_k), dtype=np.float32)
        edge_static_np = scenario.edge_static.numpy()
        for od_idx in range(num_unique_od):
            for k_idx in range(top_k):
                if masks_np[od_idx, k_idx]:
                    path = cand_np[od_idx, k_idx]
                    valid_path = path[path != -1]
                    path_fftt = edge_static_np[valid_path, 4].sum()
                    self.fftt_matrix[od_idx, k_idx] = path_fftt
                else:
                    self.fftt_matrix[od_idx, k_idx] = np.inf

        # ── Compute max total path length needed for [A, MaxTotal] ───
        # For each agent: sum of (1 + max_route_len) for each leg + (n_legs-1) separators
        max_total = 0
        for i in range(A):
            n = num_legs_np[i]
            # Each leg: 1 (first edge) + max_route_len (path)
            # Separators: (n-1)
            total = n * (1 + max_route_len) + (n - 1)
            max_total = max(max_total, total)
        self._max_path_len = int(max_total)

        # ── Pre-compute scatter plan for vectorized path assembly (Opt-2) ──
        # For every (agent, leg) we record:
        #   - which agent row in `paths`  → _scatter_agent_row
        #   - the column offset for first_edge  → _scatter_fe_col
        #   - for each route edge position e: column offset → _scatter_route_cols[e]
        # All stored on CPU first, moved to device after.
        # Layout per leg inside paths[i]:
        #   [leg-sep?] [first_edge] [route_edge_0 … route_edge_{MaxRouteLen-1}]
        # where leg-sep (-2) is written for leg_j > 0.
        # We compute column offsets exactly as the old Python loop did.
        _max_rl = max_route_len  # shorthand
        scatter_agent = []   # destination row in paths  [NumSlots]
        scatter_col   = []   # destination col in paths  [NumSlots]
        scatter_src_type = []  # 0=fe, 1=route_edge, 2=separator
        scatter_route_pos = []  # position in route (for type==1 slots)
        scatter_leg_ptr   = []  # index into leg_to_agent list (for fe/route)

        leg_ptr = 0
        for i in range(A):
            n = num_legs_np[i]
            ptr = 0
            for leg_j in range(n):
                if leg_j > 0:
                    # leg separator slot
                    scatter_agent.append(i)
                    scatter_col.append(ptr)
                    scatter_src_type.append(2)   # constant -2
                    scatter_route_pos.append(0)
                    scatter_leg_ptr.append(leg_ptr)
                    ptr += 1
                # first-edge slot
                scatter_agent.append(i)
                scatter_col.append(ptr)
                scatter_src_type.append(0)       # fe index from first_edges_all_legs
                scatter_route_pos.append(0)
                scatter_leg_ptr.append(leg_ptr)
                ptr += 1
                # route-edge slots (one per position in MaxRouteLen)
                for e in range(_max_rl):
                    scatter_agent.append(i)
                    scatter_col.append(ptr)
                    scatter_src_type.append(1)   # selected_routes[leg_ptr, e]
                    scatter_route_pos.append(e)
                    scatter_leg_ptr.append(leg_ptr)
                    ptr += 1
                leg_ptr += 1

        _st = torch.tensor(scatter_src_type, dtype=torch.uint8)
        # Separate the three slot categories for efficient scatter
        _fe_mask  = (_st == 0)
        _re_mask  = (_st == 1)
        _sep_mask = (_st == 2)

        # FE slots
        self._fe_agent_col = torch.stack([
            torch.tensor([scatter_agent[k] for k in range(len(_st)) if _fe_mask[k]],  dtype=torch.long),
            torch.tensor([scatter_col[k]   for k in range(len(_st)) if _fe_mask[k]],  dtype=torch.long),
        ], dim=1)  # [NumLegs, 2]
        self._fe_leg_ptr = torch.tensor(
            [scatter_leg_ptr[k] for k in range(len(_st)) if _fe_mask[k]], dtype=torch.long
        )  # [NumLegs]

        # Route-edge slots
        _re_indices = [k for k in range(len(_st)) if _re_mask[k]]
        self._re_agent_col = torch.stack([
            torch.tensor([scatter_agent[k]    for k in _re_indices], dtype=torch.long),
            torch.tensor([scatter_col[k]      for k in _re_indices], dtype=torch.long),
        ], dim=1)  # [NumLegs*MaxRouteLen, 2]
        self._re_leg_ptr  = torch.tensor([scatter_leg_ptr[k]   for k in _re_indices], dtype=torch.long)
        self._re_route_pos = torch.tensor([scatter_route_pos[k] for k in _re_indices], dtype=torch.long)

        # Separator slots
        _sep_indices = [k for k in range(len(_st)) if _sep_mask[k]]
        self._sep_agent_col = torch.stack([
            torch.tensor([scatter_agent[k] for k in _sep_indices], dtype=torch.long),
            torch.tensor([scatter_col[k]   for k in _sep_indices], dtype=torch.long),
        ], dim=1)  # [NumSeps, 2] — may be empty

        # ── Gymnasium VectorEnv setup ────────────────────────────────
        self.single_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(top_k,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(top_k)

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_envs, top_k), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([top_k] * self.num_envs)

        self.spec = None
        self.render_mode = None
        self.closed = False

        # ── Move pre-computed index tensors to device ─────────────────
        dev = self._device
        self._leg_agent_idx = self._leg_agent_idx.to(dev)
        self._leg_leg_idx   = self._leg_leg_idx.to(dev)
        self._fe_agent_col  = self._fe_agent_col.to(dev)
        self._fe_leg_ptr    = self._fe_leg_ptr.to(dev)
        self._re_agent_col  = self._re_agent_col.to(dev)
        self._re_leg_ptr    = self._re_leg_ptr.to(dev)
        self._re_route_pos  = self._re_route_pos.to(dev)
        if len(_sep_indices) > 0:
            self._sep_agent_col = self._sep_agent_col.to(dev)
        else:
            self._sep_agent_col = torch.zeros((0, 2), dtype=torch.long, device=dev)

        # Enforce event tracking for TimeDependentEvaluator (needed for Nash metrics)
        self.bandit._track_events = True
        self.evaluator = TimeDependentEvaluator.from_wrapper(self)

    def _get_obs(self) -> np.ndarray:
        """Fetch blind observation (only action masks) for each leg's OD pair."""
        # [TotalLegs, K]
        obs_t = self.action_masks[self.od_indices_all_legs].float()
        return obs_t.cpu().numpy()

    def _get_info(self, travel_times: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fetch masks and metrics."""
        # [TotalLegs, K]
        masks_t = self.action_masks[self.od_indices_all_legs]
        
        info = {
            "action_mask": masks_t.cpu().numpy(),
        }
        if travel_times is not None:
            info.update({
                "travel_times": travel_times,
                "mean_travel_time": float(travel_times.mean()),
            })
        return info

    # ══════════════════════════════════════════════════════════════════
    #  VectorEnv API
    # ══════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._get_obs(), self._get_info()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Run one full DTA simulation.

        Args:
            actions: [TotalLegs] int array. Selects the route for each individual leg.
        """
        # ── Blindage de l'input ──────────────────────────────────────
        if isinstance(actions, torch.Tensor):
            actions_t = actions.detach().contiguous().to(self._device, dtype=torch.long)
        else:
            actions_t = torch.tensor(actions, dtype=torch.long, device=self._device)
        
        A = self.bandit.num_agents
        
        # ── Get routes for EVERY leg ─────────────────────────────────
        # [TotalLegs, MaxRouteLen]
        selected_routes = self.candidate_routes[self.od_indices_all_legs, actions_t]

        # ── Assemble full multi-leg paths tensor [A, MaxPathLen] — GPU Opt-2 ──
        # Vectorized scatter: no Python loop over agents or legs.
        paths = torch.full(
            (A, self._max_path_len), -1,
            dtype=torch.long, device=self._device
        )
        # 1) Write separators (-2) at pre-computed positions
        if self._sep_agent_col.shape[0] > 0:
            paths[self._sep_agent_col[:, 0], self._sep_agent_col[:, 1]] = -2
        # 2) Write first edges
        paths[self._fe_agent_col[:, 0], self._fe_agent_col[:, 1]] = (
            self.first_edges_all_legs[self._fe_leg_ptr]
        )
        # 3) Write route edges — only valid (non -1) positions contribute
        #    re_values: [NumRouteSlots] – raw route edge values (may be -1 for padding)
        re_values = selected_routes[self._re_leg_ptr, self._re_route_pos]  # [NumRouteSlots]
        # We only scatter valid edges; padding (-1) leaves paths at its -1 default.
        valid_re = re_values >= 0
        if valid_re.any():
            paths[
                self._re_agent_col[valid_re, 0],
                self._re_agent_col[valid_re, 1]
            ] = re_values[valid_re]

        # ── Run the bandit simulation ────────────────────────────────
        paths = paths.detach().contiguous()
        self.bandit.reset(paths)
        _ = self.bandit.step()  # Returns [A] per-vehicle rewards, we ignore it

        # ── Extract per-leg rewards — GPU Opt-1 ─────────────────────
        # leg_metrics shape: [A, MaxLegs, 2] → [:, :, 1] is final travel time
        tt_matrix = self.bandit.dnl.leg_metrics[:, :, 1]
        # Vectorized gather: zero Python loop, single GPU index op
        tt_obs = tt_matrix[self._leg_agent_idx, self._leg_leg_idx]  # [TotalLegs] on device
        rewards = (-tt_obs).cpu().numpy().astype(np.float32)         # one device→host copy

        # ── Extract edge costs for semi-bandit feedback ──
        semi_bandit_costs = None
        if self.feedback_type == "semi":
            # 1. Get average travel times per edge over the episode
            dynamic_tt = self.bandit.dnl.get_dynamic_link_travel_times()
            if dynamic_tt is not None:
                # Average over time intervals to get a single cost per edge
                edge_tt = dynamic_tt.mean(dim=0)
            else:
                # Fallback - get freeflow travel times
                edge_tt = self.bandit.dnl.edge_static[:, 4]
                
            # selected_routes is shape [TotalLegs, MaxRouteLen], padded with -1.
            safe_routes = torch.where(selected_routes >= 0, selected_routes, torch.zeros_like(selected_routes))
            
            # Vectorized extraction of edge costs
            semi_bandit_costs = edge_tt[safe_routes]
            
            # Set costs for padding positions to 0
            semi_bandit_costs = torch.where(selected_routes >= 0, semi_bandit_costs, torch.zeros_like(semi_bandit_costs))

        # ── Package outputs ──────────────────────────────────────────
        terminated = np.ones(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        travel_times = (-rewards).astype(np.float32)
        info = self._get_info(travel_times)
        if semi_bandit_costs is not None:
            info["semi_bandit_feedback"] = semi_bandit_costs.cpu().numpy()
        info["_episode"] = {
            "r": rewards,
            "l": np.ones(self.num_envs, dtype=np.int32),
            "t": travel_times,
        }

        # ── Compute Path-Based Empirical Regret Metrics ──────────────
        estimated_times, _ = self.evaluator.evaluate(
            dnl=self.bandit.dnl,
            departure_times=self.bandit.scenario.departure_times,
            od_indices=self.od_indices_all_legs,
        )
        
        path_metrics = compute_empirical_nash_metrics_tensor(
            actual_travel_times=tt_obs,  # already on device, no numpy round-trip
            actions=actions_t,
            estimated_times=estimated_times
        )
        info.update(path_metrics)

        return self._get_obs(), rewards, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════════

    def get_candidate_paths_info(self) -> Dict[str, Any]:
        """Return summary info about the candidate paths structure."""
        return {
            "num_ods_leg0": self.candidate_routes_leg0.shape[0],
            "K": self.K,
            "max_path_len": self._max_path_len,
            "num_agents": self.num_envs,
            "candidate_routes_leg0_shape": list(self.candidate_routes_leg0.shape),
        }

    def close(self, **kwargs):
        """Clean up resources."""
        pass
