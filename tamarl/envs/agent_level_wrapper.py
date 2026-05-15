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

        # -- Pre-compute per-leg info for vectorized path assembly (Opt-2) --
        # Routes in paths[] MUST be compact (no -1 gaps before -2 separators).
        # We store "assumed" fe base columns (assuming max_rl per prior leg).
        # At step()-time we correct using actual route lengths via cumsum.
        self._max_rl = max_route_len
        _max_rl = self._max_rl  # local alias for the loop below
        leg_agent_list  = []
        leg_fe_base     = []   # fe col assuming max_rl per all prior legs of same agent
        leg_is_first    = []   # True if leg_j == 0 (no separator before fe)
        re_leg_list     = []
        re_route_list   = []

        leg_ptr = 0
        for i in range(A):
            n = num_legs_np[i]
            ptr = 0
            for leg_j in range(n):
                if leg_j > 0:
                    ptr += 1   # separator
                fe_col = ptr
                ptr += 1       # first edge
                for e in range(_max_rl):
                    re_leg_list.append(leg_ptr)
                    re_route_list.append(e)
                ptr += _max_rl  # assumed max_rl route slots (corrected at step-time)

                leg_agent_list.append(i)
                leg_fe_base.append(fe_col)
                leg_is_first.append(leg_j == 0)
                leg_ptr += 1

        self._leg_agent2    = torch.tensor(leg_agent_list, dtype=torch.long)
        self._leg_fe_base   = torch.tensor(leg_fe_base,   dtype=torch.long)  # [TotalLegs]
        self._leg_is_first  = torch.tensor(leg_is_first,  dtype=torch.bool)  # [TotalLegs]
        self._re_leg_ptr2   = torch.tensor(re_leg_list,   dtype=torch.long)  # [TotalLegs*MaxRouteLen]
        self._re_route_pos2 = torch.tensor(re_route_list, dtype=torch.long)  # [TotalLegs*MaxRouteLen]

        # -- Gymnasium VectorEnv setup ----------------------------------------
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

        # -- Move pre-computed index tensors to device ------------------------
        dev = self._device
        self._leg_agent_idx  = self._leg_agent_idx.to(dev)
        self._leg_leg_idx    = self._leg_leg_idx.to(dev)
        self._leg_agent2     = self._leg_agent2.to(dev)
        self._leg_fe_base    = self._leg_fe_base.to(dev)
        self._leg_is_first   = self._leg_is_first.to(dev)
        self._re_leg_ptr2    = self._re_leg_ptr2.to(dev)
        self._re_route_pos2  = self._re_route_pos2.to(dev)

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
        # The simulator reads paths compactly: valid edges only, terminated by -1 or -2.
        # We CANNOT leave -1 gaps mid-route before a -2 separator.
        # Strategy: pre-compute fe/separator positions (fixed per scenario),
        # compute compact route column offsets episode-by-episode via cumsum.
        paths = torch.full(
            (A, self._max_path_len), -1,
            dtype=torch.long, device=self._device
        )

        # All route edges for this episode: [TotalLegs * MaxRouteLen]
        re_all = selected_routes[self._re_leg_ptr2, self._re_route_pos2]  # raw (may be -1)
        valid_re = re_all >= 0  # [TotalLegs * MaxRouteLen]

        # Group ID per route slot: which leg does each slot belong to?  [TotalLegs * MaxRouteLen]
        # _re_leg_ptr2 already stores this (0..TotalLegs-1).
        # Compact column within each leg = position among valid edges before this slot.
        # cumsum within group: reset at each leg boundary.
        # local_rank[k] = number of valid edges in leg _re_leg_ptr2[k] before slot k
        valid_int = valid_re.long()
        global_cs = torch.cumsum(valid_int, dim=0)  # [TotalLegs*MaxRouteLen]
        # Offset: value of cumsum just before the first slot of each leg
        # = global_cs at the last slot of the previous leg
        leg_ids = self._re_leg_ptr2  # [TotalLegs * MaxRouteLen]
        # First slot of each leg: where leg_ids changes (or position 0)
        first_slot_mask = torch.zeros(leg_ids.shape[0], dtype=torch.bool, device=self._device)
        first_slot_mask[0] = True
        if leg_ids.shape[0] > 1:
            first_slot_mask[1:] = (leg_ids[1:] != leg_ids[:-1])
        # Scatter the cumsum value at each leg boundary start (subtract 1 since cumsum is inclusive)
        leg_cs_offset = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        # global_cs[first_slot] gives the cumsum value AT the first slot.
        # The offset for a leg is the cumsum value JUST BEFORE first slot.
        # = global_cs[first_slot] - valid_int[first_slot]
        first_slot_indices = torch.nonzero(first_slot_mask, as_tuple=True)[0]
        leg_cs_offset[leg_ids[first_slot_indices]] = (
            global_cs[first_slot_indices] - valid_int[first_slot_indices]
        )
        # local_rank = global_cs - leg_cs_offset[leg_id]
        leg_offset_per_slot = leg_cs_offset[leg_ids]  # [TotalLegs*MaxRouteLen]
        local_rank = global_cs - leg_offset_per_slot  # 1-indexed count of valid edges up to here

        # Destination column for each valid route edge:
        #   paths[agent, fe_base_of_leg + 1 + (local_rank - 1)]
        #   = paths[agent, fe_base_of_leg + local_rank]
        # fe_base per slot (looked up from leg_id)
        fe_base_per_slot = self._leg_fe_base[leg_ids]  # [TotalLegs*MaxRouteLen]
        re_dst_col = fe_base_per_slot + local_rank      # [TotalLegs*MaxRouteLen], only valid where valid_re
        re_dst_agent = self._leg_agent2[leg_ids]        # [TotalLegs*MaxRouteLen]

        # But fe_base_per_slot assumes max_rl per prior leg (from __init__).
        # We need to correct fe_base for legs > 0: each prior leg contributes
        # its actual route length (not max_rl) to the column offset.
        # This correction is the delta between actual and assumed column.
        # Compute: for each leg, actual route length = number of valid edges
        actual_len = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        # Total valid edges per leg = global_cs at last slot of each leg
        # last slot of leg l = first_slot_indices[l+1] - 1 (or end of array for last leg)
        last_slot_indices = torch.zeros_like(first_slot_indices)
        last_slot_indices[:-1] = first_slot_indices[1:] - 1
        last_slot_indices[-1]  = leg_ids.shape[0] - 1
        actual_len_vals = global_cs[last_slot_indices] - leg_cs_offset  # [TotalLegs]

        # Column correction per leg: sum of (max_rl - actual_len) for all PRIOR legs of same agent.
        # This accounts for the difference between assumed (max_rl) and actual column positions.
        _max_rl = self._max_rl  # max route length (set in __init__)
        savings = (_max_rl - actual_len_vals).long()  # [TotalLegs]

        # For each leg, correction = exclusive cumsum of savings within the same agent group.
        # Legs of the same agent are contiguous in our ordering.
        agent_per_leg = self._leg_agent2[first_slot_indices]  # [TotalLegs]
        savings_cumsum = torch.cumsum(savings, dim=0)          # global cumsum

        # Offset at first leg of each agent = savings_cumsum value JUST BEFORE that leg
        first_leg_mask2 = torch.zeros(first_slot_indices.shape[0], dtype=torch.bool, device=self._device)
        first_leg_mask2[0] = True
        if first_slot_indices.shape[0] > 1:
            first_leg_mask2[1:] = (agent_per_leg[1:] != agent_per_leg[:-1])
        agent_cs_offset = torch.zeros(A, dtype=torch.long, device=self._device)
        first_leg_global_indices = torch.nonzero(first_leg_mask2, as_tuple=True)[0]
        agent_cs_offset[agent_per_leg[first_leg_global_indices]] = (
            savings_cumsum[first_leg_global_indices] - savings[first_leg_global_indices]
        )
        # correction[leg] = exclusive cumsum of savings for legs of the same agent
        correction_per_leg = (
            savings_cumsum - savings - agent_cs_offset[agent_per_leg]
        )  # [TotalLegs]

        # Apply correction to fe_base_per_slot
        correction_per_slot = correction_per_leg[leg_ids]   # [TotalLegs*MaxRouteLen]
        re_dst_col_corrected = re_dst_col - correction_per_slot

        # Write valid route edges
        if valid_re.any():
            paths[re_dst_agent[valid_re], re_dst_col_corrected[valid_re]] = re_all[valid_re]

        # Write first edges (corrected base columns)
        # fe_base correction per leg
        fe_base_corrected = self._leg_fe_base - correction_per_leg  # [TotalLegs]
        paths[self._leg_agent2, fe_base_corrected] = self.first_edges_all_legs

        # Write separators (-2): column = fe_base_corrected[leg] - 1 for leg_j > 0
        multi_leg_mask = ~self._leg_is_first
        if multi_leg_mask.any():
            sep_agents = self._leg_agent2[multi_leg_mask]
            sep_cols   = fe_base_corrected[multi_leg_mask] - 1
            paths[sep_agents, sep_cols] = -2

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
