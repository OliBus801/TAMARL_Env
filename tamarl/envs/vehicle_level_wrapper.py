"""VehicleLevelWrapper — gymnasium.vector.VectorEnv bridge for DTABanditEnv.

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
from gymnasium.vector import VectorEnv

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.components.path_enumerator import enumerate_top_k_paths


class VehicleLevelWrapper(VectorEnv):
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
    ):
        self.bandit = bandit
        self.K = top_k
        self._device = bandit._device
        self.alpha = 0.1  # EMA coefficient
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

        paths_dict = enumerate_top_k_paths(
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
        costs_np = np.zeros((num_unique_od, top_k), dtype=np.float32)
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
                
                # Initial cost = free-flow travel time
                path_ff = sum(ff_times[e] for e in path)
                costs_np[od_idx, k_idx] = float(fe_ff + path_ff)
                
                # Action is valid if it's one of the unique paths found
                masks_np[od_idx, k_idx] = (k_idx < len(paths_list))

        self.candidate_routes = torch.tensor(
            cand_np, dtype=torch.long, device=self._device
        )  # [Num_Unique_OD, K, MaxRouteLen]
        
        self.historical_costs = torch.tensor(
            costs_np, dtype=torch.float32, device=self._device
        )  # [Num_Unique_OD, K]
        
        self.action_masks = torch.tensor(
            masks_np, dtype=torch.bool, device=self._device
        )  # [Num_Unique_OD, K]

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

    def _get_obs(self) -> np.ndarray:
        """Fetch historical costs for each leg's OD pair."""
        # [TotalLegs, K]
        obs_t = self.historical_costs[self.od_indices_all_legs]
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
        actions_t = torch.tensor(actions, dtype=torch.long, device=self._device)
        A = self.bandit.num_agents
        
        # ── Get routes for EVERY leg ─────────────────────────────────
        # [TotalLegs, MaxRouteLen]
        selected_routes = self.candidate_routes[self.od_indices_all_legs, actions_t]

        # ── Assemble full multi-leg paths tensor [A, MaxPathLen] ─────
        paths = torch.full(
            (A, self._max_path_len), -1,
            dtype=torch.long, device=self._device
        )

        leg_ptr = 0
        for i in range(A):
            n_legs = self.bandit.scenario.num_legs[i].item()
            ptr = 0
            for leg_j in range(n_legs):
                if leg_j > 0:
                    paths[i, ptr] = -2  # leg separator
                    ptr += 1
                
                paths[i, ptr] = self.first_edges_all_legs[leg_ptr]
                ptr += 1
                
                route = selected_routes[leg_ptr]
                # Filter out padding (-1)
                valid_edges = route[route != -1]
                L = valid_edges.size(0)
                paths[i, ptr : ptr + L] = valid_edges
                ptr += L
                
                leg_ptr += 1

        # ── Run the bandit simulation ────────────────────────────────
        self.bandit.reset(paths)
        _ = self.bandit.step()  # Returns [A] per-vehicle rewards, we ignore it

        # ── Extract per-leg rewards and update history ───────────────
        # leg_metrics shape: [A, MaxLegs, 2] -> [:, :, 1] is final travel time
        tt_matrix = self.bandit.dnl.leg_metrics[:, :, 1]
        
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        tt_obs = torch.zeros(self.num_envs, device=self._device)
        for idx, (i, leg_j) in enumerate(self.leg_to_agent):
            tt = float(tt_matrix[i, leg_j].item())
            rewards[idx] = -tt
            tt_obs[idx] = tt

        # Update historical costs (EMA)
        # Using flattened index for scatter_reduce
        flat_idx = self.od_indices_all_legs * self.K + actions_t
        
        num_od_k = self.candidate_routes.shape[0] * self.K
        episode_costs = torch.zeros(num_od_k, device=self._device)
        
        # Mean of travel times for each (OD, K) pair in this episode
        episode_costs.scatter_reduce_(0, flat_idx, tt_obs, reduce='mean', include_self=False)
        
        # Only update where we had samples
        mask_updated = torch.zeros(num_od_k, device=self._device, dtype=torch.bool)
        mask_updated.scatter_(0, flat_idx, True)
        
        ep_costs_2d = episode_costs.view(-1, self.K)
        mask_updated_2d = mask_updated.view(-1, self.K)
        
        self.historical_costs = torch.where(
            mask_updated_2d,
            (1.0 - self.alpha) * self.historical_costs + self.alpha * ep_costs_2d,
            self.historical_costs
        )

        # ── Package outputs ──────────────────────────────────────────
        terminated = np.ones(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        travel_times = (-rewards).astype(np.float32)
        info = self._get_info(travel_times)
        info["_episode"] = {
            "r": rewards,
            "l": np.ones(self.num_envs, dtype=np.int32),
            "t": travel_times,
        }

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
