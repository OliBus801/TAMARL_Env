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
from tamarl.envs.components.route_utils import build_routes_csr


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
        reload_paths: bool = False,
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
            force_recompute=reload_paths,
        )

        # ── Build candidate routes in CSR format (memory-efficient) ────
        # Replaces dense [NumOD, K, MaxRouteLen] padded tensor.
        # CSR layout: route (od_idx, k) → routes_flat_csr[offsets[od*K+k] : offsets[od*K+k+1]]
        num_unique_od = len(unique_od)
        flat_np, offsets_np, masks_np, fftt_np = build_routes_csr(
            paths_dict=paths_dict,
            unique_od=unique_od,
            top_k=top_k,
            edge_static_np=scenario.edge_static.numpy(),
        )
        self.routes_flat_csr    = torch.tensor(flat_np,    dtype=torch.int32, device=self._device)
        self.routes_offsets_csr = torch.tensor(offsets_np, dtype=torch.long,  device=self._device)
        self.action_masks       = torch.tensor(masks_np,   dtype=torch.bool,  device=self._device)
        self.fftt_matrix        = fftt_np   # [NumUniqueOD, K] float32
        self.num_unique_od      = num_unique_od




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

        # Enforce event tracking for TimeDependentEvaluator (needed for Nash metrics)
        # self.bandit._track_events = True is no longer needed since
        # TimeDependentEvaluator now uses interval link travel times.
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
        
        # ── CSR route lookup: compute per-leg route start/end from offsets ─
        A = self.bandit.num_agents
        num_legs_np = self.bandit.scenario.num_legs.cpu()
        agent_per_leg = self._leg_agent_idx  # [TotalLegs]

        route_rows   = self.od_indices_all_legs * self.K + actions_t  # [TotalLegs]
        route_starts = self.routes_offsets_csr[route_rows]            # [TotalLegs]
        route_ends   = self.routes_offsets_csr[route_rows + 1]        # [TotalLegs]
        route_lens   = (route_ends - route_starts).long()             # [TotalLegs]

        # ── Build sparse CSR paths (paths_flat + path_offsets) ────────
        leg_total      = 1 + route_lens  # per-leg contribution (first_edge + route edges)
        agent_flat_len = torch.zeros(A, device=self._device, dtype=torch.long)
        agent_flat_len.scatter_add_(0, agent_per_leg, leg_total)
        agent_flat_len += (num_legs_np.to(self._device).long() - 1)  # separators

        path_offsets = torch.zeros(A + 1, device=self._device, dtype=torch.long)
        path_offsets[1:] = torch.cumsum(agent_flat_len, dim=0)
        total_flat_len = int(path_offsets[-1].item())
        paths_flat = torch.empty(total_flat_len, device=self._device, dtype=torch.int32)

        # Compute per-leg write start positions inside paths_flat
        leg_contrib_with_sep = leg_total.clone()
        first_mask = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        first_mask[0] = True
        if self.num_envs > 1:
            first_mask[1:] = agent_per_leg[1:] != agent_per_leg[:-1]
        leg_contrib_with_sep[~first_mask] += 1  # +1 separator before non-first legs

        global_cs = torch.cumsum(leg_contrib_with_sep, dim=0)
        agent_cs_start = torch.zeros(A, device=self._device, dtype=torch.long)
        first_leg_pos = torch.nonzero(first_mask, as_tuple=True)[0]
        agent_cs_start[agent_per_leg[first_leg_pos]] = (
            global_cs[first_leg_pos] - leg_contrib_with_sep[first_leg_pos]
        )
        intra_offset   = global_cs - leg_contrib_with_sep - agent_cs_start[agent_per_leg]
        leg_write_start = path_offsets[agent_per_leg] + intra_offset

        # Write separators for non-first legs
        non_first = ~first_mask
        if non_first.any():
            paths_flat[leg_write_start[non_first]] = -2
            leg_write_start[non_first] += 1

        # Write first edges
        paths_flat[leg_write_start] = self.first_edges_all_legs.int()

        # Write route edges directly from CSR — chunked to avoid OOM
        total_route_edges = int(route_lens.sum().item())
        if total_route_edges > 0:
            CHUNK_SIZE = 65536
            for start_idx in range(0, self.num_envs, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, self.num_envs)
                chunk_route_lens = route_lens[start_idx:end_idx]
                chunk_total_edges = int(chunk_route_lens.sum().item())
                if chunk_total_edges == 0:
                    continue

                chunk_leg_of_edge = torch.repeat_interleave(
                    torch.arange(start_idx, end_idx, device=self._device, dtype=torch.long),
                    chunk_route_lens,
                )

                chunk_cumsum_lens = torch.zeros(end_idx - start_idx + 1, device=self._device, dtype=torch.long)
                chunk_cumsum_lens[1:] = torch.cumsum(chunk_route_lens, dim=0)

                edge_rank = (
                    torch.arange(chunk_total_edges, device=self._device, dtype=torch.long)
                    - chunk_cumsum_lens[chunk_leg_of_edge - start_idx]
                )
                
                src_idx = route_starts[chunk_leg_of_edge] + edge_rank
                dst_idx = leg_write_start[chunk_leg_of_edge] + 1 + edge_rank
                paths_flat[dst_idx] = self.routes_flat_csr[src_idx]

                del chunk_leg_of_edge, chunk_cumsum_lens, edge_rank, src_idx, dst_idx

        # Free all build intermediates before the long simulation loop
        del leg_total, leg_contrib_with_sep, first_mask, global_cs
        del agent_cs_start, intra_offset, non_first, first_leg_pos, agent_per_leg
        import gc; gc.collect()

        # ── Run the bandit simulation ────────────────────────────────
        paths_flat   = paths_flat.detach().contiguous()
        path_offsets = path_offsets.detach().contiguous()
        self.bandit.reset(paths_flat=paths_flat, path_offsets=path_offsets)
        _ = self.bandit.step()

        # ── Extract per-leg rewards ──────────────────────────────────
        tt_matrix = self.bandit.dnl.leg_metrics[:, :, 1]
        tt_obs    = tt_matrix[self._leg_agent_idx, self._leg_leg_idx]  # [TotalLegs]

        # ── Build valid_leg_mask: True for legs that actually departed ─
        dep_matrix = self.bandit.dnl.leg_departure_times  # [A, MaxLegs]
        dep_per_leg = dep_matrix[self._leg_agent_idx, self._leg_leg_idx]  # [TotalLegs]
        valid_leg_mask = (dep_per_leg >= 0)  # legs with departure_time == -1 never started

        rewards   = (-tt_obs).cpu().numpy().astype(np.float32)

        # ── Free-flow travel times for the chosen actions ────────────
        # fftt_matrix is [NumUniqueOD, K] numpy float32
        od_np = self.od_indices_all_legs.cpu().numpy()
        act_np = actions_t.cpu().numpy()
        fftt_chosen = self.fftt_matrix[od_np, act_np]  # [TotalLegs] numpy

        # ── Semi-bandit feedback (reconstructs dense tensor lazily) ──
        semi_bandit_costs = None
        if self.feedback_type == "semi":
            dynamic_tt = self.bandit.dnl.get_dynamic_link_travel_times()
            edge_tt    = dynamic_tt.mean(dim=0) if dynamic_tt is not None else self.bandit.dnl.edge_static[:, 4]
            max_len_s  = int(route_lens.max().item()) if route_lens.numel() > 0 else 0
            if max_len_s > 0:
                sel = torch.full((self.num_envs, max_len_s), -1, device=self._device, dtype=torch.int32)
                total_s = int(route_lens.sum().item())
                if total_s > 0:
                    loe = torch.repeat_interleave(
                        torch.arange(self.num_envs, device=self._device), route_lens)
                    cs_s = torch.zeros(self.num_envs + 1, device=self._device, dtype=torch.long)
                    cs_s[1:] = torch.cumsum(route_lens, dim=0)
                    rk_s = torch.arange(total_s, device=self._device) - cs_s[loe]
                    sel[loe, rk_s] = self.routes_flat_csr[route_starts[loe] + rk_s]
                safe = torch.where(sel >= 0, sel, torch.zeros_like(sel))
                semi_bandit_costs = torch.where(sel >= 0, edge_tt[safe], torch.zeros_like(edge_tt[safe]))
        del route_rows, route_starts, route_ends, route_lens, leg_write_start

        # ── Package outputs (filtered by valid_leg_mask) ─────────────
        terminated = np.ones(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        # Only count legs that actually departed for travel time metrics
        valid_mask_np = valid_leg_mask.cpu().numpy()
        travel_times = (-rewards).astype(np.float32)
        valid_travel_times = travel_times[valid_mask_np]

        info = self._get_info(valid_travel_times if valid_travel_times.size > 0 else travel_times)
        if semi_bandit_costs is not None:
            info["semi_bandit_feedback"] = semi_bandit_costs.cpu().numpy()
        info["_episode"] = {
            "r": rewards[valid_mask_np],
            "l": np.ones(int(valid_mask_np.sum()), dtype=np.int32),
            "t": valid_travel_times,
        }
        info["fftt_chosen"] = fftt_chosen[valid_mask_np]
        info["valid_leg_mask"] = valid_mask_np

        n_masked = int((~valid_leg_mask).sum().item())
        if n_masked > 0:
            info["n_masked_legs"] = n_masked

        # ── Compute Path-Based Empirical Regret Metrics ──────────────
        if self.bandit.collect_link_tt:
            estimated_times, _ = self.evaluator.evaluate(
                dnl=self.bandit.dnl,
                departure_times=self.bandit.scenario.departure_times,
                od_indices=self.od_indices_all_legs,
            )
            
            path_metrics = compute_empirical_nash_metrics_tensor(
                actual_travel_times=tt_obs,  # already on device, no numpy round-trip
                actions=actions_t,
                estimated_times=estimated_times,
                valid_mask=valid_leg_mask,
            )
            info.update(path_metrics)

        return self._get_obs(), rewards, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════════

    def get_candidate_paths_info(self) -> Dict[str, Any]:
        """Return summary info about the candidate paths structure."""
        total_edges = int(self.routes_flat_csr.shape[0])
        num_routes  = self.num_unique_od * self.K
        avg_len     = total_edges / num_routes if num_routes > 0 else 0
        return {
            "num_unique_od": self.num_unique_od,
            "K": self.K,
            "num_agents": self.num_envs,
            "routes_flat_size": total_edges,
            "avg_route_len": round(avg_len, 1),
        }

    def close(self, **kwargs):
        """Clean up resources."""
        pass
