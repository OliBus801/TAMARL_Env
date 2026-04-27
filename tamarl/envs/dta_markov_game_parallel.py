"""PettingZoo ParallelEnv wrapper for TorchDNLMATSim in RL mode.

This environment wraps the DNL simulator as a Markov Game where each
agent independently chooses outgoing edges at decision points (capacity buffer).

Supports two formulations:
  - link-based (default): agents choose next edge at every node.
  - path-based: agents choose a complete path at departure, then auto-follow.
"""
from __future__ import annotations

import functools
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

from tamarl.core.dnl_matsim import TorchDNLMATSim
from tamarl.envs.scenario_loader import load_scenario
from tamarl.envs.components.scheduler import DecisionScheduler
from tamarl.envs.components.actions import ActionManager
from tamarl.envs.components.observations import ObservationBuilder
from tamarl.envs.components.rewards import Rewarder
from tamarl.envs.components.path_enumerator import enumerate_top_k_paths


class DTAMarkovGameEnv(ParallelEnv):
    """Multi-agent traffic assignment environment using PettingZoo Parallel API.
    
    Each agent is a driver routing through a traffic network. At each decision
    point (when an agent reaches the end of a link), it chooses the next
    outgoing link. The environment advances the DNL simulator between decisions.
    
    Observation: [current_node, destination, normalized_time, 
                  outgoing_occupancies, outgoing_ff_times]
    Action: Discrete(max_out_degree) — index of outgoing edge to take
    Reward: -dt per simulation tick (dense, γ=1 → minimizes total travel time)
    """

    metadata = {"render_modes": ["ansi"], "name": "dta_markov_game_v0"}

    def __init__(
        self,
        scenario_path: str,
        population_filter: Optional[str] = None,
        timestep: float = 1.0,
        scale_factor: float = 1.0,
        max_steps: int = 3600,
        device: str = "cpu",
        seed: Optional[int] = None,
        stuck_threshold: int = 10,
        track_events: bool = False,
        formulation: str = "link-based",
        top_k_paths: int = 3,
    ):
        super().__init__()
        
        self._scenario_path = scenario_path
        self._population_filter = population_filter
        self._timestep = timestep
        self._scale_factor = scale_factor
        self._max_steps = max_steps
        self._device = device
        self._seed = seed
        self._stuck_threshold = stuck_threshold
        self._formulation = formulation
        self._top_k_paths = top_k_paths
        
        # Load scenario
        self._scenario = load_scenario(
            scenario_path,
            population_filter=population_filter,
            timestep=timestep,
            scale_factor=scale_factor,
        )
        
        # Create DNL in RL mode
        self.dnl = TorchDNLMATSim(
            edge_static=self._scenario.edge_static,
            paths=None,  # RL mode
            device=device,
            departure_times=self._scenario.departure_times,
            edge_endpoints=self._scenario.edge_endpoints,
            stuck_threshold=stuck_threshold,
            dt=timestep,
            seed=seed,
            first_edges=self._scenario.first_edges,
            destinations=self._scenario.destinations,
            num_legs=self._scenario.num_legs,
            act_end_times=self._scenario.act_end_times,
            act_durations=self._scenario.act_durations,
            track_events=track_events,
        )
        
        # Components
        self._scheduler = DecisionScheduler(self.dnl)
        self._action_mgr = ActionManager(self.dnl)
        self._obs_builder = ObservationBuilder(self.dnl, max_steps=max_steps)
        self._rewarder = Rewarder(self.dnl)
        
        # ── OD pairs (computed once, used by Q-learning) ──
        self._od_pairs = self._compute_od_pairs()
        
        # ── Path-based formulation ──
        self._paths_per_od: Optional[Dict[Tuple[int,int], List[List[int]]]] = None
        if formulation == "path-based":
            self._paths_per_od = self._enumerate_paths(top_k_paths)
            # Per-agent chosen path tracking
            self._agent_chosen_path: List[Optional[List[int]]] = [
                None for _ in range(self.dnl.num_agents)
            ]
            self._agent_path_ptr = np.zeros(self.dnl.num_agents, dtype=np.int64)
        
        # PettingZoo agent list
        self.possible_agents = [f"agent_{i}" for i in range(self.dnl.num_agents)]
        self.agents = []
        
        # Track cumulative rewards per episode (tensor-based)
        self._cumulative_rewards_t = torch.zeros(
            self.dnl.num_agents, device=device, dtype=torch.float32
        )
        # Active agent indices (tensor) — avoids string parsing
        self._active_indices = torch.empty(0, device=device, dtype=torch.long)
        
        # Legacy dict for PettingZoo compat
        self._cumulative_rewards: Dict[str, float] = {}
        self._ticks_since_last_step = 0
        
        # Pre-allocate zero masks
        self._zero_mask = torch.zeros(
            self.dnl.max_out_degree, device=device, dtype=torch.int8
        )

    # ── OD pair + path helpers ────────────────────────────────────────

    def _compute_od_pairs(self) -> np.ndarray:
        """Compute (origin_node, dest_node) for each agent (first leg).

        The origin is the *to_node* of the first edge — the node where
        the agent actually arrives and starts making routing decisions.

        Returns:
            [N, 2] int32 numpy array.
        """
        first_edges = self._scenario.first_edges[:, 0].numpy()  # [N]
        edge_eps = self._scenario.edge_endpoints.numpy()          # [E, 2]
        origins = edge_eps[first_edges, 1]                        # to_node of first edge
        dests = self._scenario.destinations[:, 0].numpy()         # dest_node of first leg
        return np.stack([origins, dests], axis=1).astype(np.int32)

    def _enumerate_paths(self, k: int) -> Dict[Tuple[int,int], List[List[int]]]:
        """Enumerate top-k loopless paths for each unique OD pair."""
        edge_eps = self._scenario.edge_endpoints.numpy()
        ff_times = self.dnl.ff_travel_time_steps.cpu().numpy().astype(np.float64)
        unique_ods = np.unique(self._od_pairs, axis=0)
        paths = enumerate_top_k_paths(
            num_nodes=self.dnl.num_nodes,
            edge_endpoints=edge_eps,
            ff_times=ff_times,
            od_pairs=unique_ods,
            k=k,
        )
        return paths

    @property
    def od_pairs(self) -> np.ndarray:
        """[N, 2] array of (origin_node, dest_node) per agent."""
        return self._od_pairs

    @property
    def paths_per_od(self) -> Optional[Dict[Tuple[int,int], List[List[int]]]]:
        """Enumerated paths per OD pair (path-based only)."""
        return self._paths_per_od

    @property
    def formulation(self) -> str:
        return self._formulation

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_builder.observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        if self._formulation == "path-based":
            return spaces.Discrete(self._top_k_paths)
        return spaces.Discrete(self.dnl.max_out_degree)

    # ══════════════════════════════════════════════════════════════════
    #  BATCHED API  (tensors in, tensors out — no string processing)
    # ══════════════════════════════════════════════════════════════════

    def reset_batched(
        self, seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset the environment — tensor API.
        
        Returns:
            obs_all:          [N, obs_dim]  observations for ALL agents
            masks_deciding:   [K, max_deg]  action masks for deciding agents only
            deciding_indices: [K]           which agent indices are deciding
            active_indices:   [N_active]    which agents are still alive
        """
        if seed is not None:
            self._seed = seed
        
        self.dnl.reset()
        if seed is not None:
            self.dnl.rng.manual_seed(seed)
        
        N = self.dnl.num_agents
        
        # All agents start alive
        self._active_indices = torch.arange(N, device=self.dnl.device)
        self.agents = list(self.possible_agents)
        self._cumulative_rewards_t.zero_()
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._ticks_since_last_step = 0
        self._rewarder.reset()
        
        # Reset path-based state
        if self._formulation == "path-based":
            self._agent_chosen_path = [None for _ in range(N)]
            self._agent_path_ptr = np.zeros(N, dtype=np.int64)
        
        # Advance simulation until first decision
        self._advance_to_next_decisions()
        
        # Build observations
        deciding = self._scheduler.get_deciding_agents()  # [K]
        obs_deciding = self._obs_builder.build_observations_batched(deciding)  # [K, obs_dim]
        
        # Build initial observations for all agents
        obs_all = self._obs_builder.build_initial_observations_batched()  # [N, obs_dim]
        
        # Overwrite deciding agents' obs with their actual observations
        if deciding.numel() > 0:
            obs_all[deciding] = obs_deciding
        
        # Action masks for deciding agents
        if self._formulation == "path-based":
            masks = self._build_path_masks(deciding)
        else:
            masks = self._action_mgr.get_action_masks_batched(deciding)  # [K, max_deg]
        
        return obs_all, masks, deciding, self._active_indices.clone()

    def step_batched(
        self,
        action_indices: torch.Tensor,
        action_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one macro-step — tensor API.
        
        Args:
            action_indices: [K] agent indices that are acting
            action_values:  [K] action values (link-based: local edge indices,
                            path-based: path choice indices)
            
        Returns:
            obs_active:        [N_active_new, obs_dim]  observations for remaining active agents
            rewards_active:    [N_active_old]           rewards for previously-active agents
            terminated_mask:   [N_active_old]           bool mask — which agents terminated
            truncated_mask:    [N_active_old]           bool mask — which agents were truncated
            masks_deciding:    [K', max_deg]            action masks for newly-deciding agents
            deciding_indices:  [K']                     which agent indices are newly deciding
        """
        prev_active = self._active_indices  # [N_active_old]
        
        # 1. Apply actions (vectorised)
        if action_indices.numel() > 0:
            if self._formulation == "path-based":
                self._apply_path_actions(action_indices, action_values)
            else:
                self._action_mgr.apply_actions_tensor(action_indices, action_values)
        
        # 2. Advance DNL
        n_ticks = self._advance_to_next_decisions()
        
        # 3. Compute rewards (vectorised)
        rewards = self._rewarder.compute_batch_rewards(prev_active, n_ticks)  # [N_active_old]
        
        # 4. Terminations & truncations (vectorised)
        statuses = self.dnl.status[prev_active]  # [N_active_old]
        terminated_mask = (statuses == 3)  # arrived
        truncated_mask = torch.zeros_like(terminated_mask)
        
        if self.dnl.current_step >= self._max_steps:
            truncated_mask = ~terminated_mask  # truncate all non-terminated
        
        # 5. Update cumulative rewards
        self._cumulative_rewards_t[prev_active] += rewards
        
        # 6. Filter to surviving agents
        alive_mask = ~terminated_mask & ~truncated_mask
        self._active_indices = prev_active[alive_mask]
        
        # Update PettingZoo agents list
        self.agents = [f"agent_{i}" for i in self._active_indices.tolist()]
        
        # 7. Build observations for active agents
        deciding = self._scheduler.get_deciding_agents()  # [K']
        
        # Filter deciding to only include agents that are still active
        if deciding.numel() > 0 and self._active_indices.numel() > 0:
            is_active = (deciding.unsqueeze(1) == self._active_indices.unsqueeze(0)).any(dim=1)
            deciding = deciding[is_active]
        elif self._active_indices.numel() == 0:
            deciding = torch.empty(0, device=self.dnl.device, dtype=torch.long)
        
        # In path-based mode, auto-route deciding agents that already have a chosen path
        if self._formulation == "path-based" and deciding.numel() > 0:
            deciding = self._auto_route_path_agents(deciding)
        
        n_active = self._active_indices.numel()
        obs_dim = self._obs_builder.obs_size
        
        n_actions = self._top_k_paths if self._formulation == "path-based" else self.dnl.max_out_degree
        
        if n_active == 0:
            obs_active = torch.empty((0, obs_dim), device=self.dnl.device)
            masks = torch.empty((0, n_actions), device=self.dnl.device, dtype=torch.int8)
            return obs_active, rewards, terminated_mask, truncated_mask, masks, deciding
        
        # Get deciding agents' observations
        obs_deciding = self._obs_builder.build_observations_batched(deciding)  # [K', obs_dim]
        
        # Build obs for all active agents
        obs_active = torch.zeros((n_active, obs_dim), device=self.dnl.device)
        
        # Create a quick lookup: agent_idx -> position in active array
        # We need to map deciding indices to positions in _active_indices
        if deciding.numel() > 0 and n_active > 0:
            # For each deciding agent, find its position in _active_indices
            # Use broadcasting: _active_indices[pos] == deciding[j]
            dec_positions = (self._active_indices.unsqueeze(1) == deciding.unsqueeze(0)).any(dim=1)
            obs_active[dec_positions] = obs_deciding
        
        # For non-deciding active agents, build minimal observations
        if n_active > 0:
            active_idx = self._active_indices
            active_statuses = self.dnl.status[active_idx]
            
            # Create set of deciding agent indices for fast lookup
            if deciding.numel() > 0:
                is_deciding = (active_idx.unsqueeze(1) == deciding.unsqueeze(0)).any(dim=1)
            else:
                is_deciding = torch.zeros(n_active, device=self.dnl.device, dtype=torch.bool)
            
            needs_obs = ~is_deciding & ((active_statuses == 1) | (active_statuses == 2))
            
            if needs_obs.any():
                need_idx = active_idx[needs_obs]
                curr_edges = self.dnl.current_edge[need_idx]
                nodes = self.dnl.edge_endpoints[curr_edges, 1].float()
                c_legs = self.dnl.current_leg[need_idx]
                dests = self.dnl.destinations[need_idx, c_legs].float()
                n_bins = self._obs_builder.N_TIME_BINS
                norm_time = float(min((self.dnl.current_step * n_bins) // self._max_steps, n_bins - 1))
                
                obs_active[needs_obs, 0] = nodes
                obs_active[needs_obs, 1] = dests
                obs_active[needs_obs, 2] = norm_time
        
        # 8. Action masks for deciding agents
        if self._formulation == "path-based":
            masks = self._build_path_masks(deciding)
        else:
            masks = self._action_mgr.get_action_masks_batched(deciding)  # [K', max_deg]
        
        return obs_active, rewards, terminated_mask, truncated_mask, masks, deciding

    def has_active_agents(self) -> bool:
        """Check if any agents are still active."""
        return self._active_indices.numel() > 0

    # ══════════════════════════════════════════════════════════════════
    #  PETTINGZOO DICT API  (delegates to batched, converts to dicts)
    # ══════════════════════════════════════════════════════════════════

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset the environment for a new episode (PettingZoo API)."""
        obs_all_t, masks_t, deciding, active = self.reset_batched(seed)
        
        # Convert to dicts
        obs_all_np = obs_all_t.cpu().numpy()
        observations = {f"agent_{i}": obs_all_np[i] for i in range(self.dnl.num_agents)}
        
        # Infos with action masks
        deciding_set = set(deciding.tolist())
        masks_np = masks_t.cpu().numpy() if masks_t.numel() > 0 else np.empty((0, self.dnl.max_out_degree), dtype=np.int8)
        deciding_list = deciding.tolist()
        masks_dict = {deciding_list[i]: masks_np[i] for i in range(len(deciding_list))}
        
        infos = {}
        zero_mask_np = np.zeros(self.dnl.max_out_degree, dtype=np.int8)
        
        for agent_id in self.agents:
            idx = int(agent_id.split("_")[-1])
            info = {}
            if idx in masks_dict:
                info["action_mask"] = masks_dict[idx]
            else:
                info["action_mask"] = zero_mask_np.copy()
            info["curr_leg"] = int(self.dnl.current_leg[idx].item())
            infos[agent_id] = info
        
        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """Execute one macro-step (PettingZoo API)."""
        current_agents = list(self.agents)
        
        # Convert dict actions to tensors
        if actions:
            act_indices = []
            act_values = []
            for agent_id, action in actions.items():
                act_indices.append(int(agent_id.split("_")[-1]))
                act_values.append(action)
            action_indices_t = torch.tensor(act_indices, device=self.dnl.device, dtype=torch.long)
            action_values_t = torch.tensor(act_values, device=self.dnl.device, dtype=torch.long)
        else:
            action_indices_t = torch.empty(0, device=self.dnl.device, dtype=torch.long)
            action_values_t = torch.empty(0, device=self.dnl.device, dtype=torch.long)
        
        prev_active = self._active_indices.clone()
        prev_active_list = prev_active.tolist()
        
        obs_active_t, rewards_t, term_mask, trunc_mask, masks_t, deciding = self.step_batched(
            action_indices_t, action_values_t
        )
        
        # Convert rewards to dict
        rewards_np = rewards_t.cpu().numpy()
        rewards = {f"agent_{prev_active_list[i]}": float(rewards_np[i]) 
                   for i in range(len(prev_active_list))}
        
        # Convert terminations/truncations to dict
        term_np = term_mask.cpu().numpy()
        trunc_np = trunc_mask.cpu().numpy()
        terminations = {f"agent_{prev_active_list[i]}": bool(term_np[i])
                        for i in range(len(prev_active_list))}
        truncations = {f"agent_{prev_active_list[i]}": bool(trunc_np[i])
                       for i in range(len(prev_active_list))}
        
        # Update cumulative rewards dict
        for agent_id in current_agents:
            self._cumulative_rewards[agent_id] = (
                self._cumulative_rewards.get(agent_id, 0.0) + rewards.get(agent_id, 0.0)
            )
        
        # Convert observations to dict
        obs_np = obs_active_t.cpu().numpy() if obs_active_t.numel() > 0 else np.empty((0, self._obs_builder.obs_size), dtype=np.float32)
        active_list = self._active_indices.tolist()
        observations = {f"agent_{active_list[i]}": obs_np[i] 
                        for i in range(len(active_list))}
        
        # Build infos dict
        deciding_set = set(deciding.tolist()) if deciding.numel() > 0 else set()
        masks_np = masks_t.cpu().numpy() if masks_t.numel() > 0 else np.empty((0, self.dnl.max_out_degree), dtype=np.int8)
        deciding_list = deciding.tolist() if deciding.numel() > 0 else []
        masks_lookup = {deciding_list[i]: masks_np[i] for i in range(len(deciding_list))}
        
        infos = {}
        zero_mask_np = np.zeros(self.dnl.max_out_degree, dtype=np.int8)
        for agent_id in current_agents:
            idx = int(agent_id.split("_")[-1])
            info = {}
            if idx in masks_lookup:
                info["action_mask"] = masks_lookup[idx]
            else:
                info["action_mask"] = zero_mask_np.copy()
            info["curr_leg"] = int(self.dnl.current_leg[idx].item())
            infos[agent_id] = info
        
        return observations, rewards, terminations, truncations, infos

    # ══════════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _advance_to_next_decisions(self) -> int:
        """Advance DNL tick-by-tick until a decision event occurs or episode ends."""
        ticks = 0
        max_idle = self._max_steps
        
        while ticks < max_idle:
            if ticks > 0 and self._scheduler.has_deciding_agents():
                break
            all_done = (self.dnl.status >= 3).all().item()
            if all_done:
                break
            if self.dnl.current_step >= self._max_steps:
                break
            self.dnl.step()
            ticks += 1
        
        self._ticks_since_last_step = ticks
        return ticks

    # ── Path-based helpers ───────────────────────────────────────────

    def _build_path_masks(self, deciding: torch.Tensor) -> torch.Tensor:
        """Build action masks for path-based formulation.
        
        Each action index corresponds to one of the top-k paths for the
        agent's OD pair.  Mask is 1 if the path exists, 0 otherwise.
        """
        K = deciding.numel()
        masks = torch.zeros((K, self._top_k_paths), device=self.dnl.device, dtype=torch.int8)
        if K == 0:
            return masks
        
        dec_np = deciding.cpu().numpy()
        for i in range(K):
            aid = int(dec_np[i])
            od_key = (int(self._od_pairs[aid, 0]), int(self._od_pairs[aid, 1]))
            paths = self._paths_per_od.get(od_key, [])
            n_paths = min(len(paths), self._top_k_paths)
            masks[i, :n_paths] = 1
        
        return masks

    def _apply_path_actions(self, action_indices: torch.Tensor, action_values: torch.Tensor):
        """Apply path-based actions: record chosen path and set next_edge."""
        agent_ids = action_indices.cpu().numpy()
        path_choices = action_values.cpu().numpy()
        
        for i in range(len(agent_ids)):
            aid = int(agent_ids[i])
            choice = int(path_choices[i])
            od_key = (int(self._od_pairs[aid, 0]), int(self._od_pairs[aid, 1]))
            paths = self._paths_per_od.get(od_key, [])
            
            if choice < len(paths):
                self._agent_chosen_path[aid] = paths[choice]
                self._agent_path_ptr[aid] = 0
                # Set next_edge to the first edge of the chosen path
                if len(paths[choice]) > 0:
                    self.dnl.next_edge[aid] = paths[choice][0]
            else:
                # Fallback: choose first path
                if paths:
                    self._agent_chosen_path[aid] = paths[0]
                    self._agent_path_ptr[aid] = 0
                    if len(paths[0]) > 0:
                        self.dnl.next_edge[aid] = paths[0][0]

    def _auto_route_path_agents(self, deciding: torch.Tensor) -> torch.Tensor:
        """For path-based mode: auto-set next_edge for agents following a chosen path.
        
        Agents that already have a chosen path don't need RL decisions —
        they just follow the path.  Remove them from the deciding tensor
        and set their next_edge directly.
        
        Returns:
            deciding: filtered tensor with only agents that need a new path choice.
        """
        if deciding.numel() == 0:
            return deciding
        
        dec_np = deciding.cpu().numpy()
        needs_decision = []
        
        for i in range(len(dec_np)):
            aid = int(dec_np[i])
            path = self._agent_chosen_path[aid]
            
            if path is not None:
                # Agent has a chosen path — advance pointer and set next_edge
                ptr = self._agent_path_ptr[aid]
                # Current node
                curr_edge = self.dnl.current_edge[aid].item()
                curr_to_node = self.dnl.edge_endpoints[curr_edge, 1].item()
                
                # Find the next edge in the path that starts at curr_to_node
                found = False
                for j in range(ptr, len(path)):
                    e_from = self.dnl.edge_endpoints[path[j], 0].item()
                    if e_from == curr_to_node:
                        self.dnl.next_edge[aid] = path[j]
                        self._agent_path_ptr[aid] = j + 1
                        found = True
                        break
                
                if not found:
                    # Agent is at the end of path or off-path, needs new decision
                    self._agent_chosen_path[aid] = None
                    needs_decision.append(aid)
            else:
                # No path chosen yet — needs a decision
                needs_decision.append(aid)
        
        if len(needs_decision) == len(dec_np):
            return deciding  # All need decisions
        
        if len(needs_decision) == 0:
            return torch.empty(0, device=self.dnl.device, dtype=torch.long)
        
        return torch.tensor(needs_decision, device=self.dnl.device, dtype=torch.long)

    def close(self):
        """Clean up resources."""
        self.agents = []

    def get_network_density(self) -> torch.Tensor:
        """Compute per-link density ratio (vehicles / storage_capacity).

        Returns a [E] float tensor where each entry is the occupancy ratio
        of the corresponding link.  Values range from 0 (empty) upward
        (can exceed 1.0 when the stuck-threshold forces agents onto a full
        link).
        """
        occupancy = self.dnl.edge_occupancy.float()
        capacity = self.dnl.storage_capacity.float().clamp(min=1.0)
        return occupancy / capacity
