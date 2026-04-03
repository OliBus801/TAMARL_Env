"""PettingZoo ParallelEnv wrapper for TorchDNLMATSim in RL mode.

This environment wraps the DNL simulator as a Markov Game where each
agent independently chooses outgoing edges at decision points (capacity buffer).
"""
from __future__ import annotations

import functools
from typing import Dict, Optional, Set, Tuple

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_builder.observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
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
            action_values:  [K] action values (local edge indices)
            
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
        
        n_active = self._active_indices.numel()
        obs_dim = self._obs_builder.obs_size
        
        if n_active == 0:
            obs_active = torch.empty((0, obs_dim), device=self.dnl.device)
            masks = torch.empty((0, self.dnl.max_out_degree), device=self.dnl.device, dtype=torch.int8)
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
                norm_time = self.dnl.current_step / self._max_steps
                
                obs_active[needs_obs, 0] = nodes
                obs_active[needs_obs, 1] = dests
                obs_active[needs_obs, 2] = norm_time
        
        # 8. Action masks for deciding agents
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

    def close(self):
        """Clean up resources."""
        self.agents = []
