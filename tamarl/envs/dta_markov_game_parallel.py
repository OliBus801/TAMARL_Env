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
        """Initialize the DTA Markov Game environment.
        
        Args:
            scenario_path: path to scenario folder with network.xml + population.xml
            population_filter: substring to match population file (e.g. '100')
            timestep: simulation timestep dt in seconds
            scale_factor: scale factor for network capacities
            max_steps: maximum number of simulation ticks per episode
            device: 'cpu' or 'cuda'
            seed: random seed for reproducibility
            stuck_threshold: stuck threshold for DNL simulator
        """
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
        
        # Track cumulative rewards per episode
        self._cumulative_rewards: Dict[str, float] = {}
        self._ticks_since_last_step = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_builder.observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(self.dnl.max_out_degree)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset the environment for a new episode.
        
        Returns:
            observations: dict of agent_id → observation array
            infos: dict of agent_id → info dict (no action_mask at reset, agents are waiting)
        """
        if seed is not None:
            self._seed = seed
        
        self.dnl.reset()
        if seed is not None:
            self.dnl.rng.manual_seed(seed)
        
        # All agents start alive
        self.agents = list(self.possible_agents)
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._ticks_since_last_step = 0
        self._rewarder.reset()
        
        # Advance simulation until first decision occurs
        # (agents need to depart, enter first link, traverse it, then decide at buffer)
        self._advance_to_next_decisions()
        
        # Build observations for deciding agents
        deciding = self._scheduler.get_deciding_agents()
        observations = self._obs_builder.build_observations(deciding)
        
        # For agents that are not yet deciding (still waiting/traveling), 
        # provide initial observations
        initial_obs = self._obs_builder.build_initial_observations()
        for agent_id in self.agents:
            if agent_id not in observations:
                observations[agent_id] = initial_obs.get(agent_id, 
                    np.zeros(self._obs_builder.obs_size, dtype=np.float32))
        
        # Infos with action masks
        infos = {}
        action_masks = self._action_mgr.get_action_masks(deciding)
        for agent_id in self.agents:
            info = {}
            if agent_id in action_masks:
                info["action_mask"] = action_masks[agent_id]
            else:
                # Not deciding yet — provide all-zeros mask
                info["action_mask"] = np.zeros(self.dnl.max_out_degree, dtype=np.int8)
            
            idx = int(agent_id.split("_")[-1])
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
        """Execute one macro-step: apply actions, advance DNL to next decision event.
        
        Args:
            actions: dict of agent_id → action index (only for deciding agents)
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        current_agents = list(self.agents)
        
        # 1. Apply actions for deciding agents
        if actions:
            self._action_mgr.apply_actions(actions)
        
        # 2. Advance DNL until next decision event (macro-step)
        n_ticks = self._advance_to_next_decisions()
        
        # 3. Compute rewards
        rewards = self._rewarder.compute_step_rewards(set(current_agents), n_ticks)
        
        # 4. Determine terminations and truncations
        terminations = {}
        truncations = {}
        
        if current_agents:
            active_indices = [int(a.split("_")[-1]) for a in current_agents]
            active_tensor = torch.tensor(active_indices, device=self.dnl.device, dtype=torch.long)
            statuses = self.dnl.status[active_tensor].cpu().numpy()
            
            for i, agent_id in enumerate(current_agents):
                terminations[agent_id] = bool(statuses[i] == 3)  # Arrived at destination
                truncations[agent_id] = False
        
        # Check max steps truncation
        if self.dnl.current_step >= self._max_steps:
            for agent_id in current_agents:
                if not terminations[agent_id]:
                    truncations[agent_id] = True
        
        # 5. Update cumulative rewards
        for agent_id in current_agents:
            self._cumulative_rewards[agent_id] = (
                self._cumulative_rewards.get(agent_id, 0.0) + rewards.get(agent_id, 0.0)
            )
        
        # 6. Remove finished agents
        self.agents = [
            a for a in current_agents
            if not terminations[a] and not truncations[a]
        ]
        
        # 7. Build observations for remaining agents
        deciding = self._scheduler.get_deciding_agents()
        observations = self._obs_builder.build_observations(deciding)
        
        # For non-deciding remaining agents, build a current obs
        if self.agents:
            agent_indices = np.array([int(a.split("_")[-1]) for a in self.agents])
            idx_tensor = torch.from_numpy(agent_indices).to(self.dnl.device)
            statuses = self.dnl.status[idx_tensor].cpu().numpy()
            
            needs_obs_mask = np.array([a not in observations for a in self.agents])
            valid_mask = needs_obs_mask & ((statuses == 1) | (statuses == 2))
            
            if valid_mask.any():
                valid_idx_tensor = torch.from_numpy(agent_indices[valid_mask]).to(self.dnl.device)
                curr_edges = self.dnl.current_edge[valid_idx_tensor]
                nodes = self.dnl.edge_endpoints[curr_edges, 1].cpu().numpy()
                c_legs = self.dnl.current_leg[valid_idx_tensor]
                dests = self.dnl.destinations[valid_idx_tensor, c_legs].cpu().numpy()
                norm_time = self.dnl.current_step / self._max_steps
                
                valid_agent_ids = [self.agents[i] for i in np.where(valid_mask)[0]]
                for i, agent_id in enumerate(valid_agent_ids):
                    obs = np.zeros(self._obs_builder.obs_size, dtype=np.float32)
                    obs[0] = float(nodes[i])
                    obs[1] = float(dests[i])
                    obs[2] = norm_time
                    observations[agent_id] = obs
                    
            zero_mask = needs_obs_mask & ~valid_mask
            if zero_mask.any():
                zero_agent_ids = [self.agents[i] for i in np.where(zero_mask)[0]]
                for agent_id in zero_agent_ids:
                    observations[agent_id] = np.zeros(self._obs_builder.obs_size, dtype=np.float32)
        
        # 8. Build infos with action masks
        infos = {}
        action_masks = self._action_mgr.get_action_masks(deciding)
        for agent_id in current_agents:
            info = {}
            if agent_id in action_masks:
                info["action_mask"] = action_masks[agent_id]
            else:
                info["action_mask"] = np.zeros(self.dnl.max_out_degree, dtype=np.int8)
            
            idx = int(agent_id.split("_")[-1])
            info["curr_leg"] = int(self.dnl.current_leg[idx].item())
            infos[agent_id] = info
        
        return observations, rewards, terminations, truncations, infos

    def _advance_to_next_decisions(self) -> int:
        """Advance DNL tick-by-tick until a decision event occurs or episode ends.
        
        A decision event is when at least one agent needs to choose its next edge
        (status=2, wakeup_time <= current_step, next_edge == -1).
        
        Returns:
            Number of ticks advanced.
        """
        ticks = 0
        max_idle = self._max_steps  # Safety limit
        
        while ticks < max_idle:
            # Check if we already have deciding agents
            if ticks > 0 and self._scheduler.has_deciding_agents():
                break
            
            # Check if all agents are done
            all_done = (self.dnl.status >= 3).all().item()
            if all_done:
                break
            
            # Check max steps
            if self.dnl.current_step >= self._max_steps:
                break
            
            # Advance one tick
            self.dnl.step()
            ticks += 1
        
        self._ticks_since_last_step = ticks
        return ticks

    def close(self):
        """Clean up resources."""
        self.agents = []
