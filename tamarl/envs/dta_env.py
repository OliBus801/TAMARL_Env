"""PettingZoo Parallel environment for dynamic traffic assignment."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import functools
import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

from tamarl.core.network import NetworkMetadata, build_out_edges_per_node, edge_id_to_nodes
from tamarl.core.network_loading import (
    compute_link_travel_times,
    compute_step_rewards,
    update_cumulative_flows,
)

AgentID = str
Observation = Dict[str, np.ndarray | int]
Action = int


class DynamicTrafficAssignmentEnv(ParallelEnv):
    """A simple multi-agent traffic assignment environment using Parallel API."""

    metadata = {"render.modes": ["human"], "name": "dta_par"}

    def __init__(
        self,
        data,
        origins: List[int],
        destinations: List[int],
        max_steps: int = 20,
        alpha: float = 0.15,
        beta: float = 4.0,
    ) -> None:
        super().__init__()
        self.data = data
        self.origins = origins
        self.destinations = destinations
        self.max_steps = max_steps
        self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.beta = torch.tensor(beta, dtype=torch.float)

        self.possible_agents: List[AgentID] = [f"agent_{i}" for i in range(len(origins))]

        self.network_meta: NetworkMetadata = build_out_edges_per_node(self.data)
        self._ff_time = self.data.edge_attr[:, 0]
        self._capacity = self.data.edge_attr[:, 1]

        self._flows = torch.zeros(self.data.edge_index.shape[1], dtype=torch.float)
        self._step_count = 0

        self.agents: List[AgentID] = []
        self._current_nodes: Dict[AgentID, int] = {}
        self._visited_edges: Dict[AgentID, List[int]] = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[AgentID, Observation], Dict[AgentID, dict]]:
        if seed is not None:
            np.random.seed(seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._flows = torch.zeros_like(self._flows)
        self._visited_edges = {agent: [] for agent in self.agents}
        self._current_nodes = {
            agent: self.origins[int(agent.split("_")[-1])] for agent in self.agents
        }

        observations = {agent: self._build_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    # Observation and action spaces -------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        return spaces.Dict(
            {
                "current_node": spaces.Discrete(self.data.num_nodes),
                "destination_node": spaces.Discrete(self.data.num_nodes),
                "action_mask": spaces.MultiBinary(self.network_meta.max_out_degree),
                "visited_edges": spaces.MultiBinary(self.data.edge_index.shape[1]),
            }
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        return spaces.Discrete(self.network_meta.max_out_degree)

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    # Core step logic --------------------------------------------------------------
    def step(
        self, actions: Dict[AgentID, Action]
    ) -> Tuple[
        Dict[AgentID, Observation],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, dict],
    ]:
        # If no actions provided, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        current_agents = list(self.agents)
        chosen_edges: Dict[AgentID, int] = {}

        for agent, action in actions.items():
            if agent not in current_agents:
                continue
            node = self._current_nodes[agent]
            local_edges = self.network_meta.out_edges_per_node[node]
            if action >= len(local_edges):
                raise ValueError(f"Invalid action {action} for node {node}")
            edge_id = local_edges[action]
            chosen_edges[agent] = edge_id
            self._visited_edges[agent].append(edge_id)

        if chosen_edges:
            edges_tensor = torch.tensor(list(chosen_edges.values()), dtype=torch.long)
            self._flows = update_cumulative_flows(
                self._flows, edges_tensor, num_edges=self.data.edge_index.shape[1]
            )
            travel_times = compute_link_travel_times(
                flows=self._flows,
                ff_time=self._ff_time,
                capacity=self._capacity,
                alpha=self.alpha,
                beta=self.beta,
            )
            rewards_tensor = compute_step_rewards(edges_tensor, travel_times)
        else:
            travel_times = compute_link_travel_times(
                flows=self._flows,
                ff_time=self._ff_time,
                capacity=self._capacity,
                alpha=self.alpha,
                beta=self.beta,
            )
            rewards_tensor = torch.tensor([], dtype=torch.float)

        rewards: Dict[AgentID, float] = {agent: 0.0 for agent in self.agents}
        terminations: Dict[AgentID, bool] = {
            agent: False for agent in self.agents
        }
        truncations: Dict[AgentID, bool] = {agent: False for agent in self.agents}
        infos: Dict[AgentID, dict] = {agent: {} for agent in self.agents}

        # Move agents
        for idx, agent in enumerate(chosen_edges.keys()):
            edge_id = chosen_edges[agent]
            _, target = edge_id_to_nodes(edge_id, self.data.edge_index)
            self._current_nodes[agent] = target
            rewards[agent] = float(rewards_tensor[idx].item())

        # Determine terminations
        for agent in current_agents:
            agent_idx = int(agent.split("_")[-1])
            if self._current_nodes[agent] == self.destinations[agent_idx]:
                terminations[agent] = True

        # Check for max steps truncation
        self._step_count += 1
        if self._step_count >= self.max_steps:
            for agent in current_agents:
                if not terminations[agent]:
                    truncations[agent] = True
        

        # Remove finished agents
        self.agents = [
            agent
            for agent in current_agents
            if not terminations[agent] and not truncations[agent]
        ]

        observations: Dict[AgentID, Observation] = {
            agent: self._build_observation(agent) for agent in self.agents
        }

        # Check for agents in a dead-end (no available actions)
        # If so, truncate them, add a big negative reward, and remove them
        new_agents = []
        for agent in self.agents:
            if observations[agent]["action_mask"].sum().item() == 0:
                truncations[agent] = True
                rewards[agent] += -3600.0
            else:
                new_agents.append(agent)

        self.agents = new_agents


        return observations, rewards, terminations, truncations, infos

    # Helpers ----------------------------------------------------------------------
    def _build_observation(self, agent: AgentID) -> Observation:
        agent_idx = int(agent.split("_")[-1])
        current_node = self._current_nodes[agent]
        dest_node = self.destinations[agent_idx]
        action_mask = self._action_mask_for_node(current_node, self._visited_edges[agent])
        visited = np.zeros(self.data.edge_index.shape[1], dtype=int)
        if self._visited_edges[agent]:
            visited[self._visited_edges[agent]] = 1

        return {
            "current_node": int(current_node),
            "destination_node": int(dest_node),
            "action_mask": action_mask,
            "visited_edges": visited,
        }

    def _action_mask_for_node(
        self, node: int, visited_edges: List[int]
    ) -> np.ndarray:
        mask = np.zeros(self.network_meta.max_out_degree, dtype=np.int8)
        edges = self.network_meta.out_edges_per_node[node]
        for idx, edge_id in enumerate(edges):
            if edge_id not in visited_edges:
                mask[idx] = 1
        return mask

    # Boilerplate -----------------------------------------------------------------
    def render(self, mode: str = "human") -> None:
        return None

    def close(self) -> None:
        self.agents = []


__all__ = ["DynamicTrafficAssignmentEnv"]
