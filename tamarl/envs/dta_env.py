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
from tamarl.envs.render import RenderState, build_renderer

AgentID = str
Observation = Dict[str, np.ndarray | int]
Action = int

# Simple ANSI helpers for colorful terminal rendering
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"


class DynamicTrafficAssignmentEnv(ParallelEnv):
    """A simple multi-agent traffic assignment environment using Parallel API."""

    metadata = {"render.modes": ["human", "ansi"], "name": "dta_par"}

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
        self._last_edge_taken: Dict[AgentID, Optional[int]] = {}
        self._last_rewards: Dict[AgentID, float] = {}
        self._cumulative_rewards: Dict[AgentID, float] = {}
        self.agent_status: Dict[AgentID, str] = {}
        self._last_travel_times: Optional[torch.Tensor] = None
        self._has_reset = False
        self._renderer = None

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
        self._last_edge_taken = {agent: None for agent in self.agents}
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.agent_status = {agent: "LIVE" for agent in self.possible_agents}
        self._last_travel_times = compute_link_travel_times(
            flows=self._flows,
            ff_time=self._ff_time,
            capacity=self._capacity,
            alpha=self.alpha,
            beta=self.beta,
        )
        self._has_reset = True

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
            self._last_edge_taken[agent] = edge_id
            self._last_rewards[agent] = rewards[agent]
            self._cumulative_rewards[agent] = self._cumulative_rewards.get(agent, 0.0) + rewards[agent]

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
        
        # Track travel times for rendering
        self._last_travel_times = travel_times

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
                penalty = -3600.0
                rewards[agent] += penalty
                self._last_rewards[agent] = rewards[agent]
                self._cumulative_rewards[agent] = self._cumulative_rewards.get(agent, 0.0) + penalty
            else:
                new_agents.append(agent)

        self.agents = new_agents

        # Update agent status bookkeeping
        for agent in current_agents:
            self._last_rewards[agent] = rewards.get(
                agent, self._last_rewards.get(agent, 0.0)
            )
            if terminations.get(agent, False):
                self.agent_status[agent] = "DONE"
            elif truncations.get(agent, False):
                self.agent_status[agent] = "TRUNC"
            else:
                self.agent_status[agent] = "LIVE"
        for agent in self.agents:
            self.agent_status[agent] = "LIVE"

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

    def _build_render_state(self) -> RenderState:
        """Assemble the data needed by the graphical renderer."""

        positions = (
            self.data.pos.detach().cpu().numpy()
            if hasattr(self.data, "pos") and self.data.pos is not None
            else np.zeros((int(self.data.num_nodes), 2), dtype=np.float32)
        )
        edge_index = (
            self.data.edge_index.detach().cpu().numpy()
            if hasattr(self.data, "edge_index") and self.data.edge_index is not None
            else np.zeros((2, 0), dtype=np.int64)
        )
        flows = self._flows.detach().cpu().numpy() if self._flows is not None else np.zeros(0, dtype=np.float32)
        travel_times = (
            self._last_travel_times.detach().cpu().numpy()
            if self._last_travel_times is not None
            else np.zeros_like(flows)
        )
        rewards_values = list(self._cumulative_rewards.values()) if self._cumulative_rewards else []
        mean_reward = float(np.mean(rewards_values)) if rewards_values else 0.0
        live_agents = {agent: self._current_nodes.get(agent, -1) for agent in self.agents}

        return RenderState(
            positions=positions,
            edge_index=edge_index,
            flows=flows,
            travel_times=travel_times,
            agent_nodes=live_agents,
            step=self._step_count,
            mean_reward=mean_reward,
        )

    def _render_ascii(self, verbose: bool = False) -> str:
        """Text-based rendering (used for ANSI mode or as fallback)."""

        live = sum(1 for status in self.agent_status.values() if status == "LIVE")
        done = sum(1 for status in self.agent_status.values() if status == "DONE")
        trunc = sum(1 for status in self.agent_status.values() if status == "TRUNC")
        rewards_values = list(self._cumulative_rewards.values()) if self._cumulative_rewards else []
        avg_cum_reward = float(np.mean(rewards_values)) if rewards_values else 0.0

        header = (
            f"🕒 Step {self._step_count} | "
            f"🧍 {GREEN}{live}{RESET} | "
            f"✅ {BLUE}{done}{RESET} | "
            f"⚠️ {YELLOW}{trunc}{RESET} | "
            f"🎯 avg cumul. r: {avg_cum_reward:.2f}"
        )

        separator = f"{BOLD}" + "-" * 60 + f"{RESET}"

        lines: List[str] = [separator, header, separator, ""]

        # Final recap when no live agents remain
        if live == 0 or not self.agents:
            lines.append("🏁 Episode finished!")
            lines.append(
                f"✅ Done: {BLUE}{done}{RESET} | ⚠️ Trunc: {YELLOW}{trunc}{RESET} | "
                f"Total steps: {self._step_count} | Avg cumulative reward: {avg_cum_reward:.2f}"
            )
            lines.append(separator)
            return "\n".join(lines)

        # Verbose output includes per-agent and per-edge details
        if verbose:
            total_agents = len(self.possible_agents)
            lines.append(f"👤 Agents (showing up to 3 of {total_agents}):")
            for agent in self.possible_agents[:3]:
                status = self.agent_status.get(agent, "TRUNC")
                status_display = {
                    "LIVE": f"{GREEN}🟢 LIVE{RESET}",
                    "DONE": f"{BLUE}✅ DONE{RESET}",
                    "TRUNC": f"{YELLOW}⚠️ TRUNC{RESET}",
                }.get(status, status)

                current_node = self._current_nodes.get(agent, "-")
                dest_node = self.destinations[int(agent.split('_')[-1])] if agent in self.possible_agents else "-"
                last_edge = self._last_edge_taken.get(agent)
                if last_edge is not None:
                    src, dst = edge_id_to_nodes(last_edge, self.data.edge_index)
                    last_edge_str = f"{src}→{dst}"
                else:
                    last_edge_str = "None"
                reward_val = self._last_rewards.get(agent, 0.0)
                lines.append(
                    f"  {agent} | {status_display:<10} | node {current_node} → dest {dest_node} | "
                    f"last: {last_edge_str} | r_step = {reward_val:.2f}"
                )

            lines.append("")
            lines.append(separator)
            lines.append("🛣️ Edges (top 10 by flow):")
            flows = self._flows if self._flows is not None else torch.zeros(0)
            if self._last_travel_times is not None and self._last_travel_times.shape[0] == flows.shape[0]:
                travel_times = self._last_travel_times
            else:
                travel_times = compute_link_travel_times(
                    flows=flows,
                    ff_time=self._ff_time,
                    capacity=self._capacity,
                    alpha=self.alpha,
                    beta=self.beta,
                )
            num_edges = flows.shape[0]
            indices = list(range(num_edges))
            sorted_edges = sorted(indices, key=lambda idx: float(flows[idx]), reverse=True)
            max_edges = min(10, len(sorted_edges))
            if max_edges == 0:
                lines.append("  (no edges)")
            else:
                flow_values = flows
                tt_values = travel_times if travel_times is not None else torch.zeros(num_edges)
                threshold = flow_values.mean().item() + flow_values.std().item() if num_edges > 0 else 0.0
                for rank in range(max_edges):
                    edge_id = sorted_edges[rank]
                    src, dst = edge_id_to_nodes(edge_id, self.data.edge_index)
                    flow_val = float(flow_values[edge_id].item())
                    tt_val = float(tt_values[edge_id].item())
                    color = RED if flow_val > threshold and threshold > 0 else ""
                    reset = RESET if color else ""
                    lines.append(
                        f"  {edge_id:>2}: {src}→{dst} | flow = {color}{flow_val:6.1f}{reset} | t = {tt_val:5.2f}"
                    )

            lines.append(separator)
            
        return "\n".join(lines)

    # Boilerplate -----------------------------------------------------------------
    def render(self, mode: str = "human") -> Optional[str]:
        if mode not in {"human", "ansi"}:
            raise NotImplementedError(f"Render mode {mode} is not supported.")

        if not self._has_reset:
            msg = "Environment not initialized. Call reset() first."
            if mode == "human":
                print(msg)
                return None
            return msg

        ascii_output = self._render_ascii()

        if mode == "ansi":
            print(ascii_output)
            return ascii_output

        if self._renderer is None:
            self._renderer = build_renderer()

        drawn = False
        try:
            state = self._build_render_state()
            drawn = self._renderer.render(state) if self._renderer is not None else False
        except Exception:
            drawn = False

        if not drawn:
            print(ascii_output)
        return None

    def close(self) -> None:
        self.agents = []
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        self._renderer = None


__all__ = ["DynamicTrafficAssignmentEnv"]
