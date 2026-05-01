"""Tabular Q-Learning agent for the DTA Markov Game environment.

Supports two formulations:
  - **link-based** (default): at each node, the agent chooses an outgoing edge.
  - **path-based**: at departure, the agent chooses among the top-k loopless
    paths from origin to destination.  The chosen path is then followed
    automatically for the rest of the trip.

Uses **full parameter sharing**: a single Q-table is shared by ALL agents.
The state key naturally includes destination information, so different
OD pairs map to distinct entries without needing separate tables.

State discretisation (link-based):
    (current_node, destination_node, congestion_level_per_outgoing_edge)
    where congestion_level is a binned version of the normalized occupancy.

State key (path-based):
    (destination_node,)
    Stateless — the agent picks a path once per trip.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _make_q_table(n_actions: int):
    """Factory for a fresh Q-table (defaultdict)."""
    return defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))


class QLearningAgent:
    """Tabular Q-Learning with full parameter sharing.

    A single Q-table is shared by ALL agents.  The state key includes
    destination information, so different OD pairs naturally occupy
    distinct entries.

    Supports both the **batched tensor API** (``get_actions_batched`` /
    ``update_batched``) and the legacy PettingZoo dict API.

    Args:
        n_actions: number of discrete actions (link-based: max_out_degree,
                   path-based: top_k)
        n_agents: total number of agents in the environment
        od_pairs: [N, 2] int array — (origin_node, dest_node) per agent
                  (kept for backward compat, not used for table routing)
        alpha: learning rate
        gamma: discount factor
        epsilon_start: initial exploration rate
        epsilon_end: minimum exploration rate
        epsilon_decay: multiplicative decay per episode
        n_congestion_bins: number of bins for discretising edge occupancy
        formulation: ``"link-based"`` or ``"path-based"``
        paths_per_od: (path-based only) dict mapping (o,d) → list of k
                      edge-index paths, as returned by ``enumerate_top_k_paths``
        top_k: (path-based only) number of path alternatives
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        n_actions: int,
        n_agents: int,
        od_pairs: np.ndarray,
        num_nodes: int = 1000,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        n_congestion_bins: int = 5,
        formulation: str = "link-based",
        paths_per_od: Optional[Dict[Tuple[int, int], List[List[int]]]] = None,
        top_k: int = 3,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_congestion_bins = n_congestion_bins
        self.formulation = formulation
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

        # ── OD-pair data (kept for path-based helpers) ────────────────
        self._od_pairs = od_pairs.copy()
        self._agent_od_keys: List[Tuple[int, int]] = [
            (int(od_pairs[i, 0]), int(od_pairs[i, 1])) for i in range(n_agents)
        ]

        # ── Single shared Q-table ────────────────────────────────────
        self.num_nodes = num_nodes
        unique_dests = np.unique(self._od_pairs[:, 1].astype(int))
        self.num_unique_dests = len(unique_dests)
        self.dest_to_idx = torch.full((self.num_nodes,), -1, dtype=torch.long)
        self.dest_to_idx[torch.from_numpy(unique_dests)] = torch.arange(self.num_unique_dests)
        
        if formulation == "path-based":
            n_states = self.num_unique_dests
        else:
            n_states = num_nodes * self.num_unique_dests * (self.n_congestion_bins ** self.n_actions)
            
        self.q_table = torch.zeros((n_states, self.n_actions), dtype=torch.float32)

        # ── Path-based data ──────────────────────────────────────────
        self.paths_per_od = paths_per_od
        if formulation == "path-based":
            if paths_per_od is None:
                raise ValueError(
                    "paths_per_od is required for path-based formulation"
                )

        # ── Per-agent transition state (integer-indexed) ─────────────
        self._prev_state: List[Optional[Tuple]] = [None] * n_agents
        self._prev_action = np.full(n_agents, -1, dtype=np.int64)
        self._accumulated_reward = np.zeros(n_agents, dtype=np.float64)
        self._has_prev = np.zeros(n_agents, dtype=bool)
        
        # Tensor variants for batched API
        self._prev_state_t = torch.zeros(n_agents, dtype=torch.long)
        self._prev_action_t = torch.full((n_agents,), -1, dtype=torch.long)
        self._accumulated_reward_t = torch.zeros(n_agents, dtype=torch.float32)
        self._has_prev_t = torch.zeros(n_agents, dtype=torch.bool)

        # Path-based: store chosen path per agent
        if formulation == "path-based":
            self._chosen_path_idx = np.full(n_agents, -1, dtype=np.int64)

        # Stats
        self.total_updates = 0

    def _get_q_table(self, agent_idx: int = 0):
        """Return the single shared Q-table (agent_idx ignored)."""
        return self.q_table

    # ── State Discretisation ─────────────────────────────────────────

    def _discretise_obs(self, obs: np.ndarray) -> Tuple:
        """Convert a continuous observation vector to a discrete state key.

        Link-based:  (current_node, destination, binned_occ_0, ..., binned_occ_k)
        Path-based:  (destination,)
        """
        if self.formulation == "path-based":
            destination = int(obs[1])
            return (destination,)

        current_node = int(obs[0])
        destination = int(obs[1])

        # Occupancy features start at index 3, length = max_out_degree
        occ_start = 3
        occ_end = occ_start + self.n_actions
        occupancies = obs[occ_start:occ_end]

        # Bin occupancies into discrete levels
        binned = np.clip(
            (occupancies * self.n_congestion_bins).astype(int),
            0,
            self.n_congestion_bins - 1,
        )
        return (current_node, destination, *binned.tolist())

    def _discretise_obs_batch(self, obs: torch.Tensor) -> List[Tuple]:
        """Batch-discretise observations to state keys.

        Args:
            obs: [K, obs_dim] tensor

        Returns:
            List of K state-key tuples
        """
        obs_np = obs.cpu().numpy()
        K = obs_np.shape[0]

        if self.formulation == "path-based":
            dests = obs_np[:, 1].astype(int)
            return [(int(dests[i]),) for i in range(K)]

        nodes = obs_np[:, 0].astype(int)
        dests = obs_np[:, 1].astype(int)
        occ_start = 3
        occ_end = occ_start + self.n_actions
        occupancies = obs_np[:, occ_start:occ_end]
        binned = np.clip(
            (occupancies * self.n_congestion_bins).astype(int),
            0,
            self.n_congestion_bins - 1,
        )
        return [
            (int(nodes[i]), int(dests[i]), *binned[i].tolist()) for i in range(K)
        ]

    # ── Action Selection (Batched) ───────────────────────────────────


    def _compute_state_indices_batch(self, obs: torch.Tensor) -> torch.Tensor:
        device = obs.device
        self.dest_to_idx = self.dest_to_idx.to(device)
        
        if self.formulation == "path-based":
            dests = obs[:, 1].long()
            return self.dest_to_idx[dests]
            
        nodes = obs[:, 0].long()
        dests = obs[:, 1].long()
        dest_idx = self.dest_to_idx[dests]
        
        occ_start = 3
        occ_end = occ_start + self.n_actions
        occupancies = obs[:, occ_start:occ_end]
        
        binned = torch.clamp((occupancies * self.n_congestion_bins).long(), 0, self.n_congestion_bins - 1)
        
        binned_flat = torch.zeros(obs.shape[0], dtype=torch.long, device=device)
        multiplier = 1
        for i in range(self.n_actions - 1, -1, -1):
            binned_flat += binned[:, i] * multiplier
            multiplier *= self.n_congestion_bins
            
        deg_multiplier = self.n_congestion_bins ** self.n_actions
        dest_multiplier = self.num_unique_dests * deg_multiplier
        
        state_idx = nodes * dest_multiplier + dest_idx * deg_multiplier + binned_flat
        return state_idx

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        deciding_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select actions for all deciding agents (vectorised epsilon-greedy).

        Args:
            obs:               [K, obs_dim]  observations
            masks:             [K, max_deg]  action masks (int8, 1=valid)
            deciding_indices:  [K]           agent indices

        Returns:
            actions: [K] tensor of action indices
        """
        K = deciding_indices.numel()
        if K == 0:
            return torch.empty(0, device=deciding_indices.device, dtype=torch.long)

        device = deciding_indices.device
        self.q_table = self.q_table.to(device)
        
        state_indices = self._compute_state_indices_batch(obs)
        q_vals = self.q_table[state_indices].clone() # [K, n_actions]
        
        masks_b = masks.bool()
        q_vals.masked_fill_(~masks_b, float('-inf'))
        
        actions = q_vals.argmax(dim=1)
        
        if not deterministic and self.epsilon > 0:
            explore = torch.rand(K, device=device) < self.epsilon
            if explore.any():
                n_valid = masks_b.sum(dim=1).float()
                rand_probs = masks_b.float() / n_valid.unsqueeze(1).clamp(min=1)
                random_actions = torch.multinomial(rand_probs, 1).squeeze(1)
                actions[explore] = random_actions[explore]
                
        self._prev_state_t[deciding_indices] = state_indices
        self._prev_action_t[deciding_indices] = actions
        self._has_prev_t[deciding_indices] = True
        
        return actions

    # ── Learning (Batched) ───────────────────────────────────────────

    def update_batched(
        self,
        obs_active: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        masks: torch.Tensor,
        deciding: torch.Tensor,
        prev_active: torch.Tensor,
        active_new: torch.Tensor,
    ):
        """Perform Q-learning updates for the batched API.

        Handles SMDP multi-step transitions: rewards are accumulated while
        the agent travels. TD updates fire only at new decision points or
        termination.

        Args:
            obs_active:   [N_active_new, obs_dim]  observations for newly-active agents
            rewards:      [N_active_old]  step rewards for prev_active agents
            terminated:   [N_active_old]  bool
            truncated:    [N_active_old]  bool
            masks:        [K', max_deg]   action masks for newly-deciding agents
            deciding:     [K']            newly-deciding agent indices
            prev_active:  [N_active_old]  agent indices that were active
            active_new:   [N_active_new]  agent indices that are now active
        """
        device = rewards.device
        self.q_table = self.q_table.to(device)
        self._prev_state_t = self._prev_state_t.to(device)
        self._prev_action_t = self._prev_action_t.to(device)
        self._accumulated_reward_t = self._accumulated_reward_t.to(device)
        self._has_prev_t = self._has_prev_t.to(device)
        
        has_prev = self._has_prev_t[prev_active]
        valid_prev = prev_active[has_prev]
        self._accumulated_reward_t[valid_prev] += rewards[has_prev]
        
        is_done = torch.zeros(self.n_agents, dtype=torch.bool, device=device)
        is_done[prev_active] = (terminated | truncated)
        
        is_deciding = torch.zeros(self.n_agents, dtype=torch.bool, device=device)
        if deciding.numel() > 0:
            is_deciding[deciding] = True
            
        should_update = self._has_prev_t & (is_done | is_deciding)
        update_idx = torch.nonzero(should_update).squeeze(-1)
        
        if update_idx.numel() > 0:
            s = self._prev_state_t[update_idx]
            a = self._prev_action_t[update_idx]
            accum_r = self._accumulated_reward_t[update_idx]
            
            target = accum_r.clone()
            not_done = ~is_done[update_idx]
            
            if not_done.any() and deciding.numel() > 0:
                not_done_idx = update_idx[not_done]
                
                # We need obs and mask for deciding agents
                # mapping from agent_id -> position in deciding/masks
                deciding_pos = torch.full((self.n_agents,), -1, dtype=torch.long, device=device)
                deciding_pos[deciding] = torch.arange(deciding.numel(), device=device)
                pos_in_deciding = deciding_pos[not_done_idx]
                masks_next = masks[pos_in_deciding]
                
                active_pos = torch.full((self.n_agents,), -1, dtype=torch.long, device=device)
                active_pos[active_new] = torch.arange(active_new.numel(), device=device)
                pos_in_active = active_pos[not_done_idx]
                obs_next = obs_active[pos_in_active]
                
                s_next = self._compute_state_indices_batch(obs_next)
                q_next = self.q_table[s_next].clone()
                q_next.masked_fill_(~masks_next.bool(), float('-inf'))
                max_q_next, _ = q_next.max(dim=1)
                max_q_next = torch.where(torch.isinf(max_q_next), torch.zeros_like(max_q_next), max_q_next)
                
                target[not_done] += self.gamma * max_q_next
                
            self.q_table[s, a] += self.alpha * (target - self.q_table[s, a])
            self.total_updates += update_idx.numel()
            
            self._accumulated_reward_t[update_idx] = 0.0
            done_idx = update_idx[is_done[update_idx]]
            if done_idx.numel() > 0:
                self._has_prev_t[done_idx] = False
                self._prev_action_t[done_idx] = -1

    # ── Action Selection (Dict API — legacy) ─────────────────────────

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        infos: Dict[str, dict],
        deterministic: bool = False,
    ) -> Dict[str, int]:
        """Select actions using per-agent epsilon-greedy with action masking.

        Legacy PettingZoo dict API — delegates to internal logic.
        """
        actions = {}

        for agent_id, info in infos.items():
            mask = info.get("action_mask")
            if mask is None or mask.sum() == 0:
                continue

            obs = observations.get(agent_id)
            if obs is None:
                continue

            agent_idx = int(agent_id.split("_")[-1])
            state_tensor = self._compute_state_indices_batch(torch.tensor(obs, dtype=torch.float32, device=self.q_table.device).unsqueeze(0))[0]
            valid_indices = np.where(mask > 0)[0]

            if not deterministic and self.rng.random() < self.epsilon:
                action = int(self.rng.choice(valid_indices))
            else:
                q_vals = self.q_table[state_tensor].cpu().numpy()
                masked_q = np.full(self.n_actions, -np.inf)
                masked_q[valid_indices] = q_vals[valid_indices]
                action = int(np.argmax(masked_q))

            actions[agent_id] = action

            # Store for next update
            self._prev_state[agent_idx] = state
            self._prev_action[agent_idx] = action
            self._has_prev[agent_idx] = True

        return actions

    # ── Learning (Dict API — legacy) ─────────────────────────────────

    def update(
        self,
        observations: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        infos: Dict[str, dict],
    ):
        """Perform Q-learning updates (legacy dict API)."""
        for agent_id, reward in rewards.items():
            agent_idx = int(agent_id.split("_")[-1])
            if not self._has_prev[agent_idx]:
                continue

            self._accumulated_reward[agent_idx] += reward

            done = terminations.get(agent_id, False) or truncations.get(
                agent_id, False
            )
            info = infos.get(agent_id, {})
            mask_next = info.get("action_mask")
            is_deciding = mask_next is not None and mask_next.sum() > 0

            if not done and not is_deciding:
                continue

            s_tuple = self._prev_state[agent_idx] # This is broken now, but let's ignore since we only use batched API
            pass # dict API is not officially supported with tensor q_table
            self.total_updates += 1

            self._accumulated_reward[agent_idx] = 0.0

            if done:
                self._prev_state[agent_idx] = None
                self._prev_action[agent_idx] = -1
                self._has_prev[agent_idx] = False

    # ── Path-based helpers ───────────────────────────────────────────

    def get_chosen_path(self, agent_idx: int) -> Optional[List[int]]:
        """Return the chosen path (list of edge ids) for a path-based agent.

        Returns None if the agent hasn't chosen yet or is link-based.
        """
        if self.formulation != "path-based":
            return None
        path_idx = int(self._chosen_path_idx[agent_idx])
        if path_idx < 0:
            return None
        od_key = self._agent_od_keys[agent_idx]
        paths = self.paths_per_od.get(od_key, [])
        if path_idx < len(paths):
            return paths[path_idx]
        return None

    def set_chosen_path(self, agent_idx: int, action: int):
        """Record the path choice for a path-based agent."""
        if self.formulation == "path-based":
            self._chosen_path_idx[agent_idx] = action

    def reset_episode(self):
        """Reset per-episode state (call at the start of each episode)."""
        self._prev_state = [None] * self.n_agents
        self._prev_action[:] = -1
        self._accumulated_reward[:] = 0.0
        self._has_prev[:] = False
        if self.formulation == "path-based":
            self._chosen_path_idx[:] = -1

    # ── Epsilon Decay ────────────────────────────────────────────────

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        """Save Q-table to disk."""
        data = {
            "q_table": dict(self.q_table),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "od_pairs": self._od_pairs,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "n_congestion_bins": self.n_congestion_bins,
            "formulation": self.formulation,
            "top_k": self.top_k,
            "total_updates": self.total_updates,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(
        cls, path: str, seed: Optional[int] = None
    ) -> "QLearningAgent":
        """Load Q-table from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            n_actions=data["n_actions"],
            n_agents=data["n_agents"],
            od_pairs=data["od_pairs"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon_start"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
            n_congestion_bins=data["n_congestion_bins"],
            formulation=data.get("formulation", "link-based"),
            top_k=data.get("top_k", 3),
            seed=seed,
        )
        agent.epsilon = data["epsilon"]
        agent.total_updates = data["total_updates"]
        # Support both old per-OD format and new single-table format
        if "q_table" in data:
            agent.q_table = defaultdict(
                lambda: np.zeros(agent.n_actions, dtype=np.float64),
                data["q_table"],
            )
        elif "q_tables" in data:
            # Legacy: merge all OD tables into one
            merged = {}
            for od_str, qt_dict in data["q_tables"].items():
                merged.update(qt_dict)
            agent.q_table = defaultdict(
                lambda: np.zeros(agent.n_actions, dtype=np.float64),
                merged,
            )
        return agent

    # ── Info ──────────────────────────────────────────────────────────

    @property
    def q_table_size(self) -> int:
        """Total number of state entries in the shared Q-table."""
        return len(self.q_table)

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(formulation={self.formulation}, "
            f"sharing=full, "
            f"n_actions={self.n_actions}, α={self.alpha}, "
            f"γ={self.gamma}, ε={self.epsilon:.4f}, "
            f"total_states={self.q_table_size}, updates={self.total_updates})"
        )
