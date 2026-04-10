"""Tabular Q-Learning agent for the DTA Markov Game environment.

Supports two formulations:
  - **link-based** (default): at each node, the agent chooses an outgoing edge.
  - **path-based**: at departure, the agent chooses among the top-k loopless
    paths from origin to destination.  The chosen path is then followed
    automatically for the rest of the trip.

In both formulations, agents with the same Origin-Destination (OD) pair share a
single Q-table (parameter sharing), which is standard in traffic assignment
literature and yields faster convergence with fewer tables.

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
    """Tabular Q-Learning with OD-pair shared Q-tables.

    All agents that share the same (origin, destination) pair use the same
    Q-table.  This dramatically reduces the number of tables and accelerates
    both learning and computation.

    Supports both the **batched tensor API** (``get_actions_batched`` /
    ``update_batched``) and the legacy PettingZoo dict API.

    Args:
        n_actions: number of discrete actions (link-based: max_out_degree,
                   path-based: top_k)
        n_agents: total number of agents in the environment
        od_pairs: [N, 2] int array — (origin_node, dest_node) per agent
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

        # ── OD-pair mapping ──────────────────────────────────────────
        # od_pairs: [N, 2] — (origin_node, dest_node) per agent
        self._od_pairs = od_pairs.copy()
        # Map each agent to its OD key (tuple)
        self._agent_od_keys: List[Tuple[int, int]] = [
            (int(od_pairs[i, 0]), int(od_pairs[i, 1])) for i in range(n_agents)
        ]
        # Unique OD pairs
        self._unique_ods = list(set(self._agent_od_keys))

        # ── Q-tables: keyed by OD pair ───────────────────────────────
        self.q_tables: Dict[Tuple[int, int], Dict[Tuple, np.ndarray]] = {}
        for od in self._unique_ods:
            self.q_tables[od] = defaultdict(
                lambda: np.zeros(self.n_actions, dtype=np.float64)
            )

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

        # Path-based: store chosen path per agent
        if formulation == "path-based":
            self._chosen_path_idx = np.full(n_agents, -1, dtype=np.int64)

        # Stats
        self.total_updates = 0

    def _get_q_table(self, agent_idx: int):
        """Get the shared Q-table for an agent (by its OD pair)."""
        od_key = self._agent_od_keys[agent_idx]
        return self.q_tables[od_key]

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

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        deciding_indices: torch.Tensor,
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
        agent_ids = deciding_indices.cpu().numpy()

        # Batch discretise
        state_keys = self._discretise_obs_batch(obs)

        # Vectorised epsilon-greedy decision
        rand_vals = self.rng.random(K)
        explore_mask = rand_vals < self.epsilon

        # Prepare masks for random sampling
        masks_np = masks.cpu().numpy().astype(np.float64)

        actions = np.zeros(K, dtype=np.int64)

        for i in range(K):
            aid = int(agent_ids[i])
            state = state_keys[i]
            q_table = self._get_q_table(aid)
            mask_i = masks_np[i]

            if explore_mask[i]:
                # Random among valid actions
                valid = np.where(mask_i > 0)[0]
                if len(valid) > 0:
                    actions[i] = self.rng.choice(valid)
            else:
                # Greedy among valid actions
                q_vals = q_table[state]
                masked_q = np.full(self.n_actions, -np.inf)
                valid = np.where(mask_i > 0)[0]
                if len(valid) > 0:
                    masked_q[valid] = q_vals[valid]
                    actions[i] = int(np.argmax(masked_q))

            # Store transition state
            self._prev_state[aid] = state
            self._prev_action[aid] = actions[i]
            self._has_prev[aid] = True

        return torch.from_numpy(actions).to(device)

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
        rewards_np = rewards.cpu().numpy()
        prev_ids = prev_active.cpu().numpy()
        done_mask = (terminated | truncated).cpu().numpy()

        # 1. Accumulate rewards for all previously-active agents
        for i in range(len(prev_ids)):
            aid = int(prev_ids[i])
            if self._has_prev[aid]:
                self._accumulated_reward[aid] += float(rewards_np[i])

        # 2. Pre-build lookup for deciding agents: agent_idx → (state_key, mask_np)
        #    This avoids repeated searches in the inner loop.
        deciding_info: Dict[int, Tuple[Tuple, np.ndarray]] = {}
        if deciding.numel() > 0:
            dec_np = deciding.cpu().numpy()
            masks_np = masks.cpu().numpy()
            active_new_np = active_new.cpu().numpy()
            obs_active_np = obs_active.cpu().numpy()

            for j in range(len(dec_np)):
                aid = int(dec_np[j])
                mask_j = masks_np[j]
                # Find this agent's observation in obs_active (aligned with active_new)
                pos = np.where(active_new_np == aid)[0]
                if len(pos) > 0:
                    obs_j = obs_active_np[int(pos[0])]
                    state_next = self._discretise_obs(obs_j)
                    deciding_info[aid] = (state_next, mask_j)

        # 3. TD updates for agents that are done or newly deciding
        for i in range(len(prev_ids)):
            aid = int(prev_ids[i])
            if not self._has_prev[aid]:
                continue

            is_done = bool(done_mask[i])
            is_deciding = aid in deciding_info

            if not is_done and not is_deciding:
                continue

            s = self._prev_state[aid]
            a = int(self._prev_action[aid])
            q_table = self._get_q_table(aid)
            accum_r = self._accumulated_reward[aid]

            if is_done:
                target = accum_r
            else:
                # Use the next state's Q-values for the TD target
                s_next, mask_next = deciding_info[aid]
                q_next = q_table[s_next]
                valid_next = np.where(mask_next > 0)[0]
                if len(valid_next) > 0:
                    max_q_next = float(q_next[valid_next].max())
                else:
                    max_q_next = 0.0
                target = accum_r + self.gamma * max_q_next

            q_table[s][a] += self.alpha * (target - q_table[s][a])
            self.total_updates += 1

            # Reset accumulated reward
            self._accumulated_reward[aid] = 0.0

            if is_done:
                self._prev_state[aid] = None
                self._prev_action[aid] = -1
                self._has_prev[aid] = False

    # ── Action Selection (Dict API — legacy) ─────────────────────────

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        infos: Dict[str, dict],
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
            state = self._discretise_obs(obs)
            valid_indices = np.where(mask > 0)[0]
            q_table = self._get_q_table(agent_idx)

            # Epsilon-greedy over valid actions
            if self.rng.random() < self.epsilon:
                action = int(self.rng.choice(valid_indices))
            else:
                q_vals = q_table[state]
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

            s = self._prev_state[agent_idx]
            a = int(self._prev_action[agent_idx])
            q_table = self._get_q_table(agent_idx)
            accum_r = self._accumulated_reward[agent_idx]

            if done:
                target = accum_r
            else:
                obs_next = observations.get(agent_id)
                if obs_next is not None:
                    s_next = self._discretise_obs(obs_next)
                    q_next = q_table[s_next]
                    valid_next = np.where(mask_next > 0)[0]
                    max_q_next = float(q_next[valid_next].max())
                    target = accum_r + self.gamma * max_q_next
                else:
                    target = accum_r

            q_table[s][a] += self.alpha * (target - q_table[s][a])
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
        """Save all Q-tables to disk."""
        data = {
            "q_tables": {
                str(od): dict(qt) for od, qt in self.q_tables.items()
            },
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
        """Load Q-tables from disk."""
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
        for od_str, qt_dict in data["q_tables"].items():
            od_key = eval(od_str)  # Convert string back to tuple
            agent.q_tables[od_key] = defaultdict(
                lambda: np.zeros(agent.n_actions, dtype=np.float64),
                qt_dict,
            )
        return agent

    # ── Info ──────────────────────────────────────────────────────────

    @property
    def q_table_size(self) -> int:
        """Total number of (OD-pair, state) entries across all Q-tables."""
        return sum(len(qt) for qt in self.q_tables.values())

    @property
    def n_od_pairs(self) -> int:
        """Number of unique OD pairs with Q-tables."""
        return len(self.q_tables)

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(formulation={self.formulation}, "
            f"n_actions={self.n_actions}, α={self.alpha}, "
            f"γ={self.gamma}, ε={self.epsilon:.4f}, "
            f"OD_pairs={self.n_od_pairs}, "
            f"total_states={self.q_table_size}, updates={self.total_updates})"
        )
