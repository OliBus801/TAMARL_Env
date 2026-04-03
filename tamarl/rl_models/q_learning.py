"""Tabular Q-Learning agent for the DTA Markov Game environment.

Independent Q-Learning (IQL): each agent maintains its own Q-table keyed on
discretised local observations. Agents learn independently from their own
(state, action, reward, next_state) transitions.

State discretisation:
    (current_node, destination_node, congestion_level_per_outgoing_edge)
    where congestion_level is a binned version of the normalized occupancy.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def _make_q_table(n_actions: int):
    """Factory for a fresh per-agent Q-table (defaultdict)."""
    return defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))


class QLearningAgent:
    """Tabular Q-Learning with per-agent independent Q-tables.

    Every agent in the Markov Game has its **own** Q-table so that agents
    with different OD pairs learn distinct routing policies.

    Args:
        n_actions: number of discrete actions (= max_out_degree)
        alpha: learning rate
        gamma: discount factor
        epsilon_start: initial exploration rate
        epsilon_end: minimum exploration rate
        epsilon_decay: multiplicative decay per episode
        n_congestion_bins: number of bins for discretising edge occupancy
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        n_congestion_bins: int = 5,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_congestion_bins = n_congestion_bins
        self.rng = np.random.default_rng(seed)

        # Per-agent Q-tables: agent_id -> { state_key -> Q-values[n_actions] }
        self.q_tables: Dict[str, Dict[Tuple, np.ndarray]] = {}

        # Transition buffer: stores (state, action) per agent for TD update
        self._prev_state: Dict[str, Tuple] = {}
        self._prev_action: Dict[str, int] = {}
        self._accumulated_reward: Dict[str, float] = defaultdict(float)

        # Stats
        self.total_updates = 0

    def _get_q_table(self, agent_id: str):
        """Get or create the Q-table for a specific agent."""
        if agent_id not in self.q_tables:
            self.q_tables[agent_id] = defaultdict(
                lambda: np.zeros(self.n_actions, dtype=np.float64)
            )
        return self.q_tables[agent_id]

    # ── State Discretisation ─────────────────────────────────────────────

    def _discretise_obs(self, obs: np.ndarray) -> Tuple:
        """Convert a continuous observation vector to a discrete state key.

        obs layout: [current_node, destination, norm_time, occ_0..occ_k, ff_0..ff_k]
        We use: (current_node, destination, binned_occ_0, ..., binned_occ_k)
        """
        current_node = int(obs[0])
        destination = int(obs[1])

        # Occupancy features start at index 3, length = n_actions
        occ_start = 3
        occ_end = occ_start + self.n_actions
        occupancies = obs[occ_start:occ_end]

        # Bin occupancies into discrete levels
        binned = np.clip((occupancies * self.n_congestion_bins).astype(int), 0, self.n_congestion_bins - 1)

        return (current_node, destination, *binned.tolist())

    # ── Action Selection ─────────────────────────────────────────────────

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        infos: Dict[str, dict],
    ) -> Dict[str, int]:
        """Select actions using per-agent epsilon-greedy with action masking.

        Args:
            observations: agent_id → observation ndarray
            infos: agent_id → info dict with 'action_mask'

        Returns:
            Dict of agent_id → selected action index
        """
        actions = {}

        for agent_id, info in infos.items():
            mask = info.get("action_mask")
            if mask is None or mask.sum() == 0:
                continue

            obs = observations.get(agent_id)
            if obs is None:
                continue

            state = self._discretise_obs(obs)
            valid_indices = np.where(mask > 0)[0]
            q_table = self._get_q_table(agent_id)

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
            self._prev_state[agent_id] = state
            self._prev_action[agent_id] = action

        return actions

    # ── Learning ─────────────────────────────────────────────────────────

    def update(
        self,
        observations: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        infos: Dict[str, dict],
    ):
        """Perform per-agent Q-learning updates natively handling SMDP multi-step transitions.
        
        Rewards are accumulated while the agent travels. The Q-table is updated 
        only when the agent reaches the next decision event or terminates.
        """
        for agent_id, reward in rewards.items():
            if agent_id not in self._prev_state:
                continue

            # Accumulate reward over the link traversal
            self._accumulated_reward[agent_id] += reward

            done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
            info = infos.get(agent_id, {})
            mask_next = info.get("action_mask")
            is_deciding = mask_next is not None and mask_next.sum() > 0

            # Only perform TD update if the agent makes a new decision or is done
            if not done and not is_deciding:
                continue

            s = self._prev_state[agent_id]
            a = self._prev_action[agent_id]
            q_table = self._get_q_table(agent_id)
            accum_r = self._accumulated_reward[agent_id]

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

            # Q-learning update on this agent's own table
            q_table[s][a] += self.alpha * (target - q_table[s][a])
            self.total_updates += 1

            # Reset accumulated reward for the next link
            self._accumulated_reward[agent_id] = 0.0

            if done:
                del self._prev_state[agent_id]
                del self._prev_action[agent_id]
                del self._accumulated_reward[agent_id]

    # ── Epsilon Decay ────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str):
        """Save all Q-tables to disk."""
        data = {
            'q_tables': {aid: dict(qt) for aid, qt in self.q_tables.items()},
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'n_congestion_bins': self.n_congestion_bins,
            'total_updates': self.total_updates,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, seed: Optional[int] = None) -> 'QLearningAgent':
        """Load Q-tables from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        agent = cls(
            n_actions=data['n_actions'],
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon_start=data['epsilon_start'],
            epsilon_end=data['epsilon_end'],
            epsilon_decay=data['epsilon_decay'],
            n_congestion_bins=data['n_congestion_bins'],
            seed=seed,
        )
        agent.epsilon = data['epsilon']
        agent.total_updates = data['total_updates']
        for aid, qt_dict in data['q_tables'].items():
            agent.q_tables[aid] = defaultdict(
                lambda: np.zeros(agent.n_actions, dtype=np.float64),
                qt_dict,
            )
        return agent

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def q_table_size(self) -> int:
        """Total number of (agent, state) entries across all Q-tables."""
        return sum(len(qt) for qt in self.q_tables.values())

    @property
    def n_agents_with_tables(self) -> int:
        """Number of agents that have learned at least one state."""
        return len(self.q_tables)

    def __repr__(self) -> str:
        return (f"QLearningAgent(n_actions={self.n_actions}, α={self.alpha}, "
                f"γ={self.gamma}, ε={self.epsilon:.4f}, "
                f"agents={self.n_agents_with_tables}, "
                f"total_states={self.q_table_size}, updates={self.total_updates})")
